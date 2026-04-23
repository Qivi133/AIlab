import os
import json
import re
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr

import requests
import edge_tts
from flask import Flask, request, jsonify, Response
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import JsonOutputParser

app = Flask(__name__, static_folder="public", static_url_path="")

OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:0.6b")
DINGTALK_WEBHOOK = os.environ.get("DINGTALK_WEBHOOK", "")
EDGE_TTS_VOICE = os.environ.get("EDGE_TTS_VOICE", "zh-CN-XiaoxiaoNeural")
EDGE_TTS_RATE = os.environ.get("EDGE_TTS_RATE", "+0%")
EDGE_TTS_VOLUME = os.environ.get("EDGE_TTS_VOLUME", "+0%")
EDGE_TTS_PITCH = os.environ.get("EDGE_TTS_PITCH", "+0Hz")

LANGUAGE_PROMPTS = {
    "zh-CN": "你是一个友善的本地智能聊天数字人助手。",
    "en-US": "You are a friendly local AI chat digital human assistant.",
}

# ==================== 步骤1：使用@tool装饰器定义工具 ====================

CITY_CODE_MAP = {
    "北京": "101010100",
    "上海": "101020100",
    "广州": "101280101",
    "深圳": "101280601",
    "杭州": "101210101",
    "重庆": "101040100",
    "成都": "101270101",
    "武汉": "101200101",
}


@tool
def get_time() -> str:
    """获取当前时间信息

    当用户询问当前时间、今天日期、现在是几点时调用此工具。

    Returns:
        包含当前时间的字典，包括小时、分钟、秒、日期、星期等信息
    """
    now = datetime.now()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return json.dumps({
        "current_time": now.strftime("%H:%M:%S"),
        "current_date": now.strftime("%Y-%m-%d"),
        "weekday": now.strftime("%A"),
        "weekday_cn": weekdays[now.weekday()],
        "full_datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
    }, ensure_ascii=False)


@tool
def get_weather(city_name: str) -> str:
    """查询指定城市的天气信息

    当用户询问某个城市的天气、气温、是否下雨等情况时调用此工具。
    必须传入正确的城市名称，如"北京"、"上海"、"广州"等。

    Args:
        city_name: 城市名称，必须是中文，如"北京"、"广州"等

    Returns:
        包含城市天气信息的字典，包括温度、湿度、天气状况等
    """
    city_name = (city_name or "").strip()
    if not city_name:
        return json.dumps({"code": 400, "msg": "城市名称不能为空", "data": None})

    if city_name not in CITY_CODE_MAP:
        return json.dumps({"code": 404, "msg": f"未找到城市：{city_name}", "data": None})

    try:
        response = requests.get(
            f"http://t.weather.sojson.com/api/weather/city/{CITY_CODE_MAP[city_name]}",
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()
        return json.dumps({
            "code": 200,
            "msg": "查询成功",
            "data": {
                "city": data["cityInfo"]["city"],
                "temperature": data["data"]["wendu"],
                "humidity": data["data"]["shidu"],
                "today_weather": data["data"]["forecast"][0]["type"],
                "low_temperature": data["data"]["forecast"][0]["low"],
                "high_temperature": data["data"]["forecast"][0]["high"],
            },
        }, ensure_ascii=False)
    except Exception as error:
        return json.dumps({"code": 500, "msg": f"天气查询失败：{error}", "data": None})


@tool
def get_stock_price(ticker: str) -> str:
    """查询中国A股股票的实时价格

    当用户询问股票价格、股价、行情、涨跌时调用此工具。
    传入中国A股股票代码，如"000001"（平安银行）、"600000"（浦发银行）等。

    Args:
        ticker: 股票代码，6位数字，如"000001"、"600036"等

    Returns:
        包含股票价格信息的字典，包括当前价、开盘价、昨日收盘价、最高价、最低价、涨跌幅等
    """
    ticker = str(ticker or "").strip()
    if not ticker:
        return json.dumps({"code": 400, "msg": "股票代码不能为空", "data": None})

    try:
        if ticker.startswith("6"):
            code = f"sh{ticker}"
        else:
            code = f"sz{ticker}"

        url = f"https://hq.sinajs.cn/list={code}"
        headers = {"Referer": "https://finance.sina.com/"}
        res = requests.get(url, headers=headers, timeout=5)
        res.raise_for_status()

        data = res.text.split('"')[1].split(',')

        if len(data) < 3:
            return json.dumps({"code": 404, "msg": f"未找到股票：{ticker}", "data": None})

        return json.dumps({
            "code": 200,
            "msg": "查询成功",
            "data": {
                "name": data[0],
                "ticker": ticker,
                "current_price": float(data[3]),
                "open": float(data[1]),
                "last_close": float(data[2]),
                "high": float(data[4]),
                "low": float(data[5]),
                "change_percent": round((float(data[3]) - float(data[2])) / float(data[2]) * 100, 2),
            },
        }, ensure_ascii=False)
    except Exception as error:
        return json.dumps({"code": 500, "msg": f"股价查询失败：{error}", "data": None})


@tool
def send_email_tool(to_email: str, subject: str, body: str) -> str:
    """发送电子邮件

    当用户要求发送邮件、发送电子邮件时调用此工具。
    需要提供收件人邮箱地址、邮件主题和邮件正文。

    Args:
        to_email: 收件人邮箱地址，如"user@example.com"
        subject: 邮件主题，不能为空
        body: 邮件正文内容，不能为空

    Returns:
        包含发送结果的字典，成功时包含发送时间，失败时包含错误信息
    """
    if not re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", to_email or ""):
        return json.dumps({"code": 400, "msg": f"邮箱格式不合法：{to_email}", "data": None})

    if not subject or not body:
        return json.dumps({"code": 400, "msg": "邮件主题和正文不能为空", "data": None})

    smtp_server = os.environ.get("SMTP_SERVER", "smtp.163.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "465"))
    smtp_username = os.environ.get("SMTP_USERNAME", "")
    smtp_password = os.environ.get("SMTP_PASSWORD", "")
    from_email = os.environ.get("FROM_EMAIL", "")
    from_name = os.environ.get("FROM_NAME", "AI Agent")

    if not smtp_username or not smtp_password or not from_email:
        return json.dumps({"code": 500, "msg": "邮件服务配置不完整", "data": None})

    try:
        message = MIMEMultipart()
        message["From"] = formataddr((from_name, from_email))
        message["To"] = to_email
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain", "utf-8"))

        server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        server.login(smtp_username, smtp_password)
        server.sendmail(from_email, [to_email], message.as_string())
        server.quit()

        return json.dumps({
            "code": 200,
            "msg": "邮件发送成功",
            "data": {
                "to_email": to_email,
                "subject": subject,
                "sent_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        }, ensure_ascii=False)
    except Exception as error:
        return json.dumps({"code": 500, "msg": f"邮件发送失败：{error}", "data": None})


@tool
def send_dingtalk(content: str) -> str:
    """发送钉钉群消息

    当用户要求发送钉钉消息、发送到钉钉群时调用此工具。
    需要提供要发送的具体消息内容。

    Args:
        content: 要发送到钉钉群的消息内容，不能为空

    Returns:
        包含发送结果的字典，成功时包含发送时间，失败时包含错误信息
    """
    webhook_url = DINGTALK_WEBHOOK
    if not webhook_url:
        return json.dumps({"code": 500, "msg": "钉钉Webhook未配置", "data": None})

    content = (content or "").strip()
    if not content:
        return json.dumps({"code": 400, "msg": "消息内容不能为空", "data": None})

    # 清理内容前缀
    for separator in ("：", ":"):
        if separator in content and content.index(separator) < 10:
            content = content.split(separator, 1)[1].strip()

    if not content:
        return json.dumps({"code": 400, "msg": "消息内容不能为空", "data": None})

    # 添加发送者标识
    if not content.startswith("AI Agent："):
        content = f"AI Agent：{content}"

    try:
        response = requests.post(
            webhook_url,
            json={
                "msgtype": "text",
                "text": {"content": content},
                "at": {"atMobiles": [], "isAtAll": False},
            },
            timeout=10,
        )
        response.raise_for_status()
        result = response.json()

        if result.get("errcode") == 0:
            return json.dumps({
                "code": 200,
                "msg": "钉钉消息发送成功",
                "data": {"sent_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
            }, ensure_ascii=False)

        return json.dumps({
            "code": 500,
            "msg": f"钉钉消息发送失败：{result.get('errmsg', '未知错误')}",
            "data": result,
        }, ensure_ascii=False)
    except Exception as error:
        return json.dumps({"code": 500, "msg": f"钉钉消息发送失败：{error}", "data": None})


# ==================== 步骤2：创建工具列表并绑定到模型 ====================

TOOLS = [get_time, get_weather, get_stock_price, send_email_tool, send_dingtalk]

# 初始化 ChatOllama 模型
llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_URL,
    temperature=0.2,
    top_p=0.9,
)

# 绑定工具到模型
llm_with_tools = llm.bind_tools(TOOLS)


# ==================== 步骤3：构建 LCEL Chain ====================

def make_system_prompt(language: str) -> str:
    """生成系统提示词"""
    base = LANGUAGE_PROMPTS.get(language, LANGUAGE_PROMPTS["zh-CN"])
    return (
        f"{base}\n"
        "你现在以 '本地智能聊天数字人' 的身份与用户对话。要求：\n"
        "回答自然，不要机械列条目，除非用户明确需要。\n"
        "默认简洁，适当体现情绪和陪伴感。\n"
        "如果用户在做口语聊天，就像真人一样接话。\n"
        "用户询问时间时必须调用 get_time 工具。\n"
        "用户询问天气时必须调用 get_weather 工具，传入城市名称。\n"
        "用户询问股票价格时必须调用 get_stock_price 工具，传入股票代码。\n"
        "用户要求发送邮件时必须调用 send_email_tool 工具。\n"
        "用户要求发送钉钉消息时必须调用 send_dingtalk 工具。\n"
        "所有工具调用结果会直接返回给你，你可以基于结果回答用户。"
    )


# 构建 Chain：提示词 -> 模型 -> 工具调用 -> 结果解析
def build_chain(language: str, messages: list) -> str:
    """构建并执行 LCEL Chain"""
    system_msg = SystemMessage(content=make_system_prompt(language))

    # 将历史消息转换为 HumanMessage
    langchain_messages = [system_msg]
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            langchain_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            langchain_messages.append(SystemMessage(content=content))

    # 调用模型（会自动触发工具调用）
    response = llm_with_tools.invoke(langchain_messages)

    # 如果有工具调用，执行工具
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})

            # 根据工具名称选择工具
            if tool_name == "get_time":
                tool_result = get_time.invoke({})
            elif tool_name == "get_weather":
                tool_result = get_weather.invoke(tool_args)
            elif tool_name == "get_stock_price":
                tool_result = get_stock_price.invoke(tool_args)
            elif tool_name == "send_email_tool":
                tool_result = send_email_tool.invoke(tool_args)
            elif tool_name == "send_dingtalk":
                tool_result = send_dingtalk.invoke(tool_args)
            else:
                tool_result = json.dumps({"error": f"未知工具：{tool_name}"})

            # 将工具结果添加回消息列表
            langchain_messages.append(response)
            langchain_messages.append(HumanMessage(content=f"[工具结果]: {tool_result}"))

            # 再次调用模型获取最终回复
            final_response = llm.invoke(langchain_messages)
            return final_response.content

    return response.content


# ==================== 步骤4：接入 Memory（会话记忆）====================

# 存储会话历史
store = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """获取或创建会话历史"""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# 创建带记忆的 Chain
chain_with_memory = RunnableWithMessageHistory(
    runnable=llm_with_tools,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


# ==================== 步骤5：添加输出解析 ====================

# 使用 JsonOutputParser 解析输出
json_parser = JsonOutputParser()


# ==================== Flask 路由 ====================

@app.route("/")
def index() -> Response:
    return app.send_static_file("index.html")


def get_latest_user_message(messages: list[dict]) -> str:
    """获取最新用户消息"""
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", "")).strip()
    return ""


def extract_dingtalk_content(text: str) -> str:
    """提取钉钉消息内容"""
    cleaned = (text or "").strip()
    for separator in ("：", ":"):
        if separator in cleaned:
            _, content = cleaned.split(separator, 1)
            content = content.strip()
            if content:
                return content
    return cleaned


def should_send_direct_dingtalk(message: str) -> bool:
    """判断是否直接发送钉钉"""
    lowered = message.lower()
    return "钉钉" in message or "dingtalk" in lowered


def should_block_dingtalk(content: str) -> bool:
    """判断是否阻止钉钉发送"""
    cleaned = content.strip()
    generic_commands = {"发钉钉", "发送钉钉", "发个钉钉", "发送钉钉消息", "发钉钉消息", "dingtalk"}
    return not cleaned or cleaned.lower() in generic_commands


@app.post("/api/chat")
def api_chat() -> Response:
    """聊天 API 端点"""
    try:
        body = request.get_json(silent=True) or {}
        model = body.get("model") or OLLAMA_MODEL
        language = body.get("language") or "zh-CN"
        messages = body.get("messages") if isinstance(body.get("messages"), list) else []
        session_id = body.get("session_id", "default")

        # 处理直接钉钉发送
        user_message = get_latest_user_message(messages)
        if should_send_direct_dingtalk(user_message):
            content = extract_dingtalk_content(user_message)
            if should_block_dingtalk(content):
                return jsonify({"reply": "请告诉我你要发送到钉钉群的具体内容，我确认后再发送。"})

            result = send_dingtalk.invoke({"content": content})
            try:
                result_dict = json.loads(result)
                if result_dict.get("code") == 200:
                    return jsonify({"reply": "钉钉消息已发送。"})
                return jsonify({"reply": f"钉钉消息发送失败：{result_dict.get('msg', '未知错误')}"})
            except json.JSONDecodeError:
                return jsonify({"reply": f"钉钉消息发送失败：{result}"})

        # 使用 LangChain Chain 处理对话
        reply = build_chain(language, messages)

        return jsonify({"reply": reply})

    except Exception as error:
        return jsonify({"reply": f"处理失败：{str(error)}"})


async def generate_speech(text: str, voice: str) -> bytes:
    """生成语音"""
    communicate = edge_tts.Communicate(
        text,
        voice,
        rate=EDGE_TTS_RATE,
        volume=EDGE_TTS_VOLUME,
        pitch=EDGE_TTS_PITCH,
    )
    chunks = []
    async for chunk in communicate.stream():
        if chunk.get("type") == "audio":
            chunks.append(chunk["data"])
    return b"".join(chunks)


@app.post("/api/tts")
async def api_tts() -> Response:
    """TTS API 端点"""
    try:
        body = request.get_json(silent=True) or {}
        text = body.get("text", "")
        voice = body.get("voice") or EDGE_TTS_VOICE

        if not text:
            return jsonify({"error": "文本不能为空"})

        audio_data = await generate_speech(text, voice)
        return Response(audio_data, mimetype="audio/mpeg")

    except Exception as error:
        return jsonify({"error": str(error)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
