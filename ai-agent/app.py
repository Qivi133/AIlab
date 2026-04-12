from __future__ import annotations

import asyncio
import datetime
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import edge_tts
import requests
from flask import Flask, Response, jsonify, request, send_from_directory

from tools import get_stock_price_cn, send_email, send_dingtalk_message, STOCK_CODES

BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BASE_DIR / "public"
RESOURCES_DIR = BASE_DIR / "resources"


def load_env_files() -> None:
    for filename in (".env", ".env.local"):
        env_path = BASE_DIR / filename
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


load_env_files()

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "3000"))
EDGE_TTS_RATE = os.environ.get("EDGE_TTS_RATE", "+0%")
EDGE_TTS_VOLUME = os.environ.get("EDGE_TTS_VOLUME", "+0%")
EDGE_TTS_PITCH = os.environ.get("EDGE_TTS_PITCH", "+0Hz")

LANGUAGE_PROMPTS: Dict[str, str] = {
    "zh-CN": "你是一个自然、亲切、聪明的中文数字人助手。默认使用简体中文普通话口吻回答，表达口语化、简洁、有陪伴感。",
    "zh-HK": "你是一个自然、亲切、聪明的中文数字人助手。請盡量用粵語口吻回答，保持自然、易懂、友好。",
    "zh-TW": "你是一個自然、親切、聰明的繁體中文數字人助手。使用臺灣國語口吻回答，口語化、簡潔、有親和力。",
    "en-US": "You are a natural, friendly, and intelligent AI digital assistant. Respond in American English with a warm and conversational tone.",
    "en-GB": "You are a natural, friendly, and intelligent AI digital assistant. Respond in British English with a refined and warm tone.",
    "en-AU": "You are a natural, friendly, and intelligent AI digital assistant. Respond in Australian English with a cheerful and relaxed tone.",
    "ja-JP": "あなたは自然で、親切で、聡明なAIデジタルアシスタントです。自然な日本語で、やわらかく会話的な口調で回答してください。",
    "ko-KR": "당신은 자연스럽고 친절하며 똑똑한 AI 디지털 어시스턴트입니다. 자연스러운 한국어로 친절하게 답변해 주세요.",
    "fr-FR": "Vous êtes un assistant numérique IA naturel, bienveillant et intelligent. Répondez en français avec un ton chaleureux et conversationnel.",
    "de-DE": "Sie sind ein natürlicher, freundlicher und intelligenter KI-Digitalassistent. Antworten Sie auf Deutsch mit einem warmen und Gesprächston.",
    "es-ES": "Eres un asistente digital de IA natural, amigable e inteligente. Responde en español con un tono cálido y conversacional.",
    "pt-BR": "Você é um assistente digital de IA natural, amigável e inteligente. Responda em português brasileiro com um tom caloroso e conversacional.",
    "it-IT": "Sei un assistente digitale AI naturale, amichevole e intelligente. Rispondi in italiano con un tono caldo e conversazionale.",
    "ru-RU": "Вы естественный, дружелюбный и умный цифровой помощник ИИ. Отвечайте на русском языке с теплым и разговорным тоном.",
}

LANGUAGE_TTS_VOICES: Dict[str, str] = {
    "zh-CN": os.environ.get("EDGE_VOICE_ZH_CN", "zh-CN-XiaoxiaoNeural"),
    "zh-HK": os.environ.get("EDGE_VOICE_ZH_HK", "zh-HK-HiuGaaiNeural"),
    "zh-TW": os.environ.get("EDGE_VOICE_ZH_TW", "zh-TW-HsiaoYuNeural"),
    "en-US": os.environ.get("EDGE_VOICE_EN_US", "en-US-JennyNeural"),
    "en-GB": os.environ.get("EDGE_VOICE_EN_GB", "en-GB-SoniaNeural"),
    "en-AU": os.environ.get("EDGE_VOICE_EN_AU", "en-AU-NatashaNeural"),
    "ja-JP": os.environ.get("EDGE_VOICE_JA_JP", "ja-JP-NanamiNeural"),
    "ko-KR": os.environ.get("EDGE_VOICE_KO_KR", "ko-KR-SunHiNeural"),
    "fr-FR": os.environ.get("EDGE_VOICE_FR_FR", "fr-FR-DeniseNeural"),
    "de-DE": os.environ.get("EDGE_VOICE_DE_DE", "de-DE-KatjaNeural"),
    "es-ES": os.environ.get("EDGE_VOICE_ES_ES", "es-ES-ElviraNeural"),
    "pt-BR": os.environ.get("EDGE_VOICE_PT_BR", "pt-BR-FranciscaNeural"),
    "it-IT": os.environ.get("EDGE_VOICE_IT_IT", "it-IT-ElsaNeural"),
    "ru-RU": os.environ.get("EDGE_VOICE_RU_RU", "ru-RU-SvetlanaNeural"),
}

# 工具配置列表 - 供大模型调用
TOOLS_CONFIG: List[Dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price_cn",
            "description": "获取中国A股或港股的实时股价信息。当用户询问股票价格、涨跌、行情时，必须调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "stock_code": {
                        "type": "string",
                        "description": "股票代码，如：600519（贵州茅台）、000858（五粮液）、00700（腾讯控股）、002594（比亚迪）、601318（中国平安）"
                    }
                },
                "required": ["stock_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "通过SMTP协议发送邮件。当用户要求发送邮件时，必须调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "to_email": {
                        "type": "string",
                        "description": "收件人邮箱地址，如：example@qq.com"
                    },
                    "subject": {
                        "type": "string",
                        "description": "邮件主题，简短概括邮件内容"
                    },
                    "content": {
                        "type": "string",
                        "description": "邮件正文内容，必须包含用户要求发送的所有信息"
                    }
                },
                "required": ["to_email", "subject", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_dingtalk_message",
            "description": "通过钉钉机器人发送消息到群聊。当用户要求发送钉钉消息时，必须调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "要发送的消息内容"
                    }
                },
                "required": ["message"]
            }
        }
    }
]

app = Flask(__name__, static_folder=None)
TTS_STATUS_CACHE: Dict[str, Tuple[float, bool, str]] = {}
TTS_STATUS_TTL_SECONDS = 300


def strip_think_tags(text: str) -> str:
    return re.sub(
        r"<think>[\s\S]*?</think>", "", str(text or ""), flags=re.IGNORECASE
    ).strip()


async def synthesize_speech(text: str, voice: str) -> bytes:
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=EDGE_TTS_RATE,
        volume=EDGE_TTS_VOLUME,
        pitch=EDGE_TTS_PITCH,
    )
    chunks: List[bytes] = []
    async for chunk in communicate.stream():
        if chunk.get("type") == "audio":
            chunks.append(chunk["data"])
    return b"".join(chunks)


def check_tts_ready(language: str) -> Tuple[bool, str]:
    voice = LANGUAGE_TTS_VOICES.get(language, LANGUAGE_TTS_VOICES["zh-CN"])
    cached = TTS_STATUS_CACHE.get(voice)
    now = time.monotonic()
    if cached and now - cached[0] < TTS_STATUS_TTL_SECONDS:
        return cached[1], cached[2]

    try:
        audio = asyncio.run(synthesize_speech("test", voice))
        ready = bool(audio)
        message = "Edge-TTS 可用" if ready else "Edge-TTS 未返回音频数据"
    except Exception:
        app.logger.exception("Edge-TTS probe failed for voice %s", voice)
        ready = False
        message = "Edge-TTS 当前不可用"

    TTS_STATUS_CACHE[voice] = (now, ready, message)
    return ready, message


# 时间查询工具
def get_time_info() -> Dict[str, str]:
    """获取当前时间信息"""
    now = datetime.datetime.now()
    return {
        "current_time": now.strftime("%H:%M:%S"),
        "current_date": now.strftime("%Y年%m月%d日"),
        "weekday": ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][now.weekday()],
        "full_datetime": now.strftime("%Y年%m月%d日 %H:%M:%S"),
    }


def check_time_query(user_message: str) -> str | None:
    """检测用户消息是否询问时间，返回回答文本或None"""
    import sys
    import datetime
    
    print(f"[TOOL] check_time_query 被调用，参数: {user_message}", flush=True)
    sys.stdout.flush()
    
    message = user_message.lower().strip()
    
    # 检测时间相关关键词
    time_keywords = [
        "几点了", "现在几点", "现在的时间", "几点", "时间", 
        "今天星期", "今天周几", "星期几", "礼拜几",
        "今天日期", "今天几号", "几号", "日期",
        "年", "月", "日"
    ]
    
    # 简单关键词匹配
    is_time_query = any(keyword in message for keyword in time_keywords)
    
    if not is_time_query:
        return None
    
    # 获取时间信息
    now = datetime.datetime.now()
    weekday = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][now.weekday()]
    date = now.strftime("%Y年%m月%d日")
    clock = now.strftime("%H:%M:%S")
    
    # 根据具体问题生成回答
    if any(k in message for k in ["星期", "周几", "礼拜"]):
        return f"今天是{weekday}，{date}。"
    elif any(k in message for k in ["几号", "日期"]):
        return f"今天是{date}。"
    elif any(k in message for k in ["几点", "几点了", "时间"]):
        return f"现在是{clock}。"
    else:
        return f"现在是{date} {clock}，{weekday}。"


# 城市代码映射表
CITY_CODES = {
    "北京": "101010100", "北京天气": "101010100", "北京今天天气": "101010100",
    "上海": "101020100", "上海天气": "101020100", "上海今天天气": "101020100",
    "广州": "101280101", "广州天气": "101280101", "广州今天天气": "101280101",
    "深圳": "101280601", "深圳天气": "101280601", "深圳今天天气": "101280601",
    "杭州": "101210101", "杭州天气": "101210101", "杭州今天天气": "101210101",
    "南京": "101190101", "南京天气": "101190101", "南京今天天气": "101190101",
    "成都": "101270101", "成都天气": "101270101", "成都今天天气": "101270101",
    "重庆": "101040100", "重庆天气": "101040100", "重庆今天天气": "101040100",
    "武汉": "101200101", "武汉天气": "101200101", "武汉今天天气": "101200101",
    "西安": "101110101", "西安天气": "101110101", "西安今天天气": "101110101",
    "天津": "101030100", "天津天气": "101030100", "天津今天天气": "101030100",
    "苏州": "101190401", "苏州天气": "101190401", "苏州今天天气": "101190401",
    "郑州": "101180101", "郑州天气": "101180101", "郑州今天天气": "101180101",
    "长沙": "101250101", "长沙天气": "101250101", "长沙今天天气": "101250101",
    "沈阳": "101070101", "沈阳天气": "101070101", "沈阳今天天气": "101070101",
    "青岛": "101120201", "青岛天气": "101120201", "青岛今天天气": "101120201",
    "大连": "101070201", "大连天气": "101070201", "大连今天天气": "101070201",
    "厦门": "101230201", "厦门天气": "101230201", "厦门今天天气": "101230201",
    "宁波": "101210401", "宁波天气": "101210401", "宁波今天天气": "101210401",
    "东莞": "101281601", "东莞天气": "101281601", "东莞今天天气": "101281601",
    "佛山": "101280800", "佛山天气": "101280800", "佛山今天天气": "101280800",
}


def get_weather_info(city_code: str) -> dict | None:
    """调用天气API获取天气信息"""
    try:
        url = f"http://t.weather.sojson.com/api/weather/city/{city_code}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == 200:
                return data
    except Exception:
        pass
    return None


def check_weather_query(user_message: str) -> str | None:
    """检测用户消息是否询问天气，返回回答文本或None"""
    message = user_message.lower().strip()
    
    # 检测天气相关关键词
    weather_keywords = ["天气", "多少度", "温度", "下雨", "晴天", "阴天", "风", "空气质量", "pm25", "pm2.5"]
    
    if not any(kw in message for kw in weather_keywords):
        return None
    
    # 检测城市
    city = None
    for city_name in CITY_CODES:
        if city_name in message:
            city = city_name
            break
    
    # 如果没找到具体城市，尝试提取
    if not city:
        # 常见城市匹配
        common_cities = ["北京", "上海", "广州", "深圳", "杭州", "南京", "成都", "重庆", "武汉", "西安"]
        for c in common_cities:
            if c in message:
                city = c
                break
    
    if not city:
        return "请问您想查询哪个城市的天气呢？"
    
    # 获取城市代码
    city_code = CITY_CODES.get(city)
    if not city_code:
        # 尝试模糊匹配
        for key, code in CITY_CODES.items():
            if city in key:
                city_code = code
                break
    
    if not city_code:
        return f"抱歉，暂时不支持查询{city}的天气。"
    
    # 获取天气信息
    weather_data = get_weather_info(city_code)
    if not weather_data:
        return f"抱歉，无法获取{city}的天气信息，请稍后再试。"
    
    # 解析天气数据
    data = weather_data.get("data", {})
    city_info = weather_data.get("cityInfo", {})
    
    wendu = data.get("wendu", "未知")  # 温度
    quality = data.get("quality", "未知")  # 空气质量
    shidu = data.get("shidu", "未知")   # 湿度
    ganmao = data.get("ganmao", "")    # 感冒提示
    
    # 获取今天天气
    forecast = data.get("forecast", [])
    today = forecast[0] if forecast else {}
    type_ = today.get("type", "未知")   # 天气类型（从forecast获取）
    high = today.get("high", "")
    low = today.get("low", "")
    fx = today.get("fx", "")  # 风向
    fl = today.get("fl", "")  # 风力
    notice = today.get("notice", "")  # 提示
    
    # 构造回答
    answer = f"{city_info.get('city', city)}今天天气：\n"
    answer += f"🌡️ 温度：{wendu}°C ({low} ~ {high})\n"
    answer += f"🌤️ 天气：{type_}\n"
    answer += f"💨 风向：{fx} {fl}\n"
    answer += f"💧 湿度：{shidu}\n"
    answer += f"🌬️ 空气质量：{quality}\n"
    if notice:
        answer += f"📝 提示：{notice}\n"
    if ganmao:
        answer += f"🤒 {ganmao}"

    return answer


@app.post("/api/chat")
def api_chat() -> Response:
    try:
        body = request.get_json(silent=True) or {}
        model = body.get("model") or "qwen3:8b"
        language = body.get("language") or "zh-CN"
        messages = (
            body.get("messages") if isinstance(body.get("messages"), list) else []
        )

        print(f"[DEBUG] body: {body}")
        print(f"[DEBUG] messages: {messages}")
        
        # 检查是否是时间查询（获取最后一条用户消息）
        user_message = ""
        if messages and messages[-1].get("role") == "user":
            user_message = messages[-1].get("content", "")
        if not user_message:
            user_message = body.get("message", "") or body.get("content", "") or body.get("prompt", "")
        
        print(f"[DEBUG] 收到消息: {user_message}")
        print(f"[DEBUG] 调用 check_time_query 函数...")
        
        # 时间查询工具
        try:
            time_answer = check_time_query(user_message)
        except Exception as e:
            print(f"[ERROR] check_time_query 异常: {e}")
            time_answer = None
            
        print(f"[DEBUG] 时间查询结果: {time_answer}")
        
        # 时间查询工具优先
        if time_answer:
            return jsonify({
                "message": time_answer,
                "model": "时间工具",
                "is_tool_response": True
            })
        
        # 天气查询工具
        try:
            weather_answer = check_weather_query(user_message)
        except Exception as e:
            print(f"[ERROR] check_weather_query 异常: {e}")
            weather_answer = None

        print(f"[DEBUG] 天气查询结果: {weather_answer}")

        if weather_answer:
            return jsonify({
                "message": weather_answer,
                "model": "天气工具",
                "is_tool_response": True
            })

        system_prompt = (
            f"{LANGUAGE_PROMPTS.get(language, LANGUAGE_PROMPTS['zh-CN'])}\n"
            '你现在是一个智能助手，可以通过工具获取实时信息、发送邮件和钉钉消息。\n'
            '可用工具：\n'
            '1. get_stock_price_cn - 查询股票实时行情\n'
            '   当用户询问股价、股票涨跌、行情时，必须调用此工具。\n'
            '   股票代码参考：600519（贵州茅台）、000858（五粮液）、00700（腾讯控股）、002594（比亚迪）、601318（中国平安）\n'
            '2. send_email - 发送邮件\n'
            '   当用户要求发送邮件时，必须调用此工具。\n'
            '   参数：to_email（收件人邮箱）、subject（邮件主题）、content（邮件正文）\n'
            '3. send_dingtalk_message - 发送钉钉消息\n'
            '   当用户要求发送钉钉消息时，必须调用此工具。\n'
            '   参数：message（消息内容）\n'
            '要求：\n'
            '1. 如果用户询问股票相关问题，先调用 get_stock_price_cn 工具获取数据。\n'
            '2. 如果用户要求发送邮件，调用 send_email 工具。\n'
            '3. 如果用户要求发送钉钉消息，调用 send_dingtalk_message 工具。\n'
            '4. 根据工具返回的数据，用自然语言回答用户。\n'
            '5. 回答简洁、口语化。'
        )

        payload = {
            "model": model,
            "stream": False,
            "think": False,
            "messages": [{"role": "system", "content": system_prompt}, *messages],
            "options": {"temperature": 0.8, "top_p": 0.9},
            "tools": TOOLS_CONFIG,
        }

        try:
            ollama_response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        except requests.RequestException:
            app.logger.exception("Failed to reach Ollama at %s", OLLAMA_URL)
            return (
                jsonify(
                    {
                        "error": "OLLAMA_UNAVAILABLE",
                        "message": "无法连接到 Ollama 服务。",
                    }
                ),
                502,
            )

        if not ollama_response.ok:
            return jsonify(
                {
                    "error": "OLLAMA_ERROR",
                    "message": ollama_response.text or "Ollama request failed.",
                }
            ), 502

        try:
            data = ollama_response.json()
        except ValueError:
            app.logger.exception("Ollama returned non-JSON response")
            return (
                jsonify(
                    {
                        "error": "OLLAMA_BAD_RESPONSE",
                        "message": "Ollama 返回了无法解析的数据。",
                    }
                ),
                502,
            )

        # 处理工具调用
        message_content = data.get("message", {})
        tool_calls = message_content.get("tool_calls", [])

        if tool_calls:
            # 处理股票查询工具调用
            for tool_call in tool_calls:
                func = tool_call.get("function", {})
                func_name = func.get("name")
                arguments = func.get("arguments", {})

                if func_name == "get_stock_price_cn":
                    stock_code = arguments.get("stock_code", "")
                    stock_data = get_stock_price_cn(stock_code)

                    # 将工具结果作为新消息添加到上下文
                    tool_result_msg = {
                        "role": "tool",
                        "content": str(stock_data)
                    }
                    messages.append(message_content)  # 添加大模型的tool call消息
                    messages.append(tool_result_msg)  # 添加工具执行结果

                    # 重新调用大模型，让它根据工具结果生成最终回答
                    payload["messages"] = [{"role": "system", "content": system_prompt}, *messages]

                    try:
                        final_response = requests.post(OLLAMA_URL, json=payload, timeout=120)
                        final_data = final_response.json()
                        return jsonify({
                            "message": strip_think_tags(final_data.get("message", {}).get("content", "")),
                            "model": final_data.get("model") or model,
                        })
                    except Exception:
                        return jsonify({
                            "message": f"股票查询结果：{stock_data}",
                            "model": "股票工具",
                        })

                elif func_name == "send_email":
                    to_email = arguments.get("to_email", "")
                    subject = arguments.get("subject", "")
                    content = arguments.get("content", "")

                    email_result = send_email(to_email, subject, content)

                    # 将工具结果作为新消息添加到上下文
                    tool_result_msg = {
                        "role": "tool",
                        "content": str(email_result)
                    }
                    messages.append(message_content)
                    messages.append(tool_result_msg)

                    # 重新调用大模型，让它根据工具结果生成最终回答
                    payload["messages"] = [{"role": "system", "content": system_prompt}, *messages]

                    try:
                        final_response = requests.post(OLLAMA_URL, json=payload, timeout=120)
                        final_data = final_response.json()
                        return jsonify({
                            "message": strip_think_tags(final_data.get("message", {}).get("content", "")),
                            "model": final_data.get("model") or model,
                        })
                    except Exception:
                        if email_result.get("success"):
                            return jsonify({
                                "message": f"邮件已成功发送给 {to_email}，发送时间：{email_result.get('sent_time')}",
                                "model": "邮件工具",
                            })
                        else:
                            return jsonify({
                                "message": f"邮件发送失败：{email_result.get('error')}",
                                "model": "邮件工具",
                            })

                elif func_name == "send_dingtalk_message":
                    message = arguments.get("message", "")
                    webhook_url = os.environ.get("DINGTALK_WEBHOOK", "")

                    dingtalk_result = send_dingtalk_message(webhook_url, message)

                    # 将工具结果作为新消息添加到上下文
                    tool_result_msg = {
                        "role": "tool",
                        "content": str(dingtalk_result)
                    }
                    messages.append(message_content)
                    messages.append(tool_result_msg)

                    # 重新调用大模型，让它根据工具结果生成最终回答
                    payload["messages"] = [{"role": "system", "content": system_prompt}, *messages]

                    try:
                        final_response = requests.post(OLLAMA_URL, json=payload, timeout=120)
                        final_data = final_response.json()
                        return jsonify({
                            "message": strip_think_tags(final_data.get("message", {}).get("content", "")),
                            "model": final_data.get("model") or model,
                        })
                    except Exception:
                        if dingtalk_result.get("success"):
                            return jsonify({
                                "message": f"钉钉消息已成功发送，发送时间：{dingtalk_result.get('sent_time')}",
                                "model": "钉钉工具",
                            })
                        else:
                            return jsonify({
                                "message": f"钉钉消息发送失败：{dingtalk_result.get('error')}",
                                "model": "钉钉工具",
                            })

        return jsonify(
            {
                "message": strip_think_tags(message_content.get("content", "")),
                "model": data.get("model") or model,
            }
        )
    except Exception:
        app.logger.exception("Unexpected server error in /api/chat")
        return jsonify({"error": "SERVER_ERROR", "message": "服务暂时不可用。"}), 500


@app.post("/api/tts")
def api_tts() -> Response:
    try:
        body = request.get_json(silent=True) or {}
        text = str(body.get("text") or "").strip()
        language = body.get("language") or "zh-CN"
        if not text:
            return jsonify({"error": "EMPTY_TEXT", "message": "text is required"}), 400

        voice = LANGUAGE_TTS_VOICES.get(language, LANGUAGE_TTS_VOICES["zh-CN"])
        audio_bytes = asyncio.run(synthesize_speech(text, voice))
        return Response(
            audio_bytes, mimetype="audio/mpeg", headers={"Cache-Control": "no-store"}
        )
    except Exception:
        app.logger.exception("TTS request failed")
        return (
            jsonify(
                {
                    "error": "TTS_UPSTREAM_ERROR",
                    "message": "语音服务当前不可用。",
                }
            ),
            502,
        )


@app.get("/api/tts-status")
def api_tts_status() -> Response:
    language = request.args.get("language") or "zh-CN"
    voice = LANGUAGE_TTS_VOICES.get(language, LANGUAGE_TTS_VOICES["zh-CN"])
    ready, message = check_tts_ready(language)
    return jsonify(
        {
            "provider": "edge-tts",
            "ready": ready,
            "voice": voice,
            "message": message,
            "languages": list(LANGUAGE_TTS_VOICES.keys()),
        }
    )


@app.get("/")
def index() -> Response:
    return send_from_directory(PUBLIC_DIR, "index.html")


@app.get("/resources/<path:filename>")
def resources(filename: str) -> Response:
    return send_from_directory(RESOURCES_DIR, filename)


@app.get("/<path:filename>")
def public_files(filename: str) -> Response:
    return send_from_directory(PUBLIC_DIR, filename)


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=False)
