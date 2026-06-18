import asyncio
import json
import os
import re
from pathlib import Path
from typing import TypedDict

import edge_tts
from flask import Flask, Response, jsonify, request, send_from_directory
from dotenv import load_dotenv

# ── LangChain 核心导入 ──────────────────────────────────────────
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langgraph.graph import END, START, StateGraph

# ── RAG 相关导入 ─────────────────────────────────────────────────
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import OllamaEmbeddings

# ── 工具函数 ────────────────────────────
from tools import (
    get_current_time,
    get_stock_price_cn,
    get_weather_json,
    send_dingtalk_message,
    send_email,
)

load_dotenv()

# ── 全局配置 ────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BASE_DIR / "public"
RESOURCES_DIR = BASE_DIR / "resources"

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "3000"))
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DINGTALK_WEBHOOK = os.environ.get("DINGTALK_WEBHOOK", "")
EDGE_TTS_RATE = os.environ.get("EDGE_TTS_RATE", "+0%")
EDGE_TTS_VOLUME = os.environ.get("EDGE_TTS_VOLUME", "+0%")
EDGE_TTS_PITCH = os.environ.get("EDGE_TTS_PITCH", "+0Hz")

LANGUAGE_PROMPTS = {
    "zh-CN": "你是一个自然、亲切、聪明的中文数字人助手。默认使用简体中文普通话回答，表达口语化、简洁。",
    "zh-HK": "你是一个自然、亲切、聪明的中文数字人助手。请尽量使用粤语口吻回答，保持自然、易懂、友好。",
    "zh-TW": "你是一个自然、亲切、聪明的繁体中文数位助理。请使用台湾常用的繁体中文，口吻自然、清楚、友善。",
    "en-US": "You are a natural, friendly, and intelligent AI digital assistant. Respond in American English with a warm and conversational tone.",
    "en-GB": "You are a natural, friendly, and intelligent AI digital assistant. Respond in British English with a refined and warm tone.",
    "en-AU": "You are a natural, friendly, and intelligent AI digital assistant. Respond in Australian English with a cheerful and relaxed tone.",
    "ja-JP": "あなたは自然で親しみやすく、賢いAIデジタルアシスタントです。日本語で、やわらかく会話的な口調で答えてください。",
    "ko-KR": "당신은 자연스럽고 친절하며 똑똑한 AI 디지털 어시스턴트입니다. 한국어로 따뜻하고 대화체에 가깝게 답변하세요.",
    "fr-FR": "Vous etes un assistant numerique IA naturel, bienveillant et intelligent. Repondez en francais avec un ton chaleureux et conversationnel.",
    "de-DE": "Sie sind ein natuerlicher, freundlicher und intelligenter KI-Digitalassistent. Antworten Sie auf Deutsch mit einem warmen und gespraechsnahen Ton.",
    "es-ES": "Eres un asistente digital de IA natural, amigable e inteligente. Responde en espanol con un tono calido y conversacional.",
    "pt-BR": "Voce e um assistente digital de IA natural, amigavel e inteligente. Responda em portugues brasileiro com um tom caloroso e conversacional.",
    "it-IT": "Sei un assistente digitale AI naturale, amichevole e intelligente. Rispondi in italiano con un tono caldo e conversazionale.",
    "ru-RU": "Вы естественный, дружелюбный и умный ИИ-ассистент. Отвечайте по-русски теплым и разговорным тоном.",
}

LANGUAGE_TTS_VOICES = {
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

# ════════════════════════════════════════════════════════════════
# RAG 课程知识库初始化（基于《智能应用系统设计》课程介绍.md）
# ════════════════════════════════════════════════════════════════

COURSE_MD_PATH = BASE_DIR / "《智能应用系统设计》课程介绍.md"

RAG_chain = None


def init_rag_chain():
    """初始化 RAG 链，加载课程知识库。"""
    global RAG_chain

    if not COURSE_MD_PATH.exists():
        print(f"[RAG] 课程介绍文件不存在: {COURSE_MD_PATH}")
        return None

    md_text = COURSE_MD_PATH.read_text(encoding="utf-8")

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ],
        strip_headers=False
    )
    header_splits = header_splitter.split_text(md_text)

    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "，", " ", ""]
    )
    splits = char_splitter.split_documents(header_splits)

    embeddings = OllamaEmbeddings(
        model="qwen3-embedding:0.6b",
        base_url=OLLAMA_BASE_URL
    )
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 6

    from langchain_classic.retrievers import EnsembleRetriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )

    def format_docs(docs):
        return "\n\n---\n\n".join(
            f"[来源：{doc.metadata}]\n{doc.page_content}" for doc in docs
        )

    rag_prompt = ChatPromptTemplate.from_template("""
你是一个课程信息助手。请严格根据下方【参考资料】回答用户问题。

规则：
- 只能使用【参考资料】中出现的信息
- 如果资料中没有明确答案，请回答"根据已有资料无法确认"
- 不要编造或推测任何数字、名称

【参考资料】
{context}

【用户问题】
{question}

【回答】""")

    from langchain_ollama import OllamaEmbeddings as OllamaEmbeddingsForLLM
    from langchain_core.output_parsers import StrOutputParser

    llm_for_rag = ChatOllama(
        model="qwen3:0.6b",
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )

    RAG_chain = (
        {
            "context": ensemble_retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | rag_prompt
        | llm_for_rag
        | StrOutputParser()
    )

    print(f"[RAG] 课程知识库加载完成，共 {len(splits)} 个文档块")
    return RAG_chain


def rag_answer(question: str) -> str:
    """使用 RAG 链回答关于课程的问题。"""
    if RAG_chain is None:
        init_rag_chain()
    if RAG_chain is None:
        return "抱歉，课程知识库暂时不可用。"
    try:
        return RAG_chain.invoke(question)
    except Exception as e:
        print(f"[RAG] 问答错误: {e}")
        return "抱歉，RAG 检索过程中出现错误。"


init_rag_chain()


# ════════════════════════════════════════════════════════════════
# 使用 @tool 装饰器定义工具
# ════════════════════════════════════════════════════════════════


@tool
def get_time() -> str:
    """获取当前系统时间，包括日期、星期和具体时间。当用户询问现在几点、今天日期、今天星期几时调用此工具。"""
    result = get_current_time()
    return json.dumps(result, ensure_ascii=False)


@tool
def get_weather(city_name: str) -> str:
    """查询指定城市的实时天气信息，包括温度、湿度、天气状况等。

    Args:
        city_name: 城市名称，例如 北京、上海、广州、深圳
    """
    result = get_weather_json(city_name.strip())
    return json.dumps(result, ensure_ascii=False)


@tool
def get_stock_price(ticker: str) -> str:
    """查询中国A股股票的实时价格信息，包括当前价格、涨跌幅等。

    Args:
        ticker: 股票代码，例如 600519（贵州茅台）、000001（平安银行）、300750（宁德时代）
    """
    result = get_stock_price_cn(ticker.strip())
    return json.dumps(result, ensure_ascii=False)


@tool
def send_email_tool(to_email: str, subject: str, body: str) -> str:
    """发送邮件到指定邮箱地址。

    Args:
        to_email: 收件人邮箱地址
        subject: 邮件主题
        body: 邮件正文内容
    """
    result = send_email(to_email.strip(), subject.strip(), body.strip())
    return json.dumps(result, ensure_ascii=False)


@tool
def send_dingtalk(content: str) -> str:
    """向钉钉群机器人发送文本消息。只有在用户提供了明确的消息正文时才调用此工具。

    Args:
        content: 要发送到钉钉群的具体消息内容
    """
    content = content.strip()
    if not content or content in {"发钉钉", "发送钉钉", "发个钉钉", "发送钉钉消息"}:
        return json.dumps({"code": 400, "msg": "请先提供明确的钉钉消息内容"}, ensure_ascii=False)
    # 统一添加前缀
    if not content.startswith("AI Agent："):
        content = f"AI Agent：{content}"
    result = send_dingtalk_message(DINGTALK_WEBHOOK, content)
    return json.dumps(result, ensure_ascii=False)


@tool
def course_rag_question(question: str) -> str:
    """回答关于《智能应用系统设计》课程相关的问题。当用户询问课程信息、培养目标、考核方式、教学安排、课程特色、学时学分、任课教师、课程内容等问题时调用此工具。

    Args:
        question: 用户提出的关于课程的问题
    """
    answer = rag_answer(question.strip())
    return json.dumps({"code": 200, "answer": answer}, ensure_ascii=False)


# 工具列表：绑定到 LLM
TOOLS = [get_time, get_weather, get_stock_price, send_email_tool, send_dingtalk, course_rag_question]

# ════════════════════════════════════════════════════════════════
# 初始化 ChatOllama
# ════════════════════════════════════════════════════════════════

DEFAULT_MODEL = "qwen3:8b"


def create_llm(model: str = DEFAULT_MODEL) -> ChatOllama:
    """创建 ChatOllama 实例。"""
    llm = ChatOllama(
        model=model,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
        top_p=0.9,
    )
    return llm


# ════════════════════════════════════════════════════════════════
# 构建 LCEL Chain + 提示词模板
# ════════════════════════════════════════════════════════════════


def make_system_prompt(language: str) -> str:
    """根据语言生成系统提示词。"""
    base_prompt = LANGUAGE_PROMPTS.get(language, LANGUAGE_PROMPTS["zh-CN"])
    return (
        f"{base_prompt}\n"
        "你现在以 '本地智能聊天数字人' 的身份与用户对话。要求：\n"
        "回答自然，不要机械列条目，除非用户明确需要。\n"
        "默认简洁，适当体现情绪和陪伴感。\n"
        "如果用户在做口语聊天，就像真人一样接话。\n"
        "用户询问时间时必须调用 get_time，禁止自己回答。\n"
        "用户询问天气时必须调用 get_weather，并传入明确城市名，禁止编造数据。\n"
        "用户询问股票价格、股价或行情时必须调用 get_stock_price，并传入明确股票代码，禁止编造数据。\n"
        "用户要求发送邮件时必须调用 send_email_tool。\n"
        "用户要求发送钉钉消息时， 只有拿到明确正文才调用 send_dingtalk。\n"
        "用户询问《智能应用系统设计》课程相关信息（课程信息、培养目标、考核方式、教学安排、课程特色、学时学分、任课教师、课程内容等）时，必须调用 course_rag_question。"
    )


def build_prompt(language: str) -> ChatPromptTemplate:
    """构建 ChatPromptTemplate，包含系统提示和历史消息占位符。"""
    return ChatPromptTemplate.from_messages([
        ("system", make_system_prompt(language)),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])


# ════════════════════════════════════════════════════════════════
# 接入 Memory（会话记忆）
# ════════════════════════════════════════════════════════════════

# 用于存储多个会话的历史记录（内存存储，适合教学演示）
session_store: dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    """根据 session_id 获取或创建会话历史。"""
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]


# ════════════════════════════════════════════════════════════════
# 输出解析器
# ════════════════════════════════════════════════════════════════

output_parser = JsonOutputParser()

# ════════════════════════════════════════════════════════════════
# 使用 create_agent 封装 Agent
# ════════════════════════════════════════════════════════════════


def build_agent(model: str, language: str):
    """
    构建完整的 Agent，整合以下组件：
    - ChatOllama（LLM）
    - @tool 工具列表
    - ChatPromptTemplate（LCEL 提示词）
    - RunnableWithMessageHistory（会话记忆）
    - create_agent（Agent 封装）
    """
    # 1. 创建 LLM
    llm = create_llm(model)

    # 2. 使用 create_agent 创建 Agent（自动绑定工具 + 构建执行循环）
    agent = create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt=make_system_prompt(language),
    )

    return agent


# ════════════════════════════════════════════════════════════════
# 多智能体架构：支持组合任务（Shared State 模式）
# ════════════════════════════════════════════════════════════════

class MultiAgentState(TypedDict, total=False):
    """多智能体共享状态"""
    input: str                    # 用户原始输入
    model: str                    # 模型名称
    language: str                 # 语言
    task_chain: list              # 任务链：[{"route": "weather", "city_name": "北京"}, {"route": "email", ...}]
    current_task_index: int       # 当前执行到的任务索引
    intermediate_results: list   # 中间结果列表
    final_reply: str              # 最终回复


def planner_node(state: MultiAgentState) -> MultiAgentState:
    """
    TaskPlanner Agent（任务规划器）
    分析用户输入，判断是否需要多步操作，生成任务链
    """
    user_input = state["input"]
    model = state["model"]

    planner_prompt = f"""
你是一个任务规划专家。你需要分析用户输入，判断用户想要执行的任务。

用户输入：{user_input}

分析规则：
1. 如果用户请求包含多个动作（如"先...再..."、"...然后..."、"...并..."），需要拆分为多个子任务
2. 每个子任务对应一个工具路由：time、weather、stock、email、dingtalk、chat、course_rag
3. 邮件任务必须提取：to_email（收件人）、subject（主题）、body（正文）
   - 如果之前有其他任务（如查天气、查股票），必须在邮件正文中包含那些任务的完整结果
   - 正文要详细、完整，不能省略任何信息
4. 钉钉任务必须提取：content（消息内容），内容要完整

示例分析：
- "查询上海天气然后发邮件给 xxx@qq.com" → 
  任务1: {{"route": "weather", "params": {{"city_name": "上海"}}}}
  任务2: {{"route": "email", "params": {{"to_email": "xxx@qq.com", "subject": "上海天气信息", "body": "上海今日天气：温度XX℃，湿度XX%，天气状况XX。注意：由于邮件正文中需要包含任务1的完整结果，请在此处预留占位符【任务1结果将在执行时填充】"}}}}
- "帮我看看贵州茅台的股价，然后发钉钉消息告诉同事" → 
  任务1: {{"route": "stock", "params": {{"ticker": "600519"}}}}
  任务2: {{"route": "dingtalk", "params": {{"content": "贵州茅台当前股价：XX元，涨幅：XX%"}}}}

请只返回JSON格式的任务链，格式如下：
{{"task_chain": [{{"route": "路由类型", "params": {{"参数名": "参数值"}}}}, ...], "is_multi_step": true/false}}

只返回JSON，不要有其他文字。
"""

    result = invoke_json_llm(model, planner_prompt, user_input)
    task_chain = result.get("task_chain", [])

    # 如果解析失败或为空，尝试单任务路由
    if not task_chain:
        single_route = route_user_request(user_input, model)
        if single_route == ROUTES["chat"]:
            task_chain = []
        else:
            # 从原始路由推断单任务
            task_chain = [{"route": single_route, "params": {}}]

    print(f"[Planner] 任务链生成: {task_chain}")
    return {"task_chain": task_chain, "current_task_index": 0, "intermediate_results": []}


def should_execute_next(state: MultiAgentState) -> str:
    """判断是否还有下一个任务需要执行"""
    task_chain = state.get("task_chain", [])
    current_index = state.get("current_task_index", 0)

    if not task_chain or current_index >= len(task_chain):
        return "finish"

    current_task = task_chain[current_index]
    route = current_task.get("route", "chat")

    # 如果是chat（对话），不需要执行器，直接进入chat节点
    if route == "chat":
        return "chat"

    # 否则进入执行器执行具体任务
    return "executor"


def executor_node(state: MultiAgentState) -> MultiAgentState:
    """
    Execution Agent（执行器）
    根据任务链中的当前任务，调用对应的工具并收集结果
    """
    task_chain = state.get("task_chain", [])
    current_index = state.get("current_task_index", 0)
    model = state["model"]
    language = state["language"]

    if current_index >= len(task_chain):
        return {"final_reply": "任务执行完成"}

    current_task = task_chain[current_index]
    route = current_task.get("route", "chat")
    params = current_task.get("params", {})

    # 构建临时state用于调用工具节点
    temp_state = {
        "input": state["input"],
        "model": model,
        "language": language,
        "route": route,
    }

    tool_result = ""
    reply = ""

    # 根据路由调用对应工具
    try:
        if route == "time":
            tool_result = get_time.invoke({})
            reply = render_reply(model, language, "现在几点？", tool_result)
        elif route == "weather":
            city_name = params.get("city_name", "")
            if not city_name:
                city_name = state["input"]
            tool_result = get_weather.invoke({"city_name": city_name})
            reply = render_reply(model, language, f"{city_name}天气", tool_result)
        elif route == "stock":
            ticker = params.get("ticker", "")
            tool_result = get_stock_price.invoke({"ticker": ticker})
            reply = render_reply(model, language, f"股票{ticker}", tool_result)
        elif route == "email":
            # 确保邮件参数完整，并填充之前任务的结果
            to_email = params.get("to_email", "")
            subject = params.get("subject", "AI数字人通知")
            body = params.get("body", "这是一封来自AI数字人的自动通知邮件。")
            
            # 如果邮件正文包含占位符，填充之前任务的中间结果
            if "【任务" in body or "结果" in body:
                # 收集之前任务的完整结果
                prev_results = []
                for i, prev_result in enumerate(state.get("intermediate_results", [])):
                    prev_results.append(f"任务{i+1}结果：{prev_result.get('reply', '')}")
                if prev_results:
                    body = body.replace("【任务1结果将在执行时填充】", "\n\n" + "\n\n".join(prev_results))
            
            email_params = {
                "to_email": to_email,
                "subject": subject,
                "body": body
            }
            tool_result = send_email_tool.invoke(email_params)
            reply = render_reply(model, language, "发送邮件", tool_result)
        elif route == "dingtalk":
            content = params.get("content", "")
            tool_result = send_dingtalk.invoke({"content": content})
            reply = render_reply(model, language, "发送钉钉", tool_result)
        elif route == "course_rag":
            question = params.get("question", state["input"])
            tool_result = course_rag_question.invoke({"question": question})
            parsed = parse_json_text(tool_result)
            reply = parsed.get("answer", "") if isinstance(parsed, dict) else ""
    except Exception as e:
        reply = f"执行任务时出错：{str(e)}"

    # 保存中间结果
    intermediate_results = state.get("intermediate_results", [])
    intermediate_results.append({
        "task_index": current_index,
        "route": route,
        "reply": reply,
        "tool_result": tool_result
    })

    # 更新状态，准备执行下一个任务
    next_index = current_index + 1
    print(f"[Executor] 任务 {current_index + 1}/{len(task_chain)} 完成: {route}")

    return {
        "current_task_index": next_index,
        "intermediate_results": intermediate_results
    }


def chat_node_multi(state: MultiAgentState) -> MultiAgentState:
    """多智能体架构下的对话节点（处理无工具调用的聊天）"""
    llm = create_llm(state["model"])
    result = llm.invoke([
        ("system", LANGUAGE_PROMPTS.get(state["language"], LANGUAGE_PROMPTS["zh-CN"])),
        ("human", state["input"]),
    ])
    return {"final_reply": strip_think_tags(getattr(result, "content", ""))}


def synthesizer_node(state: MultiAgentState) -> MultiAgentState:
    """
    Synthesizer Agent（综合器）
    将多个任务的执行结果整合成最终回复
    """
    model = state["model"]
    language = state["language"]
    intermediate_results = state.get("intermediate_results", [])
    original_input = state["input"]

    if not intermediate_results:
        return {"final_reply": "抱歉，无法处理您的请求。"}

    # 构建综合报告
    report_parts = []
    for i, result in enumerate(intermediate_results, 1):
        task_desc = f"任务{i}（{result['route']}）"
        report_parts.append(f"【{task_desc}】\n{result['reply']}")

    synthesis_prompt = f"""
用户原始请求：{original_input}

执行结果如下：
{chr(10).join(report_parts)}

请将以上多个任务的执行结果整合成一段流畅、简洁的回复。用口语化的方式告诉用户每个任务的结果。
不要列出任务编号，直接整合成一段话。
"""

    llm = create_llm(model)
    result = llm.invoke([
        ("system", LANGUAGE_PROMPTS.get(language, LANGUAGE_PROMPTS["zh-CN"]) + "\n你负责整合多个任务的执行结果。"),
        ("human", synthesis_prompt),
    ])

    final_reply = strip_think_tags(getattr(result, "content", ""))

    # 如果LLM整合失败，手动拼接
    if not final_reply or len(final_reply) < 5:
        final_reply = " | ".join([r["reply"] for r in intermediate_results])

    return {"final_reply": final_reply}


def build_multi_agent(model: str, language: str):
    """
    构建多智能体架构的数字人系统
    核心改进：支持组合任务（先查询天气 -> 再发送邮件）
    架构：Planner -> Executor(循环) -> Synthesizer
    """

    workflow = StateGraph(MultiAgentState)

    # 添加所有节点
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("chat", chat_node_multi)
    workflow.add_node("synthesizer", synthesizer_node)

    # 设置入口
    workflow.add_edge(START, "planner")

    # Planner 之后，根据任务链情况分流
    workflow.add_conditional_edges(
        "planner",
        should_execute_next,
        {
            "executor": "executor",  # 有工具任务，进入执行器
            "chat": "chat",           # 无工具任务，进入对话
            "finish": END             # 空任务链，结束
        }
    )

    # Executor 执行完后，判断是否继续执行还是进入综合器
    workflow.add_conditional_edges(
        "executor",
        should_execute_next,
        {
            "executor": "executor",  # 还有任务，继续执行
            "chat": "chat",          # 遇到chat任务
            "finish": "synthesizer"  # 任务链完成，进入综合器
        }
    )

    workflow.add_edge("chat", END)
    workflow.add_edge("synthesizer", END)

    graph_app = workflow.compile()

    # 生成架构图
    with open("langgraph_multi_agent.png", "wb") as f:
        f.write(graph_app.get_graph().draw_mermaid_png())

    return graph_app


# ════════════════════════════════════════════════════════════════
# Flask 应用
# ════════════════════════════════════════════════════════════════

ROUTES = {
    "chat": "chat",        # 聊天路由
    "time": "time",        # 时间路由
    "weather": "weather",  # 天气路由
    "stock": "stock",      # 股票路由
    "email": "email",      # 邮件路由
    "dingtalk": "dingtalk", # 钉钉路由
    "course_rag": "course_rag", # 课程路由
}


class RouterState(TypedDict, total=False):
    input: str     # 用户问题
    model: str     # 模型名称
    language: str  # 语言
    route: str     # 路由（chat/time/weather/stock/email/dingtalk/course_rag）
    reply: str     # 回复内容   


def parse_json_text(text: str) -> dict:
    """安全解析 LLM 返回的 JSON 字符串。"""
    cleaned = strip_think_tags(text)
    try:
        return json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}


def invoke_json_llm(model: str, system_prompt: str, user_input: str) -> dict:
    """调用 LLM 输出 JSON，失败时返回空字典。"""
    llm = create_llm(model)
    result = llm.invoke([
        ("system", system_prompt),
        ("human", user_input),
    ])
    return parse_json_text(getattr(result, "content", ""))


def render_reply(model: str, language: str, user_input: str, tool_result: str) -> str:
    """将工具结果转换为用户可直接阅读的答复。"""
    llm = create_llm(model)
    result = llm.invoke([
        ("system",
         f"{LANGUAGE_PROMPTS.get(language, LANGUAGE_PROMPTS['zh-CN'])}\n"
         "请根据工具返回结果直接回答用户。"
         "如果工具返回 code 不是 200，请简要说明失败原因。"
         "不要编造未出现在工具结果里的事实。"),
        ("human", f"用户问题：{user_input}\n工具结果：{tool_result}"),
    ])
    return strip_think_tags(getattr(result, "content", ""))


def fallback_route(user_input: str) -> str:
    """当 Router JSON 失败时的简单关键词兜底。"""
    text = (user_input or "").lower()
    if "邮件" in user_input or "email" in text or "mail" in text:
        return ROUTES["email"]
    if "钉钉" in user_input or "dingtalk" in text:
        return ROUTES["dingtalk"]
    if "股票" in user_input or "股价" in user_input or "ticker" in text:
        return ROUTES["stock"]
    if "天气" in user_input:
        return ROUTES["weather"]
    if "时间" in user_input or "几点" in user_input or "date" in text or "time" in text:
        return ROUTES["time"]
    if "《智能应用系统设计》" in user_input or "课程" in user_input:
        return ROUTES["course_rag"]
    return ROUTES["chat"]


def route_user_request(user_input: str, model: str) -> str:
    """Router 节点只负责判断用户想干什么。"""
    router_prompt = """
你是一个任务路由器，只能判断用户意图，不负责回答问题。
请从以下 route 中选一个：chat、time、weather、stock、email、dingtalk、course_rag。
只返回 JSON：{"route":"xxx"}
"""
    payload = invoke_json_llm(model, router_prompt, user_input)
    route = str(payload.get("route", "")).strip()
    if route in ROUTES.values():
        return route
    return fallback_route(user_input) # 如果 JSON 解析失败，使用关键词底路由


def extract_args(model: str, user_input: str, schema_prompt: str) -> dict:
    """为具体节点提取参数。"""
    return invoke_json_llm(model, schema_prompt, user_input)


def build_tool_node(tool_fn, arg_prompt: str | None = None, arg_names: list[str] | None = None):
    """组装“参数提取 + 直接调用 tool + 生成回复”的节点。"""
    arg_names = arg_names or []

    def node(state: RouterState) -> RouterState:
        user_input = state["input"]
        model = state["model"]
        language = state["language"]

        kwargs = {}
        if arg_prompt and arg_names:
            payload = extract_args(model, user_input, arg_prompt)
            kwargs = {name: str(payload.get(name, "")).strip() for name in arg_names}

        tool_result = tool_fn.invoke(kwargs) if kwargs else tool_fn.invoke({})
        reply = render_reply(model, language, user_input, tool_result)
        return {"reply": reply}

    return node


def chat_node(state: RouterState) -> RouterState:
    """普通对话节点，不调用 tool。"""
    llm = create_llm(state["model"])
    result = llm.invoke([
        ("system", LANGUAGE_PROMPTS.get(state["language"], LANGUAGE_PROMPTS["zh-CN"])),
        ("human", state["input"]),
    ])
    return {"reply": strip_think_tags(getattr(result, "content", ""))}


def course_rag_node(state: RouterState) -> RouterState:
    """课程 RAG 节点直接调用工具，优先返回知识库答案。"""
    tool_result = course_rag_question.invoke({"question": state["input"]})
    parsed = parse_json_text(tool_result)
    reply = parsed.get("answer", "") if isinstance(parsed, dict) else ""
    if not reply:
        reply = render_reply(state["model"], state["language"], state["input"], tool_result)
    return {"reply": reply}


def router_node(state: RouterState) -> RouterState:
    """路由节点，根据用户输入判断路由。"""
    return {"route": route_user_request(state["input"], state["model"])}


def route_next(state: RouterState) -> str:
    route = state.get("route", ROUTES["chat"])
    return route if route in ROUTES.values() else ROUTES["chat"]


def build_agent(model: str, language: str):
    """使用 LangGraph 拆分 Router 和具体执行节点。"""
    weather_arg_prompt = """
请从用户输入中提取查询天气需要的城市名称。
只返回 JSON：{"city_name":"..."}
"""
    stock_arg_prompt = """
请从用户输入中提取股票代码。
只返回 JSON：{"ticker":"..."}
"""
    email_arg_prompt = """
请从用户输入中提取发送邮件所需参数。
只返回 JSON：{"to_email":"...","subject":"...","body":"..."}
"""
    dingtalk_arg_prompt = """
请从用户输入中提取需要发送到钉钉的消息内容。
只返回 JSON：{"content":"..."}
"""

    workflow = StateGraph(RouterState)  # 定义状态图
    # 添加节点
    workflow.add_node("router", router_node)
    workflow.add_node("chat", chat_node)
    workflow.add_node("time", build_tool_node(get_time))
    workflow.add_node("weather", build_tool_node(get_weather, weather_arg_prompt, ["city_name"]))
    workflow.add_node("stock", build_tool_node(get_stock_price, stock_arg_prompt, ["ticker"]))
    workflow.add_node("email", build_tool_node(send_email_tool, email_arg_prompt, ["to_email", "subject", "body"]))
    workflow.add_node("dingtalk", build_tool_node(send_dingtalk, dingtalk_arg_prompt, ["content"]))
    workflow.add_node("course_rag", course_rag_node)

    workflow.add_edge(START, "router") # 从开始节点到路由节点

    # 添加条件边 根据路由判断是否调用工具节点或普通节点
    workflow.add_conditional_edges(
        "router",
        route_next,
        {
            ROUTES["chat"]: "chat",
            ROUTES["time"]: "time",
            ROUTES["weather"]: "weather",
            ROUTES["stock"]: "stock",
            ROUTES["email"]: "email",
            ROUTES["dingtalk"]: "dingtalk",
            ROUTES["course_rag"]: "course_rag",
        },
    )
    # 添加普通节点到结束节点的边
    workflow.add_edge("chat", END)
    workflow.add_edge("time", END)
    workflow.add_edge("weather", END)
    workflow.add_edge("stock", END)
    workflow.add_edge("email", END)
    workflow.add_edge("dingtalk", END)
    workflow.add_edge("course_rag", END)
    graph_app = workflow.compile()
    with open("langgraph_ai-agent.png", "wb") as f:
        f.write(graph_app.get_graph().draw_mermaid_png())
    return graph_app


app = Flask(__name__, static_folder="public")


def strip_think_tags(text: str) -> str:
    """移除模型返回中的 <think>...</think> 标签。"""
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


@app.post("/api/chat")
def api_chat() -> Response:
    """聊天 API：接收用户消息，通过多智能体处理后返回回复。"""
    try:
        body = request.get_json(silent=True) or {}
        model = body.get("model") or DEFAULT_MODEL
        language = body.get("language") or "zh-CN"
        messages = body.get("messages") if isinstance(body.get("messages"), list) else []
        session_id = body.get("session_id") or "default"

        # 提取最新的用户消息
        user_input = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_input = str(msg.get("content", "")).strip()
                break

        if not user_input:
            return jsonify({"reply": "请输入您的问题。"})

        # 构建多智能体（支持组合任务）
        agent = build_multi_agent(model, language)

        # 调用多智能体（传入消息历史用于上下文）
        result = agent.invoke(
            {
                "input": user_input,
                "model": model,
                "language": language,
            },
            config={"configurable": {"thread_id": session_id}},
        )

        # 提取最终回复（多智能体返回 final_reply）
        reply = ""
        if isinstance(result, dict):
            reply = result.get("final_reply", result.get("reply", ""))
            # 如果没有 final_reply，尝试从 intermediate_results 构建
            if not reply:
                intermediate_results = result.get("intermediate_results", [])
                if intermediate_results:
                    reply = " | ".join([r.get("reply", "") for r in intermediate_results])

        reply = strip_think_tags(reply)

        # 使用 JsonOutputParser 尝试解析结构化输出
        try:
            parsed = output_parser.parse(reply)
            if isinstance(parsed, dict) and "reply" in parsed:
                reply = parsed["reply"]
        except Exception:
            pass  # 非 JSON 格式则保持原文本

        return jsonify({"reply": reply or "抱歉，我暂时无法回答。"})

    except Exception as error:
        return jsonify({"error": "INTERNAL_ERROR", "message": str(error)}), 500


# ── TTS 语音合成接口 ─────────────────────────────────

async def generate_speech(text: str, voice: str) -> bytes:
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
def api_tts() -> Response:
    try:
        body = request.get_json(silent=True) or {}
        text = body.get("text", "")
        language = body.get("language", "zh-CN")
        voice = LANGUAGE_TTS_VOICES.get(language, LANGUAGE_TTS_VOICES["zh-CN"])

        if not text:
            return jsonify({"error": "TEXT_REQUIRED", "message": "Text is required."}), 400

        audio_data = asyncio.run(generate_speech(text, voice))
        return Response(audio_data, mimetype="audio/mpeg")
    except Exception as error:
        return jsonify({"error": "TTS_ERROR", "message": str(error)}), 500


@app.get("/api/tts-status")
def api_tts_status() -> Response:
    return jsonify({"ready": True, "voice": "Microsoft Edge-TTS"})


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
