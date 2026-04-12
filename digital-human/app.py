#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Agent 数字人后端
接入本地 Ollama 大模型 qwen3:0.6b
"""

from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_MODEL = "qwen3:0.6b"

# 人设提示词
PERSONAS = {
    "default": "你是一个友好、智能的AI助手。请用简洁、自然的口吻回答问题，保持专业和礼貌。",
    "teacher": "你是一位经验丰富的老师。回答问题时要耐心讲解，语言通俗易懂，适当举例。",
    "friend": "你是用户的好朋友。聊天亲切自然，用轻松愉快的语气，可以适当用表情。",
    "expert": "你是一位专业专家。回答问题严谨专业，使用准确术语，详细解释。",
}

# 快捷提示词
QUICK_PROMPTS = [
    "介绍一下你自己",
    "帮我写一首诗",
    "解释一下什么是机器学习",
    "推荐一本好书",
]

def get_ollama_response(messages, model=DEFAULT_MODEL):
    """调用 Ollama API 获取回复"""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
        }
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result.get("message", {}).get("content", "")
    except requests.exceptions.ConnectionError:
        return "无法连接到 Ollama 服务，请确保 Ollama 已启动。"
    except requests.exceptions.Timeout:
        return "请求超时，请稍后重试。"
    except Exception as e:
        return f"发生错误: {str(e)}"

@app.route("/")
def index():
    """渲染主页"""
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    """处理聊天请求"""
    data = request.get_json()
    user_message = data.get("message", "").strip()
    model = data.get("model", DEFAULT_MODEL)
    persona = data.get("persona", "default")
    history = data.get("history", [])
    
    if not user_message:
        return jsonify({"error": "消息不能为空"}), 400
    
    # 获取人设提示词
    system_prompt = PERSONAS.get(persona, PERSONAS["default"])
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # 添加历史对话（保留最近10轮）
    for msg in history[-10:]:
        if msg.get("role") in ["user", "assistant"]:
            messages.append(msg)
    
    # 添加当前消息
    messages.append({"role": "user", "content": user_message})
    
    response = get_ollama_response(messages, model)
    return jsonify({"response": response})

@app.route("/api/quick-prompts", methods=["GET"])
def quick_prompts():
    """获取快捷提示词"""
    return jsonify({"prompts": QUICK_PROMPTS})

if __name__ == "__main__":
    print("=" * 40)
    print("AI Agent 数字人")
    print("Ollama (qwen3:0.6b)")
    print("=" * 40)
    app.run(host="0.0.0.0", port=5000, debug=True)
