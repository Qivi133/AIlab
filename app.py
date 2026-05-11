# ─────────────────────────────────────────────────────────────────────────────
# 学业预警知识问答系统 - 基于LangChain和RAG
# ─────────────────────────────────────────────────────────────────────────────
# 技术栈：
#   - 文档加载：PyPDFLoader
#   - 文档切分：RecursiveCharacterTextSplitter
#   - 向量数据库：Chroma
#   - Embedding模型：nomic-embed-text (Ollama)
#   - BM25检索：rank_bm25
#   - 混合检索：EnsembleRetriever
#   - 重排序模型：BAAI/bge-reranker-base
#   - LLM：qwen3:0.6b (Ollama)
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
from pathlib import Path

# 加载PDF、文本分割、向量化、LLM相关
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder

# Flask Web服务
from flask import Flask, request, jsonify, render_template

# ─────────────────────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "软件与人工智能学院本科生学业预警实施办法.pdf"
CHROMA_DIR = BASE_DIR / "chroma_db"

OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen3:0.6b"

# 全局变量存储初始化状态
_system_initialized = False
_vectorstore = None
_ensemble_retriever = None
_reranker = None
_llm = None
_prompt = None


# ─────────────────────────────────────────────────────────────────────────────
# RAG初始化
# ─────────────────────────────────────────────────────────────────────────────
def init_rag_system():
    """初始化RAG系统组件"""
    global _system_initialized, _vectorstore, _ensemble_retriever, _reranker, _llm, _prompt

    if _system_initialized:
        return

    print("[INFO] 正在初始化RAG系统...")

    # 1. 加载PDF文档
    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()
    print(f"      加载了 {len(documents)} 页")

    # 2. 切分文档
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "，", "、", " "]
    )
    splits = text_splitter.split_documents(documents)
    print(f"      切分为 {len(splits)} 个文本块")

    # 3. 初始化Embedding模型和向量库
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    _vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    # 4. 构建向量检索器
    vector_retriever = _vectorstore.as_retriever(search_kwargs={"k": 6})

    # 5. 构建BM25检索器
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 6

    # 6. 构建混合检索器
    _ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )

    # 7. 初始化Cross-Encoder重排序模型
    print("[7/7] 加载重排序模型...")
    _reranker = CrossEncoder("BAAI/bge-reranker-base")

    # 8. 初始化LLM
    _llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)

    # 9. 初始化Prompt
    _prompt = ChatPromptTemplate.from_template(
        "你是一个学业预警咨询助手，基于以下《软件与人工智能学院本科生学业预警实施办法》的内容回答学生问题。\n"
        "请严格按照文档内容回答，不要编造信息。如果问题与学业预警无关，请告知学生仅能回答学业预警相关问题。\n\n"
        "【参考文档内容】：\n{context}\n\n"
        "【学生问题】：{question}\n\n"
        "【回答】："
    )

    _system_initialized = True
    print("[INFO] RAG系统初始化完成！")


def rerank_docs(question, docs, top_n=4):
    if not docs:
        return []

    pairs = [[question, doc.page_content] for doc in docs]
    scores = _reranker.predict(pairs)

    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, score in scored_docs[:top_n]]


def answer_question(question: str) -> dict:
    """回答学业预警相关问题"""
    if not _system_initialized:
        init_rag_system()

    try:
        # 1. 混合检索
        docs = _ensemble_retriever.invoke(question)

        # 2. 重排序
        reranked_docs = rerank_docs(question, docs, top_n=4)

        if not reranked_docs:
            return {
                "code": 404,
                "question": question,
                "answer": "抱歉，我在《学业预警实施办法》中未找到与您问题相关的内容。",
                "sources": []
            }

        # 3. 拼接上下文
        context = "\n\n".join([doc.page_content for doc in reranked_docs])

        # 4. 生成回答
        chain = _prompt | _llm | StrOutputParser()
        answer = chain.invoke({"question": question, "context": context})

        # 5. 返回结果
        sources = [
            {
                "content": doc.page_content[:100] + "...",
                "score": float(score)
            }
            for doc, score in zip(reranked_docs, _reranker.predict(
                [[question, doc.page_content] for doc in reranked_docs]
            ))
        ]

        return {
            "code": 200,
            "question": question,
            "answer": answer,
            "sources": sources
        }

    except Exception as e:
        return {
            "code": 500,
            "question": question,
            "answer": f"系统出错：{str(e)}",
            "sources": []
        }


# ─────────────────────────────────────────────────────────────────────────────
# Flask Web应用
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")

# 确保目录存在
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)


@app.route("/")
def index():
    """主页"""
    return render_template("index.html")


@app.route("/api/init", methods=["POST"])
def api_init():
    """初始化RAG系统"""
    try:
        init_rag_system()
        return jsonify({"code": 200, "message": "系统初始化完成"})
    except Exception as e:
        return jsonify({"code": 500, "message": f"初始化失败：{str(e)}"})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """问答接口"""
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"code": 400, "message": "请输入问题"})

    result = answer_question(question)
    return jsonify(result)


# ─────────────────────────────────────────────────────────────────────────────
# 示例问题测试
# ─────────────────────────────────────────────────────────────────────────────
def test_questions():
    """测试示例问题"""
    print("\n" + "=" * 60)
    print("学业预警知识问答系统 - 测试")
    print("=" * 60)

    test_queries = [
        "学业预警工作由谁负责？",
        "什么情况下会被黄色预警？",
        "红色预警的条件是什么？绩点低于多少算红色预警？",
        "绩点1.7，2门必修不及格会收到什么预警？",
        "如果一学期挂了4门必修课，会收到什么预警？会有哪些措施？",
        "我有3门选修课没过，会预警吗？"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n【问题{i}】{query}")
        print("-" * 40)
        result = answer_question(query)
        print(f"回答：{result['answer']}")


# ─────────────────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # 命令行测试模式
        init_rag_system()
        test_questions()
    else:
        # Web服务模式
        print("启动学业预警知识问答系统...")
        print(f"访问地址：http://127.0.0.1:5000")
        app.run(host="0.0.0.0", port=5000, debug=True)
