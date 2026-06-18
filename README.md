# AI Agent 数字人系统

基于 LangGraph 的多智能体数字人对话系统，支持组合任务执行。

## 功能特性

- **多轮对话**：支持上下文记忆的自然语言对话
- **天气查询**：实时查询各大城市天气状况
- **股票查询**：查询A股实时股价信息
- **时间查询**：获取当前系统时间
- **邮件发送**：通过SMTP发送邮件
- **钉钉通知**：向钉钉群发送消息
- **课程问答**：基于RAG的《智能应用系统设计》课程知识库问答

## 核心架构

```
用户输入 → Planner（任务规划）→ Executor（执行器）→ Synthesizer（结果综合）→ 回复
                ↓
         ┌─────┴─────┐
         ↓           ↓
    任务1执行    任务2执行 ...
```

**多智能体组件：**
| Agent | 职责 |
|-------|------|
| TaskPlanner | 分析用户意图，拆分多步任务链 |
| ExecutionAgent | 按序执行工具任务，收集中间结果 |
| SynthesizerAgent | 整合多任务结果生成最终回复 |

## 环境要求

- Python 3.10+
- Ollama 服务（本地大模型）
- 网络连接（天气/股票API）

## 安装配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，配置以下参数：

```env
# SMTP邮件配置（可选）
SMTP_SERVER=smtp.163.com
SMTP_PORT=465
SMTP_USERNAME=your_email@163.com
SMTP_PASSWORD=your授权码
FROM_EMAIL=your_email@163.com
FROM_NAME=AI数字人助手

# Ollama服务地址
OLLAMA_URL=http://127.0.0.1:11434

# 服务器配置
HOST=127.0.0.1
PORT=3000
```

### 3. 启动Ollama服务

```bash
ollama serve
```

确保已下载模型：
```bash
ollama pull qwen3:0.6b
ollama pull qwen3-embedding:0.6b
```

### 4. 启动应用

```bash
python app.py
```

访问 `http://127.0.0.1:3000`

## 使用示例

### 单任务
- "今天天气怎么样" → 查询天气
- "现在几点了" → 查询时间
- "帮我看看贵州茅台的股价" → 查询股票

### 组合任务（多智能体协作）
- "查询北京天气然后发邮件到 xxx@163.com" → 查天气 → 发邮件
- "帮我看看茅台股价，然后发钉钉告诉同事" → 查股价 → 发钉钉

## 项目结构

```
ai-agent/
├── app.py              # 主应用（多智能体架构）
├── tools.py            # 工具函数（天气/股票/邮件/钉钉）
├── requirements.txt    # 依赖列表
├── .env.example        # 环境变量示例
├── .env                # 环境变量（需自行创建）
├── public/             # 前端静态文件
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── resources/         # 资源文件
│   └── avatar.jpg
└── 《智能应用系统设计》课程介绍.md  # RAG知识库
```

## 技术栈

- **框架**：Flask + LangGraph
- **模型**：Ollama (qwen3:0.6b)
- **向量库**：Chroma + BM25
- **语音**：Edge-TTS
- **工具调用**：LangChain Tools
