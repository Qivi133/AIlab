# AI Agent 数字人

基于 Ollama 本地大语言模型和 Edge-TTS 语音合成的数字人对话系统。

## 功能特性

- **本地大模型对话**：集成 Ollama，支持 Qwen3、DeepSeek-R1 等本地模型
- **语音合成播报**：使用 Microsoft Edge-TTS 进行语音输出
- **语音识别输入**：支持浏览器 Web Speech API 语音输入
- **智能工具调用**：
  - 获取当前时间
  - 查询天气（支持国内主要城市）
  - 查询 A 股股票价格
  - 发送电子邮件
  - 发送钉钉消息
- **多语言支持**：支持普通话、粤语、英语、日语、韩语等 14 种语言

## 技术栈

- **后端**：Python Flask
- **前端**：原生 HTML/CSS/JavaScript
- **AI 模型**：Ollama (Qwen3/DeepSeek-R1)
- **语音合成**：Microsoft Edge-TTS
- **股票数据**：efinance

## 环境要求

- Python 3.8+
- Ollama 服务（本地运行）
- 支持 Web Speech API 的现代浏览器

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件，配置以下内容：

```bash
# 邮件服务配置（如需发送邮件功能）
SMTP_SERVER=smtp.qq.com
SMTP_PORT=587
SMTP_USERNAME=your_email@qq.com
SMTP_PASSWORD=your_authorization_code
FROM_EMAIL=your_email@qq.com
FROM_NAME=AI Agent

# Ollama 服务地址
OLLAMA_URL=http://127.0.0.1:11434/api/chat

# Edge-TTS 语音参数
EDGE_TTS_RATE=+0%
EDGE_TTS_VOLUME=+0%
EDGE_TTS_PITCH=+0Hz

# 服务器配置
HOST=127.0.0.1
PORT=3000
```

### 3. 启动 Ollama

确保 Ollama 服务正在运行：

```bash
# 启动 Ollama 服务
ollama serve

# 下载模型（根据需要选择）
ollama pull qwen3:8b
ollama pull deepseek-r1:7b
```

### 4. 启动应用

```bash
python app.py
```

### 5. 访问界面

打开浏览器访问：`http://127.0.0.1:3000`

## 使用说明

### 对话交互

1. 在文本框中输入想说的话
2. 点击"发送"按钮或按 Enter 键提交
3. 数字人会自动回答并进行语音播报

### 工具调用示例

- **查询时间**："现在几点了？"
- **查询天气**："北京天气怎么样？"
- **查询股票**："茅台股票现在多少钱？"
- **发送邮件**："帮我发一封邮件，主题是测试，内容是你好"
- **发送钉钉**："在钉钉群里发一条消息，说大家好"

### 设置选项

- **模型选择**：支持切换 Qwen3:0.6b、Qwen3:8b、DeepSeek-R1:7b
- **语言选择**：支持 14 种语言的对话和语音输出
- **语音开关**：可随时开启/关闭语音播报
- **语音输入**：点击"语音输入"按钮进行语音输入

## 项目结构

```
ai-agent/
├── app.py              # Flask 后端主程序
├── tools.py            # 工具函数（天气、股票、邮件、钉钉）
├── test_stock.py       # 股票查询测试脚本
├── requirements.txt    # Python 依赖
├── .env                # 环境变量配置
├── .gitignore          # Git 忽略文件
├── public/
│   ├── index.html      # 前端页面
│   ├── app.js          # 前端逻辑
│   └── styles.css      # 样式文件
└── resources/
    └── avatar.jpg      # 数字人头像
```

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/chat` | POST | 对话接口 |
| `/api/tts` | POST | 语音合成接口 |
| `/api/tts-status` | GET | TTS 状态查询 |

## 注意事项

1. 首次使用需确保 Ollama 服务正常运行
2. 邮件发送功能需要配置 SMTP 授权码
3. 股票查询依赖 `efinance` 库，需确保网络通畅
4. 语音输入需要浏览器支持 Web Speech API
