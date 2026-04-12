# AI Agent 数字人

基于本地 Ollama 大模型的多语言智能对话数字人系统。

## 功能特性

- 多语言对话：支持普通话、粤语、台湾国语、英语、日语、韩语等14种语言/方言
- 语音输入：基于 Web Speech API 的语音识别
- 语音播报：微软 Edge TTS 神经网络语音合成
- 智能工具：内置时间查询、天气查询工具
- 数字人形象：口型动画、说话动画效果
- 本地部署：无需联网，保护隐私

## 技术栈

- 后端：Flask + Python
- 前端：HTML5 + CSS3 + JavaScript
- 大模型：Ollama (qwen3:0.6b/8b)
- 语音：Microsoft Edge TTS
- 天气API：sojson.com

## 快速开始

### 1. 安装依赖

```bash
cd ai-agent
pip install -r requirements.txt
```

### 2. 启动 Ollama

确保本地已安装并运行 Ollama：

```bash
ollama serve
ollama pull qwen3:0.6b
```

### 3. 启动服务

```bash
python app.py
```

访问 http://127.0.0.1:3000

## 项目结构

```
ai-agent/
├── app.py              # 后端主程序
├── requirements.txt    # Python 依赖
├── public/             # 前端资源
│   ├── index.html      # 页面入口
│   ├── styles.css      # 样式文件
│   ├── app.js          # 前端逻辑
│   └── resources/      # 静态资源
└── resources/          # 资源目录
    └── avatar.jpg      # 数字人头像
```

## 接口说明

| 接口 | 方法 | 说明 |
|------|------|------|
| /api/chat | POST | 对话接口 |
| /api/tts | POST | 语音合成 |
| /api/tts-status | GET | TTS状态查询 |

## 配置项

可通过环境变量配置：

- OLLAMA_URL：Ollama 服务地址（默认 http://127.0.0.1:11434）
- HOST：服务监听地址（默认 127.0.0.1）
- PORT：服务监听端口（默认 3000）
- EDGE_VOICE_*：各语言语音配置
