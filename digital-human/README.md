# AI Agent 数字人项目

## 项目结构

```
digital-human/
├── app.py              # Flask 后端
├── requirements.txt    # Python 依赖
├── static/
│   └── style.css      # 样式文件
├── templates/
│   └── index.html     # 前端页面
└── README.md          # 说明文档
```

## 运行方式

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 确保 Ollama 已启动并加载 qwen3:0.6b 模型：
```bash
ollama run qwen3:0.6b
```

3. 启动 Flask 服务：
```bash
python app.py
```

4. 打开浏览器访问：`http://127.0.0.1:5000`

## 功能特性

- 接入本地 Ollama 的 qwen3:0.6b 大模型
- 实时流式输出回复
- 简洁美观的数字人对话界面
- 支持语音合成（Edge TTS）
- 多语言支持
