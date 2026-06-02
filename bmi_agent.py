# bmi_agent.py - 基于LangGraph和MCP的BMI查询助手
import asyncio
from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient

class AgentState(TypedDict):
    messages: list

async def main():
    # MCP客户端连接BMI服务
    mcp_client = MultiServerMCPClient({
        "bmi": {
            "transport": "streamable_http",
            "url": "http://127.0.0.1:8003/mcp"
        }
    })

    # 获取MCP工具
    tools = await mcp_client.get_tools()

    # 初始化Ollama模型并绑定工具
    llm = ChatOllama(model="qwen3:0.6b", base_url="http://localhost:11434")
    llm_with_tools = llm.bind_tools(tools)

    system_prompt = """你是一个专业的BMI计算助手。

当用户提供身高和体重时：
1. 调用 calculate_bmi 工具计算BMI
2. 根据工具返回的BMI值和分类，用自然语言回复用户

BMI分类标准：
- 偏瘦：BMI < 18.5
- 正常范围：18.5 <= BMI < 24
- 偏胖：24 <= BMI < 28
- 肥胖：BMI >= 28

回复格式示例：
"根据计算，您的BMI指数约为22.49，属于正常范围。"

重要：必须根据工具返回的结果生成自然语言回复，不要只返回工具的原始输出。"""

    def agent_node(state: AgentState) -> dict:
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> Literal["tools", "END"]:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "END"

    async def tools_node(state: AgentState) -> dict:
        last_message = state["messages"][-1]
        new_messages = []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            for tool in tools:
                if tool.name == tool_name:
                    result = await tool.ainvoke(tool_args)
                    new_messages.append(AIMessage(
                        content=str(result.content[0].text) if hasattr(result, "content") else str(result),
                        tool_call_id=tool_call["id"]
                    ))
                    break

        return {"messages": new_messages}

    # 构建图
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {
        "tools": "tools",
        "END": END
    })
    graph.add_edge("tools", "agent")

    app = graph.compile()

    # 对话循环
    print("BMI查询助手已启动（输入 exit 退出）")
    while True:
        user_input = input("\n请输入: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        result = await app.ainvoke({
            "messages": [HumanMessage(content=user_input)]
        })

        print("\n助手回复:")
        for msg in result["messages"]:
            if hasattr(msg, "content") and msg.content:
                print(msg.content)

asyncio.run(main())