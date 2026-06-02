# bmi_server.py - BMI计算MCP服务
from mcp.server.fastmcp import FastMCP
import uvicorn

mcp = FastMCP("BMI Server")

@mcp.tool()
def calculate_bmi(weight: float, height: float) -> dict:
    """
    计算BMI指数
    :param weight: 体重（公斤）
    :param height: 身高（米）
    :return: BMI计算结果
    """
    if height <= 0:
        return {"error": "身高必须大于0"}
    if weight <= 0:
        return {"error": "体重必须大于0"}

    bmi = weight / (height ** 2)
    bmi_rounded = round(bmi, 2)

    if bmi < 18.5:
        category = "偏瘦"
    elif bmi < 24:
        category = "正常范围"
    elif bmi < 28:
        category = "偏胖"
    else:
        category = "肥胖"

    return {
        "bmi": bmi_rounded,
        "category": category
    }

if __name__ == "__main__":
    app = mcp.streamable_http_app()
    uvicorn.run(app, host="127.0.0.1", port=8003)