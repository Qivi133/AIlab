"""工具模块 - 提供各种查询功能"""

import datetime
import os
import smtplib
import requests
from email.mime.text import MIMEText
from email.header import Header


def get_stock_price_cn(stock_code: str) -> dict:
    """获取中国A股/港股实时股价信息

    Args:
        stock_code: 股票代码，如 600519（茅台）、000858（五粮液）、00700（腾讯）

    Returns:
        dict: 包含股价信息的字典
            - name: 股票名称
            - price: 当前价格（元）
            - change: 涨跌额（元）
            - percent: 涨跌幅（%）
            - high: 最高价（元）
            - low: 最低价（元）
            - open: 开盘价（元）
            - volume: 成交量（股）
            - market: 市场类型（sh/sz/hk）
    """
    # 判断市场：沪市以6/5开头，深市以0/3开头，港股以0/8开头
    if stock_code.startswith("6") or stock_code.startswith("5"):
        secid = f"1.{stock_code}"  # 上海
        market = "sh"
    elif stock_code.startswith("0") or stock_code.startswith("3"):
        secid = f"0.{stock_code}"  # 深圳
        market = "sz"
    else:
        secid = f"116.{stock_code}"  # 港股
        market = "hk"

    url = "https://push2.eastmoney.com/api/qt/stock/get"
    params = {
        "secid": secid,
        "fields": "f43,f44,f45,f46,f47,f48,f57,f58,f60,f169,f170",
        "ut": "fa5fd1943c7b386f172d6893dbfba10b",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://quote.eastmoney.com/",
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json().get("data", {})
            if data:
                return {
                    "name": data.get("f58", ""),
                    "price": round(data.get("f43", 0) / 100, 2),
                    "change": round(data.get("f170", 0) / 100, 2),
                    "percent": data.get("f169", 0),
                    "high": round(data.get("f44", 0) / 100, 2),
                    "low": round(data.get("f45", 0) / 100, 2),
                    "open": round(data.get("f46", 0) / 100, 2),
                    "volume": data.get("f48", 0),
                    "market": market,
                    "success": True
                }
    except Exception:
        pass

    return {"success": False, "error": "无法获取股票信息"}


# 股票代码映射表
STOCK_CODES = {
    "贵州茅台": "600519",
    "茅台": "600519",
    "五粮液": "000858",
    "腾讯": "00700",
    "腾讯控股": "00700",
    "阿里巴巴": "09988",
    "阿里": "09988",
    "京东": "09618",
    "百度": "09888",
    "中国平安": "601318",
    "平安": "601318",
    "招商银行": "600036",
    "工商银行": "601398",
    "建设银行": "601939",
    "中国银行": "601988",
    "农业银行": "601288",
    "中国石油": "601857",
    "中石油": "601857",
    "中国石化": "600028",
    "中石化": "600028",
    "比亚迪": "002594",
    "宁德时代": "300750",
    "美的集团": "000333",
    "格力电器": "000651",
    "万科": "000002",
    "万科A": "000002",
    "中信证券": "600030",
    "中信建投": "601066",
    "华泰证券": "601688",
    "海康威视": "002415",
    "中兴通讯": "000063",
    "京东方": "000725",
    "东方财富": "300059",
    "泸州老窖": "000568",
    "恒瑞医药": "600276",
    "迈瑞医疗": "300760",
    "上汽集团": "600104",
    "长安汽车": "000625",
    "长城汽车": "601633",
    "三一重工": "600031",
    "中国神华": "601088",
}


def send_email(to_email: str, subject: str, content: str) -> dict:
    """通过SMTP协议发送邮件

    Args:
        to_email: 收件人邮箱地址
        subject: 邮件主题
        content: 邮件正文内容

    Returns:
        dict: 包含发送结果的字典
            - success: 是否发送成功
            - to_email: 收件人邮箱
            - subject: 邮件主题
            - sent_time: 发送时间
            - error: 错误信息（如果失败）
    """
    # 从环境变量获取邮件配置
    smtp_server = os.environ.get("SMTP_SERVER", "")
    smtp_port = int(os.environ.get("SMTP_PORT", ""))
    smtp_username = os.environ.get("SMTP_USERNAME", "")
    smtp_password = os.environ.get("SMTP_PASSWORD", "")
    from_email = os.environ.get("FROM_EMAIL", smtp_username)

    # 如果没有配置邮件，返回错误
    if not smtp_username or not smtp_password:
        return {
            "success": False,
            "error": "邮件服务未配置，请联系管理员设置 SMTP 环境变量",
            "to_email": to_email,
            "subject": subject,
        }

    try:
        # 构建邮件内容
        message = MIMEText(content, "plain", "utf-8")
        message["From"] = Header(f"AI Agent <{from_email}>")
        message["To"] = Header(to_email)
        message["Subject"] = Header(subject, "utf-8")

        # 连接SMTP服务器并发送（163邮箱建议用SSL 465端口）
        if smtp_port == 465:
            server = smtplib.SMTP_SSL(smtp_server, 465)
        else:
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()  # 启用TLS加密
        server.login(smtp_username, smtp_password)
        server.sendmail(from_email, [to_email], message.as_string())
        server.quit()

        # 返回成功信息
        sent_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return {
            "success": True,
            "to_email": to_email,
            "subject": subject,
            "sent_time": sent_time,
            "message": f"邮件已成功发送给 {to_email}"
        }

    except smtplib.SMTPAuthenticationError:
        return {
            "success": False,
            "error": "邮件认证失败，请检查用户名和密码",
            "to_email": to_email,
            "subject": subject,
        }
    except smtplib.SMTPRecipientsRefused:
        return {
            "success": False,
            "error": f"收件人邮箱地址无效：{to_email}",
            "to_email": to_email,
            "subject": subject,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"邮件发送失败：{str(e)}",
            "to_email": to_email,
            "subject": subject,
        }


def send_dingtalk_message(webhook_url: str, message: str) -> dict:
    """通过钉钉机器人发送消息到群聊

    Args:
        webhook_url: 钉钉群机器人的Webhook地址
        message: 要发送的消息内容

    Returns:
        dict: 包含发送结果的字典
            - success: 是否发送成功
            - message: 消息内容
            - sent_time: 发送时间
            - error: 错误信息（如果失败）
    """
    import json
    import urllib.request
    import urllib.error

    if not webhook_url:
        return {
            "success": False,
            "error": "钉钉Webhook未配置，请联系管理员设置 DINGTALK_WEBHOOK 环境变量",
            "message": message,
        }

    try:
        # 构建请求数据
        data = {
            "msgtype": "text",
            "text": {
                "content": message
            }
        }

        # 发送请求
        req = urllib.request.Request(
            webhook_url,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        with urllib.request.urlopen(req, timeout=10) as response:
            result = json.loads(response.read().decode("utf-8"))

            if result.get("errcode") == 0:
                sent_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                return {
                    "success": True,
                    "message": message,
                    "sent_time": sent_time,
                }
            else:
                return {
                    "success": False,
                    "error": f"钉钉返回错误：{result.get('errmsg', '未知错误')}",
                    "message": message,
                }

    except urllib.error.URLError as e:
        return {
            "success": False,
            "error": f"网络错误：{str(e)}",
            "message": message,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"发送失败：{str(e)}",
            "message": message,
        }
