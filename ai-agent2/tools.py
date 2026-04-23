import os
import re
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr

import requests


CITY_CODE_MAP: dict[str, str] = {}
FALLBACK_CITY_CODE_MAP = {
    "北京": "101010100",
    "上海": "101020100",
    "广州": "101280101",
    "深圳": "101280601",
    "杭州": "101210101",
    "重庆": "101040100",
    "成都": "101270101",
    "武汉": "101200101",
}


def load_city_codes() -> None:
    try:
        response = requests.get("http://cdn.sojson.com/_city.json", timeout=5)
        response.raise_for_status()
        for city in response.json():
            city_name = city.get("city_name")
            city_code = city.get("city_code")
            if city_name and city_code:
                CITY_CODE_MAP[city_name] = city_code
    except Exception:
        CITY_CODE_MAP.update(FALLBACK_CITY_CODE_MAP)


def get_weather_json(city_name: str) -> dict:
    if not CITY_CODE_MAP:
        load_city_codes()

    city_name = (city_name or "").strip()
    if city_name not in CITY_CODE_MAP:
        return {"code": 404, "msg": f"未找到城市：{city_name}", "data": None}

    try:
        response = requests.get(
            f"http://t.weather.sojson.com/api/weather/city/{CITY_CODE_MAP[city_name]}",
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()
        return {
            "code": 200,
            "msg": "查询成功",
            "data": {
                "city": data["cityInfo"]["city"],
                "temperature": data["data"]["wendu"],
                "humidity": data["data"]["shidu"],
                "today_weather": data["data"]["forecast"][0]["type"],
                "low_temperature": data["data"]["forecast"][0]["low"],
                "high_temperature": data["data"]["forecast"][0]["high"],
            },
        }
    except Exception as error:
        return {"code": 500, "msg": f"天气查询失败：{error}", "data": None}


def get_current_time() -> dict:
    now = datetime.now()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return {
        "code": 200,
        "msg": "查询成功",
        "data": {
            "current_time": now.strftime("%H:%M:%S"),
            "current_date": now.strftime("%Y-%m-%d"),
            "weekday": now.strftime("%A"),
            "weekday_cn": weekdays[now.weekday()],
            "full_datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }


def get_stock_price_cn(ticker: str) -> dict:
    ticker = str(ticker or "").strip()
    if not ticker:
        return {"code": 400, "msg": "股票代码不能为空", "data": None}

    try:
        if ticker.startswith("6"):
            code = f"sh{ticker}"
        else:
            code = f"sz{ticker}"

        url = f"https://hq.sinajs.cn/list={code}"
        headers = {"Referer": "https://finance.sina.com/"}
        res = requests.get(url, headers=headers, timeout=5)
        res.raise_for_status()

        data = res.text.split('"')[1].split(',')

        if len(data) < 3:
            return {"code": 404, "msg": f"未找到股票：{ticker}", "data": None}

        result = {
            "name": data[0],
            "ticker": ticker,
            "current_price": float(data[3]),
            "open": float(data[1]),
            "last_close": float(data[2]),
            "high": float(data[4]),
            "low": float(data[5]),
            "change_percent": round((float(data[3]) - float(data[2])) / float(data[2]) * 100, 2),
        }
        return {"code": 200, "msg": "查询成功", "data": result}
    except Exception as error:
        return {"code": 500, "msg": f"股价查询失败：{error}", "data": None}


def is_valid_email(email: str) -> bool:
    return re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", email or "") is not None


def send_email(to_email: str, subject: str, body: str) -> dict:
    if not is_valid_email(to_email):
        return {"code": 400, "msg": f"邮箱格式不合法：{to_email}", "data": None}
    if not subject or not body:
        return {"code": 400, "msg": "邮件主题和正文不能为空", "data": None}

    smtp_server = os.environ.get("SMTP_SERVER", "smtp.qq.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_username = os.environ.get("SMTP_USERNAME", "")
    smtp_password = os.environ.get("SMTP_PASSWORD", "")
    from_email = os.environ.get("FROM_EMAIL", "")
    from_name = os.environ.get("FROM_NAME", "AI Agent")

    if not smtp_username or not smtp_password or not from_email:
        return {"code": 500, "msg": "邮件服务配置不完整", "data": None}

    try:
        message = MIMEMultipart()
        message["From"] = formataddr((from_name, from_email))
        message["To"] = to_email
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain", "utf-8"))

        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.sendmail(from_email, [to_email], message.as_string())
        server.quit()

        return {
            "code": 200,
            "msg": "邮件发送成功",
            "data": {
                "to_email": to_email,
                "subject": subject,
                "sent_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        }
    except Exception as error:
        return {"code": 500, "msg": f"邮件发送失败：{error}", "data": None}


def send_dingtalk_message(
    webhook_url: str, content: str, at_mobiles: list[str] | None = None, is_at_all: bool = False
) -> dict:
    if not webhook_url or not content:
        return {"code": 400, "msg": "Webhook 和消息内容不能为空", "data": None}

    try:
        response = requests.post(
            webhook_url,
            json={
                "msgtype": "text",
                "text": {"content": content},
                "at": {"atMobiles": at_mobiles or [], "isAtAll": is_at_all},
            },
            timeout=10,
        )
        response.raise_for_status()
        result = response.json()

        if result.get("errcode") == 0:
            return {
                "code": 200,
                "msg": "钉钉消息发送成功",
                "data": {"sent_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
            }

        return {
            "code": 500,
            "msg": f"钉钉消息发送失败：{result.get('errmsg', '未知错误')}",
            "data": result,
        }
    except Exception as error:
        return {"code": 500, "msg": f"钉钉消息发送失败：{error}", "data": None}
