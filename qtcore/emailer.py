"""
邮件发送模块
============
支持 163/QQ 等 SMTP(SSL)。

密钥优先从环境变量读取(可在项目根目录 .env 中配置):
    EMAIL_SENDER / EMAIL_SENDER_NAME / EMAIL_AUTH_CODE
    EMAIL_SMTP_HOST / EMAIL_SMTP_PORT / EMAIL_RECIPIENTS(逗号分隔)
未配置的项回退到 config/email_config.json(该文件已加入 .gitignore)。
"""

from __future__ import annotations

import json
import mimetypes
import os
import smtplib
from email.header import Header
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
from pathlib import Path
from typing import Any

from qtcore.dotenv import load_dotenv


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "email_config.json"


def load_email_config(path: Path | None = None) -> dict[str, Any]:
    """读取邮件配置: 环境变量优先, config/email_config.json 兜底。"""
    load_dotenv()
    p = path or DEFAULT_CONFIG_PATH
    data: dict[str, Any] = {}
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))

    def pick(env_name: str, json_key: str) -> Any:
        value = os.environ.get(env_name)
        return value if value else data.get(json_key)

    sender = str(pick("EMAIL_SENDER", "sender") or "")
    auth_code = str(pick("EMAIL_AUTH_CODE", "auth_code") or "")
    recipients_raw = pick("EMAIL_RECIPIENTS", "recipients")
    if isinstance(recipients_raw, str):
        recipients = [r.strip() for r in recipients_raw.split(",") if r.strip()]
    else:
        recipients = [str(r) for r in (recipients_raw or [])]

    cfg = {
        "sender": sender,
        "sender_name": str(pick("EMAIL_SENDER_NAME", "sender_name") or "QuantTrader Auto"),
        "auth_code": auth_code,
        "smtp_host": str(pick("EMAIL_SMTP_HOST", "smtp_host") or "smtp.163.com"),
        "smtp_port": int(pick("EMAIL_SMTP_PORT", "smtp_port") or 465),
        "recipients": recipients,
    }
    if not cfg["sender"] or not cfg["auth_code"]:
        raise RuntimeError(
            "缺少 sender 或 auth_code: 请在 .env 配置 EMAIL_SENDER / EMAIL_AUTH_CODE, "
            "或填写 config/email_config.json"
        )
    if not cfg["recipients"]:
        raise RuntimeError(
            "缺少 recipients(收件人列表): 请在 .env 配置 EMAIL_RECIPIENTS, "
            "或填写 config/email_config.json"
        )
    return cfg


def send_email(
    subject: str,
    body: str,
    config: dict[str, Any] | None = None,
    html: bool = False,
    attachments: list[Path | str] | None = None,
) -> dict[str, Any]:
    """发送一封邮件; 失败抛出异常。日志不回显授权码。"""
    cfg = config or load_email_config()
    smtp_host = str(cfg.get("smtp_host", "smtp.163.com"))
    smtp_port = int(cfg.get("smtp_port", 465))
    sender = str(cfg["sender"])
    auth_code = str(cfg["auth_code"])
    recipients = [str(r) for r in cfg["recipients"]]

    if attachments:
        msg = MIMEMultipart("mixed")
        msg.attach(MIMEText(body, "html" if html else "plain", "utf-8"))
        for att in attachments:
            att_path = Path(att)
            ctype, _ = mimetypes.guess_type(str(att_path))
            maintype, subtype = (ctype or "application/octet-stream").split("/", 1)
            if maintype == "image":
                part: Any = MIMEImage(att_path.read_bytes(), _subtype=subtype)
            else:
                part = MIMEApplication(att_path.read_bytes(), _subtype=subtype)
            part.add_header("Content-Disposition", "attachment", filename=att_path.name)
            msg.attach(part)
    else:
        msg = MIMEText(body, "html" if html else "plain", "utf-8")
    msg["Subject"] = Header(subject, "utf-8")
    msg["From"] = formataddr((str(cfg.get("sender_name", "QuantTrader Auto")), sender))
    msg["To"] = ", ".join(recipients)

    with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=30) as server:
        server.login(sender, auth_code)
        server.sendmail(sender, recipients, msg.as_string())

    return {"from": sender, "to": recipients, "subject": subject}
