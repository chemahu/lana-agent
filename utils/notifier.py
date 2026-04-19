"""Telegram 通知"""
import httpx
from loguru import logger
from config import CFG

def notify(text: str):
    if not CFG.TELEGRAM_BOT_TOKEN or not CFG.TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{{CFG.TELEGRAM_BOT_TOKEN}}/sendMessage"
        httpx.post(url, json={
            "chat_id": CFG.TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "Markdown",
        }, timeout=5)
    except Exception as e:
        logger.error(f"telegram notify fail: {{e}}")
