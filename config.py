"""全局配置"""
import os
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()

@dataclass
class TradingConfig:
    # —— API 凭证 ——
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET: str = os.getenv("BINANCE_API_SECRET", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # —— 资金管理 ——
    MAX_LOSS_PER_TRADE_USDT: float = 200.0
    MAX_RISK_PCT: float = 0.01
    DEFAULT_LEVERAGE: int = 5
    NEW_COIN_LEVERAGE: int = 3
    NEW_COIN_LOSS_MULTIPLIER: float = 2.0
    NEW_COIN_DAYS: int = 14

    # —— 选标漏斗 ——
    MIN_SOCIAL_POSTS_1H: int = 50
    MIN_UNIQUE_AUTHORS: int = 30
    MIN_POSTS_GROWTH: float = 1.0
    GAINERS_TOP_N: int = 20
    MIN_PRICE_CHANGE_1H: float = 0.03
    MIN_OI_CHANGE_4H: float = 0.15

    # —— AI 评估 ——
    ENTRY_P_UP_THRESHOLD: float = 0.6
    ENTRY_CONFIDENCE_THRESHOLD: float = 0.65
    HOLD_P_UP_MARGIN: float = 0.1
    SCALE_OUT_PCT_RANGE: tuple = (0.01, 0.05)
    HIGH_ROI_THRESHOLD: float = 1.0

    # —— Cron 调度 ——
    EVALUATION_INTERVAL_MINUTES: int = 15
    AMBIGUOUS_INTERVAL_MINUTES: int = 5
    SCAN_INTERVAL_MINUTES: int = 10

    # —— 黑天鹅 ——
    FLASH_CRASH_PCT: float = 0.15
    FLASH_CRASH_WINDOW_MIN: int = 5
    API_FAILURE_TIMEOUT_SEC: int = 180

    # —— 运行模式 ——
    TESTNET: bool = False
    DRY_RUN: bool = False


CFG = TradingConfig()
