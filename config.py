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
    SCALE_OUT_PCT_RANGE: tuple = (0.1, 0.9)
    HIGH_ROI_THRESHOLD: float = 1.0
    BREAKEVEN_ROI_TRIGGER: float = 0.05  # 浮盈达此比例时将止损移至保本价

    # —— 仓位限制 ——
    MAX_OPEN_POSITIONS: int = 5

    # —— Cron 调度 ——
    EVALUATION_INTERVAL_MINUTES: int = 15
    AMBIGUOUS_INTERVAL_MINUTES: int = 5
    SCAN_INTERVAL_MINUTES: int = 10

    # —— 黑天鹅 ——
    FLASH_CRASH_PCT: float = 0.15
    FLASH_CRASH_WINDOW_MIN: int = 5
    API_FAILURE_TIMEOUT_SEC: int = 180

    # —— 止损滑点冗余 ——
    # sizing 时按 (stop_pct + buffer) 反推仓位，确保在 stopMarket 出现滑点时
    # 最坏单笔账户回撤仍约等于 MAX_RISK_PCT；挂单触发价仍用 stop_pct，触发位置不变。
    STOP_SLIPPAGE_BUFFER_PCT: float = 0.005
    NEW_COIN_STOP_SLIPPAGE_BUFFER_PCT: float = 0.015

    # —— 自适应止损（ATR based）——
    # 用过去 ATR_PERIOD 根 1h K 线的均幅（ATR）× ATR_MULTIPLIER / 入场价 作为止损幅度，
    # 避免固定 2% 在高波动行情中被频繁扫损。结果被夹紧在 [MIN_STOP_PCT, MAX_STOP_PCT]。
    # 新币强制以 NEW_COIN_MIN_STOP_PCT 为下限。
    ATR_PERIOD: int = 14
    ATR_MULTIPLIER: float = 1.5
    MIN_STOP_PCT: float = 0.01
    MAX_STOP_PCT: float = 0.08
    NEW_COIN_MIN_STOP_PCT: float = 0.04

    # —— 订单执行二次验证 ——
    # open_long / close_long 下单后轮询持仓，确认成交落地；未能确认时发出告警而不阻塞。
    ORDER_VERIFY_RETRIES: int = 3
    ORDER_VERIFY_DELAY_SEC: float = 2.0

    # —— 运行模式 ——
    TESTNET: bool = False
    DRY_RUN: bool = False


CFG = TradingConfig()
