"""刚性止损 + 黑天鹅兜底"""
from typing import Dict
from loguru import logger
from config import CFG
from core.data_fetcher import DataFetcher


class RiskManager:
    def __init__(self, fetcher: DataFetcher):
        self.fetcher = fetcher

    def calc_position_size(self, symbol: str, account_equity: float,
                           entry_price: float, leverage: int) -> Dict:
        is_new = self.fetcher.is_new_listing(symbol, CFG.NEW_COIN_DAYS)
        max_loss = min(account_equity * CFG.MAX_RISK_PCT, CFG.MAX_LOSS_PER_TRADE_USDT)
        if is_new:
            max_loss *= CFG.NEW_COIN_LOSS_MULTIPLIER
            leverage = min(leverage, CFG.NEW_COIN_LEVERAGE)
        stop_pct = 0.04 if is_new else 0.02
        notional = max_loss / stop_pct
        margin_required = notional / leverage
        margin_required = min(margin_required, account_equity * 0.3)
        notional = margin_required * leverage
        quantity = notional / entry_price
        stop_price = entry_price * (1 - stop_pct)
        return {
            "quantity": quantity,
            "notional": notional,
            "margin": margin_required,
            "leverage": leverage,
            "stop_price": stop_price,
            "max_loss_usdt": max_loss,
            "is_new_coin": is_new,
        }

    def check_black_swan(self, symbol: str) -> bool:
        try:
            ohlcv = self.fetcher.exchange.fetch_ohlcv(symbol, "1m", limit=5)
            highs = [c[2] for c in ohlcv]
            lows = [c[3] for c in ohlcv]
            drawdown = min(lows) / max(highs) - 1
            if drawdown <= -CFG.FLASH_CRASH_PCT:
                logger.warning(f"BLACK SWAN on {symbol}: {drawdown:.2%}")
                return True
        except Exception as e:
            logger.error(f"black swan check failed: {e}")
        return False
