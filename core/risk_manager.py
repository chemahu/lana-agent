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
        # sizing 时预留滑点冗余：按 (stop_pct + buffer) 反推 notional，
        # 这样即便 stopMarket 实际成交价比触发价滑落 buffer，单笔账户损失仍 ≈ max_loss。
        slippage_buffer = (
            CFG.NEW_COIN_STOP_SLIPPAGE_BUFFER_PCT if is_new
            else CFG.STOP_SLIPPAGE_BUFFER_PCT
        )
        sizing_stop_pct = stop_pct + slippage_buffer
        notional = max_loss / sizing_stop_pct
        margin_required = notional / leverage
        margin_required = min(margin_required, account_equity * 0.3)
        notional = margin_required * leverage
        quantity = notional / entry_price
        # 实际挂单触发价仍用 stop_pct，触发位置不变（避免被噪音扫损）
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

    def check_black_swan(
        self,
        symbol: str,
        entry_price: float = 0,
        current_price: float = 0,
    ) -> bool:
        """检测黑天鹅（闪跌）。

        优先使用调用方已有的 ``current_price`` 与 ``entry_price`` 直接比较，
        避免每次评估循环都拉取 ``1m`` K 线（持仓多 + 轮询频繁时极易触发币安
        接口权重限流，反而让真正需要逃命的时刻黑天鹅检测静默失效）。

        当 ``current_price`` 与 ``entry_price`` 都 > 0 时：
            判定 ``current_price / entry_price - 1 <= -FLASH_CRASH_PCT``，
            零额外请求即可判断"是否相对开仓价已发生闪崩"。

        否则回退到原始 K 线兜底逻辑（拉取过去 5 根 1m K 线最低价）。
        """
        # —— 快路径：复用调用方已经拿到的现价 ——
        if entry_price > 0 and current_price > 0:
            drawdown = current_price / entry_price - 1
            if drawdown <= -CFG.FLASH_CRASH_PCT:
                logger.warning(
                    f"BLACK SWAN on {symbol}: {drawdown:.2%} "
                    f"(entry={entry_price}, current={current_price})"
                )
                return True
            return False

        # —— 兜底路径：拉取 1m K 线（仅在没有现价时才走，避免高频限流） ——
        try:
            ohlcv = self.fetcher.exchange.fetch_ohlcv(symbol, "1m", limit=5)
            lows = [c[3] for c in ohlcv]
            baseline = entry_price if entry_price > 0 else max(c[2] for c in ohlcv)
            drawdown = min(lows) / baseline - 1
            if drawdown <= -CFG.FLASH_CRASH_PCT:
                logger.warning(f"BLACK SWAN on {symbol}: {drawdown:.2%} (baseline={baseline})")
                return True
        except Exception as e:
            logger.error(f"black swan check failed: {e}")
        return False
