"""刚性止损 + 黑天鹅兜底"""
from typing import Dict
from loguru import logger
from config import CFG
from core.data_fetcher import DataFetcher


class RiskManager:
    def __init__(self, fetcher: DataFetcher):
        self.fetcher = fetcher

    def _calc_atr_stop_pct(self, symbol: str, entry_price: float) -> float:
        """基于 ATR 计算自适应止损幅度，避免止损过窄被频繁扫损。

        用过去 ``CFG.ATR_PERIOD`` 根 1h K 线的平均真实幅度（ATR）乘以
        ``CFG.ATR_MULTIPLIER``，再除以入场价，得到止损比例。结果夹紧在
        ``[CFG.MIN_STOP_PCT, CFG.MAX_STOP_PCT]`` 区间内。

        若 ATR 拉取失败，回退到最小止损比例 ``CFG.MIN_STOP_PCT``。
        """
        try:
            ohlcv = self.fetcher.exchange.fetch_ohlcv(
                symbol, "1h", limit=CFG.ATR_PERIOD + 1
            )
            trs = []
            for i in range(1, len(ohlcv)):
                high = ohlcv[i][2]
                low = ohlcv[i][3]
                prev_close = ohlcv[i - 1][4]
                tr = max(
                    high - low,
                    abs(high - prev_close),
                    abs(low - prev_close),
                )
                trs.append(tr)
            if not trs or entry_price <= 0:
                return CFG.MIN_STOP_PCT
            atr = sum(trs) / len(trs)
            stop_pct = (atr * CFG.ATR_MULTIPLIER) / entry_price
            return max(CFG.MIN_STOP_PCT, min(stop_pct, CFG.MAX_STOP_PCT))
        except Exception as exc:
            logger.warning(f"[RiskManager] ATR stop calc failed for {symbol}: {exc}")
            return CFG.MIN_STOP_PCT

    def calc_position_size(self, symbol: str, account_equity: float,
                           entry_price: float, leverage: int) -> Dict:
        """计算开仓仓位大小并确定止损触发价格。

        止损触发价基于 **合约价格相对入场价的下跌幅度**（stop_pct），
        与杠杆倍数无关：

            stop_price = entry_price × (1 - stop_pct)

        止损幅度（stop_pct）由 ATR 自适应计算，夹紧在
        ``[MIN_STOP_PCT, MAX_STOP_PCT]``：
          - 普通币：ATR × ATR_MULTIPLIER / entry_price，下限 1%，上限 8%
          - 新币（≤14 天）：同上但下限为 NEW_COIN_MIN_STOP_PCT（4%）

        示例（普通币，5× 杠杆，入场价 100 USDT，ATR=2 USDT，1.5× 倍数）：
            ATR 止损幅度 = 2 × 1.5 / 100 = 3%
            止损触发价 = 100 × (1 - 0.03) = 97 USDT  ← 价格下跌 3% 触发
            若止损成交，保证金亏损 ≈ notional × 3% = margin × (3% × 5) = margin × 15%

        账户净值层面的最大亏损上限由 max_loss 控制（= min(账户净值 ×
        MAX_RISK_PCT, MAX_LOSS_PER_TRADE_USDT)），通过反推仓位 notional 来
        实现：notional = max_loss / stop_pct，从而保证止损触发时账户实际
        亏损 ≈ max_loss，而不是账户净值的某个固定百分比乘以杠杆倍数。

        总结：
          - 止损"幅度"= ATR 自适应合约价格跌幅（与杠杆无关）。
          - 止损"金额"= 单笔 max_loss（账户净值 × 1%，上限 200 USDT），
            通过仓位 sizing 保证，而非依赖杠杆计算。
        """
        is_new = self.fetcher.is_new_listing(symbol, CFG.NEW_COIN_DAYS)
        max_loss = min(account_equity * CFG.MAX_RISK_PCT, CFG.MAX_LOSS_PER_TRADE_USDT)
        if is_new:
            max_loss *= CFG.NEW_COIN_LOSS_MULTIPLIER
            leverage = min(leverage, CFG.NEW_COIN_LEVERAGE)
        # ATR 自适应止损幅度：新币下限为 4%，普通币下限为 MIN_STOP_PCT
        atr_stop_pct = self._calc_atr_stop_pct(symbol, entry_price)
        if is_new:
            stop_pct = max(CFG.NEW_COIN_MIN_STOP_PCT, atr_stop_pct)
        else:
            stop_pct = atr_stop_pct
        logger.info(
            f"[RiskManager] {symbol} stop_pct={stop_pct:.2%} "
            f"(ATR-based, is_new={is_new})"
        )
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
        # 止损挂单触发价：入场价下跌 stop_pct（价格跌幅，非账户亏损比例）
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
