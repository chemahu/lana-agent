"""仓位管理：开仓、平仓、查询持仓，通过 Binance 合约接口执行。"""
from typing import Dict, List, Optional

from loguru import logger

from config import CFG
from core.data_fetcher import DataFetcher
from core.risk_manager import RiskManager


class PositionManager:
    """封装 Binance USDM 合约的开仓 / 平仓 / 持仓查询逻辑。

    当 CFG.DRY_RUN 为 True 时，所有交易指令仅打日志而不实际下单。
    """

    def __init__(self, fetcher: DataFetcher, risk: RiskManager) -> None:
        self.fetcher = fetcher
        self.risk = risk
        self._exchange = fetcher.exchange

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_account_equity(self) -> float:
        """获取账户 USDT 保证金余额。"""
        try:
            balance = self._exchange.fetch_balance()
            return float(balance.get("USDT", {}).get("free", 0) or 0)
        except Exception as exc:
            logger.error(f"[PositionManager] fetch_balance failed: {exc}")
            return 0.0

    def _set_leverage(self, symbol: str, leverage: int) -> None:
        """设置合约杠杆倍数。"""
        try:
            market_id = self._exchange.market(symbol)["id"]
            self._exchange.fapiPrivatePostLeverage(
                {"symbol": market_id, "leverage": leverage}
            )
        except Exception as exc:
            logger.warning(f"[PositionManager] set_leverage failed for {symbol}: {exc}")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_positions(self) -> List[Dict]:
        """返回当前所有持仓（仅含非零数量的多头）。

        每条记录包含：symbol, side, size, entry_price, unrealized_pnl
        """
        try:
            positions = self._exchange.fetch_positions()
            result = []
            for pos in positions:
                contracts = float(pos.get("contracts") or pos.get("positionAmt") or 0)
                if contracts <= 0:
                    continue
                side = pos.get("side", "").lower()
                if side not in ("long", "buy", ""):
                    continue
                result.append({
                    "symbol": pos["symbol"],
                    "side": side or "long",
                    "size": contracts,
                    "entry_price": float(pos.get("entryPrice") or pos.get("info", {}).get("entryPrice", 0)),
                    "unrealized_pnl": float(pos.get("unrealizedPnl") or 0),
                    "leverage": int(pos.get("leverage") or CFG.DEFAULT_LEVERAGE),
                    "notional": float(pos.get("notional") or 0),
                })
            return result
        except Exception as exc:
            logger.error(f"[PositionManager] get_positions failed: {exc}")
            return []

    def get_position(self, symbol: str) -> Optional[Dict]:
        """返回指定标的的持仓，未持仓则返回 None。"""
        positions = self.get_positions()
        for pos in positions:
            if pos["symbol"] == symbol:
                return pos
        return None

    # ------------------------------------------------------------------
    # Open
    # ------------------------------------------------------------------

    def open_long(self, symbol: str) -> Optional[Dict]:
        """做多 symbol：计算仓位大小 → 设置杠杆 → 市价开仓 → 挂止损单。

        Returns
        -------
        dict | None
            成功时返回订单信息字典；失败或 dry-run 时返回 None。
        """
        try:
            equity = self._get_account_equity()
            if equity <= 0:
                logger.warning(f"[PositionManager] insufficient equity for {symbol}")
                return None

            # 获取最新价格
            ticker = self._exchange.fetch_ticker(symbol)
            entry_price = ticker["last"]
            if not entry_price:
                logger.warning(f"[PositionManager] cannot get price for {symbol}")
                return None

            leverage = (
                CFG.NEW_COIN_LEVERAGE
                if self.fetcher.is_new_listing(symbol, CFG.NEW_COIN_DAYS)
                else CFG.DEFAULT_LEVERAGE
            )
            sizing = self.risk.calc_position_size(symbol, equity, entry_price, leverage)
            quantity = sizing["quantity"]

            if CFG.DRY_RUN:
                logger.info(
                    f"[PositionManager][DRY-RUN] would open long {symbol} "
                    f"qty={quantity:.6f} @ {entry_price} "
                    f"notional={sizing['notional']:.2f} USDT"
                )
                return None

            self._set_leverage(symbol, leverage)

            # 市价多单
            order = self._exchange.create_order(
                symbol=symbol,
                type="market",
                side="buy",
                amount=quantity,
            )
            logger.info(
                f"[PositionManager] OPEN LONG {symbol} "
                f"qty={quantity:.6f} @ {entry_price} "
                f"orderId={order.get('id')}"
            )

            # 挂止损单
            try:
                stop_order = self._exchange.create_order(
                    symbol=symbol,
                    type="stop_market",
                    side="sell",
                    amount=quantity,
                    params={"stopPrice": sizing["stop_price"], "reduceOnly": True},
                )
                logger.info(
                    f"[PositionManager] stop-loss set for {symbol} "
                    f"@ {sizing['stop_price']:.6f} orderId={stop_order.get('id')}"
                )
            except Exception as sl_exc:
                logger.warning(f"[PositionManager] stop-loss order failed for {symbol}: {sl_exc}")

            return order

        except Exception as exc:
            logger.error(f"[PositionManager] open_long failed for {symbol}: {exc}")
            return None

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def close_long(self, symbol: str, quantity: Optional[float] = None) -> Optional[Dict]:
        """平多单（全部或部分）。

        Parameters
        ----------
        symbol:
            合约标的，例如 ``"BTC/USDT:USDT"``。
        quantity:
            平仓数量；None 表示全部平仓。

        Returns
        -------
        dict | None
            成功时返回订单字典；失败或 dry-run 时返回 None。
        """
        try:
            pos = self.get_position(symbol)
            if pos is None:
                logger.warning(f"[PositionManager] no open position for {symbol}")
                return None

            close_qty = quantity if quantity is not None else pos["size"]

            if CFG.DRY_RUN:
                logger.info(
                    f"[PositionManager][DRY-RUN] would close long {symbol} qty={close_qty:.6f}"
                )
                return None

            order = self._exchange.create_order(
                symbol=symbol,
                type="market",
                side="sell",
                amount=close_qty,
                params={"reduceOnly": True},
            )
            logger.info(
                f"[PositionManager] CLOSE LONG {symbol} "
                f"qty={close_qty:.6f} orderId={order.get('id')}"
            )
            return order

        except Exception as exc:
            logger.error(f"[PositionManager] close_long failed for {symbol}: {exc}")
            return None

    def close_all(self) -> None:
        """平掉所有持仓（紧急清仓）。"""
        for pos in self.get_positions():
            self.close_long(pos["symbol"])
