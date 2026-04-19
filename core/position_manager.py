"""仓位管理器：开仓、止损、分批止盈、全平。

职责：
- ``open_long``    — 按 RiskManager 计算仓位，在交易所下市价多单 + 止损单。
- ``close_position`` — 平全仓。
- ``scale_out``    — 按比例分批减仓。
- ``get_positions`` — 获取当前所有持仓。
- ``get_equity``   — 获取账户余额。

DRY_RUN 模式下所有操作仅打印日志，不实际下单。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from loguru import logger

from config import CFG
from core.data_fetcher import DataFetcher
from core.risk_manager import RiskManager


class PositionManager:
    """仓位管理器。

    Parameters
    ----------
    fetcher:
        ``DataFetcher`` 实例（提供交易所连接和行情）。
    risk:
        ``RiskManager`` 实例（计算仓位大小 / 止损价）。
    """

    def __init__(self, fetcher: DataFetcher, risk: RiskManager) -> None:
        self._fetcher = fetcher
        self._risk = risk

    # ------------------------------------------------------------------
    # Account helpers
    # ------------------------------------------------------------------

    def get_equity(self) -> float:
        """返回账户 USDT 权益（保证金余额）。"""
        try:
            balance = self._fetcher.exchange.fetch_balance()
            usdt = balance.get("USDT") or {}
            free = usdt.get("free")
            if free is not None:
                return float(free)
            total = (balance.get("total") or {}).get("USDT")
            return float(total) if total is not None else 0.0
        except Exception as exc:
            logger.error(f"[PositionManager] get_equity failed: {exc}")
            return 0.0

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_positions(self) -> List[Dict]:
        """返回当前所有非零净持仓列表。

        每条记录至少包含 ``symbol``、``contracts``、``unrealizedPnl``、
        ``entryPrice``、``side``。
        """
        try:
            positions = self._fetcher.exchange.fetch_positions()
            return [
                p for p in positions
                if p.get("contracts") and float(p["contracts"]) != 0
            ]
        except Exception as exc:
            logger.error(f"[PositionManager] get_positions failed: {exc}")
            return []

    def _get_position(self, symbol: str) -> Optional[Dict]:
        """获取某 symbol 的持仓，不存在返回 None。"""
        for p in self.get_positions():
            if p.get("symbol") == symbol:
                return p
        return None

    # ------------------------------------------------------------------
    # Open long
    # ------------------------------------------------------------------

    def open_long(self, symbol: str) -> Optional[Dict]:
        """开多仓。

        Parameters
        ----------
        symbol:
            如 ``"BTC/USDT:USDT"``。

        Returns
        -------
        dict or None
            交易所返回的订单信息；DRY_RUN 或失败时返回 None。
        """
        try:
            equity = self.get_equity()
            if equity <= 0:
                logger.warning(f"[PositionManager] zero equity, skip open {symbol}")
                return None

            ticker = self._fetcher.exchange.fetch_ticker(symbol)
            entry_price: float = ticker["last"]
            leverage = CFG.DEFAULT_LEVERAGE

            sizing = self._risk.calc_position_size(
                symbol, equity, entry_price, leverage
            )
            quantity: float = sizing["quantity"]
            stop_price: float = sizing["stop_price"]
            actual_leverage: int = sizing["leverage"]

            if CFG.DRY_RUN:
                logger.info(
                    f"[DRY_RUN] OPEN LONG {symbol} qty={quantity:.4f} "
                    f"entry={entry_price} stop={stop_price} lev={actual_leverage}x "
                    f"margin={sizing['margin']:.2f} USDT"
                )
                return {"symbol": symbol, "dry_run": True, "quantity": quantity}

            # 设置杠杆
            try:
                self._fetcher.exchange.set_leverage(actual_leverage, symbol)
            except Exception as exc:
                logger.warning(f"[PositionManager] set_leverage failed: {exc}")

            # 市价多单
            order = self._fetcher.exchange.create_market_order(
                symbol, "buy", quantity
            )
            logger.info(
                f"[PositionManager] OPENED LONG {symbol} qty={quantity:.4f} "
                f"@ ~{entry_price} lev={actual_leverage}x "
                f"orderId={order.get('id')}"
            )

            # 止损单
            try:
                sl_order = self._fetcher.exchange.create_order(
                    symbol=symbol,
                    type="STOP_MARKET",
                    side="sell",
                    amount=quantity,
                    params={"stopPrice": stop_price, "reduceOnly": True},
                )
                logger.info(
                    f"[PositionManager] stop-loss placed @ {stop_price} "
                    f"orderId={sl_order.get('id')}"
                )
            except Exception as exc:
                logger.error(
                    f"[PositionManager] failed to place stop-loss for {symbol}: {exc}"
                )

            return order

        except Exception as exc:
            logger.error(f"[PositionManager] open_long failed for {symbol}: {exc}")
            return None

    # ------------------------------------------------------------------
    # Close / scale-out
    # ------------------------------------------------------------------

    def close_position(self, symbol: str) -> Optional[Dict]:
        """市价全平某 symbol 的持仓。

        Returns
        -------
        dict or None
            平仓订单；DRY_RUN 或无持仓时返回 None。
        """
        try:
            position = self._get_position(symbol)
            if not position:
                logger.info(f"[PositionManager] no position to close for {symbol}")
                return None

            quantity = abs(float(position["contracts"]))
            side = position.get("side", "long")
            close_side = "sell" if side == "long" else "buy"

            if CFG.DRY_RUN:
                logger.info(
                    f"[DRY_RUN] CLOSE {side.upper()} {symbol} qty={quantity:.4f}"
                )
                return {"symbol": symbol, "dry_run": True, "action": "close"}

            order = self._fetcher.exchange.create_market_order(
                symbol, close_side, quantity, params={"reduceOnly": True}
            )
            logger.info(
                f"[PositionManager] CLOSED {side.upper()} {symbol} qty={quantity:.4f} "
                f"orderId={order.get('id')}"
            )
            return order

        except Exception as exc:
            logger.error(f"[PositionManager] close_position failed for {symbol}: {exc}")
            return None

    def scale_out(self, symbol: str, pct: float) -> Optional[Dict]:
        """按比例分批减仓。

        Parameters
        ----------
        symbol:
            交易对符号。
        pct:
            减仓比例，取值 (0, 1]，如 0.3 表示平掉 30% 仓位。

        Returns
        -------
        dict or None
            减仓订单；失败或无仓位时返回 None。
        """
        pct = max(0.0, min(1.0, pct))
        if pct <= 0:
            return None

        try:
            position = self._get_position(symbol)
            if not position:
                logger.info(f"[PositionManager] no position to scale out for {symbol}")
                return None

            total_qty = abs(float(position["contracts"]))
            reduce_qty = total_qty * pct
            if reduce_qty <= 0:
                return None

            side = position.get("side", "long")
            close_side = "sell" if side == "long" else "buy"

            if CFG.DRY_RUN:
                logger.info(
                    f"[DRY_RUN] SCALE_OUT {symbol} pct={pct:.0%} "
                    f"qty={reduce_qty:.4f}/{total_qty:.4f}"
                )
                return {
                    "symbol": symbol,
                    "dry_run": True,
                    "action": "scale_out",
                    "pct": pct,
                }

            order = self._fetcher.exchange.create_market_order(
                symbol, close_side, reduce_qty, params={"reduceOnly": True}
            )
            logger.info(
                f"[PositionManager] SCALE_OUT {symbol} pct={pct:.0%} "
                f"qty={reduce_qty:.4f} orderId={order.get('id')}"
            )
            return order

        except Exception as exc:
            logger.error(f"[PositionManager] scale_out failed for {symbol}: {exc}")
            return None
