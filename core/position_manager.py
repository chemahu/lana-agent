"""下单 / 止损 / 分批出场 / 全平仓管理器。

封装了所有与币安合约下单相关的逻辑：
- 开多单（含杠杆设置、止损单）
- 按比例减仓（分批止盈）
- 全量平仓
- 查询当前持仓列表

所有操作在 ``DRY_RUN`` 模式下只记录日志，不实际发单。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from loguru import logger

from config import CFG
from core.data_fetcher import DataFetcher
from core.risk_manager import RiskManager


class PositionManager:
    """仓位管理器：开单、减仓、平仓、查仓。

    Parameters
    ----------
    fetcher:
        ``DataFetcher`` 实例，用于取行情与账户信息。
    risk:
        ``RiskManager`` 实例，用于计算仓位大小与止损价。
    """

    def __init__(self, fetcher: DataFetcher, risk: RiskManager) -> None:
        self._fetcher = fetcher
        self._risk = risk
        self._exchange = fetcher.exchange

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _account_equity(self) -> float:
        """获取账户当前 USDT 权益（含未实现盈亏）。"""
        try:
            balance = self._exchange.fetch_balance()
            return float(balance.get("USDT", {}).get("total", 0) or 0)
        except Exception as exc:
            logger.error(f"[PositionManager] fetch_balance failed: {exc}")
            return 0.0

    def _set_leverage(self, symbol: str, leverage: int) -> None:
        try:
            self._exchange.set_leverage(leverage, symbol)
        except Exception as exc:
            logger.warning(f"[PositionManager] set_leverage {symbol} failed: {exc}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_positions(self) -> List[Dict]:
        """返回当前所有非零持仓列表。

        Returns
        -------
        list of dict
            每项包含：``symbol``, ``side``, ``contracts``,
            ``notional``, ``entry_price``, ``unrealized_pnl``, ``leverage``。
        """
        try:
            positions = self._exchange.fetch_positions()
            result = []
            for p in positions:
                contracts = float(p.get("contracts") or p.get("contractSize", 0) or 0)
                if contracts == 0:
                    continue
                result.append(
                    {
                        "symbol": p.get("symbol", ""),
                        "side": p.get("side", "long"),
                        "contracts": contracts,
                        "notional": float(p.get("notional") or p.get("initialMargin", 0) or 0),
                        "entry_price": float(p.get("entryPrice") or p.get("averagePrice", 0) or 0),
                        "unrealized_pnl": float(p.get("unrealizedPnl", 0) or 0),
                        "leverage": int(p.get("leverage", CFG.DEFAULT_LEVERAGE) or CFG.DEFAULT_LEVERAGE),
                        "mark_price": float(p.get("markPrice", 0) or 0),
                    }
                )
            return result
        except Exception as exc:
            logger.error(f"[PositionManager] get_positions failed: {exc}")
            return []

    def open_long(
        self,
        symbol: str,
        leverage: Optional[int] = None,
    ) -> Optional[Dict]:
        """开多单。

        Parameters
        ----------
        symbol:
            合约符号，如 ``"BTC/USDT:USDT"``。
        leverage:
            杠杆倍数，默认取 ``CFG.DEFAULT_LEVERAGE``。

        Returns
        -------
        dict or None
            交易所返回的订单字典；若 ``DRY_RUN`` 模式则返回模拟字典；
            失败则返回 ``None``。
        """
        lev = leverage or CFG.DEFAULT_LEVERAGE
        equity = self._account_equity()
        if equity <= 0:
            logger.warning(f"[PositionManager] no equity, skip open_long {symbol}")
            return None

        ticker = self._exchange.fetch_ticker(symbol)
        entry_price = float(ticker.get("last") or ticker.get("close") or 0)
        if entry_price <= 0:
            logger.error(f"[PositionManager] invalid price for {symbol}: {entry_price}")
            return None

        size_info = self._risk.calc_position_size(symbol, equity, entry_price, lev)
        qty = size_info["quantity"]
        stop_price = size_info["stop_price"]
        lev = size_info["leverage"]

        logger.info(
            f"[PositionManager] open_long {symbol} qty={qty:.6f} "
            f"entry={entry_price} stop={stop_price:.4f} lev={lev}x "
            f"(dry_run={CFG.DRY_RUN})"
        )

        if CFG.DRY_RUN:
            return {
                "symbol": symbol,
                "side": "buy",
                "qty": qty,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "leverage": lev,
                "dry_run": True,
            }

        try:
            self._set_leverage(symbol, lev)
            # market entry order
            order = self._exchange.create_market_buy_order(symbol, qty)
            # stop-loss order
            try:
                self._exchange.create_order(
                    symbol=symbol,
                    type="STOP_MARKET",
                    side="sell",
                    amount=qty,
                    params={"stopPrice": stop_price, "closePosition": True},
                )
            except Exception as exc:
                logger.warning(f"[PositionManager] stop-loss order failed for {symbol}: {exc}")
            return order
        except Exception as exc:
            logger.error(f"[PositionManager] open_long failed for {symbol}: {exc}")
            return None

    def scale_out(self, symbol: str, pct: float) -> Optional[Dict]:
        """按百分比减仓。

        Parameters
        ----------
        symbol:
            合约符号。
        pct:
            减仓比例 ``(0, 1]``，例如 ``0.5`` 表示减半。

        Returns
        -------
        dict or None
            交易所返回的订单字典，或 ``None`` 表示失败/空仓。
        """
        pct = max(0.0, min(1.0, pct))
        positions = self.get_positions()
        pos = next((p for p in positions if p["symbol"] == symbol), None)
        if pos is None:
            logger.warning(f"[PositionManager] scale_out: no position for {symbol}")
            return None

        qty = pos["contracts"] * pct
        if qty <= 0:
            return None

        logger.info(
            f"[PositionManager] scale_out {symbol} qty={qty:.6f} ({pct:.0%}) "
            f"(dry_run={CFG.DRY_RUN})"
        )

        if CFG.DRY_RUN:
            return {"symbol": symbol, "side": "sell", "qty": qty, "dry_run": True}

        try:
            return self._exchange.create_market_sell_order(
                symbol, qty, params={"reduceOnly": True}
            )
        except Exception as exc:
            logger.error(f"[PositionManager] scale_out failed for {symbol}: {exc}")
            return None

    def close_position(self, symbol: str) -> Optional[Dict]:
        """全量平仓指定 symbol。

        Returns
        -------
        dict or None
            交易所返回的订单字典，或 ``None``。
        """
        positions = self.get_positions()
        pos = next((p for p in positions if p["symbol"] == symbol), None)
        if pos is None:
            logger.info(f"[PositionManager] close_position: no open position for {symbol}")
            return None

        qty = pos["contracts"]
        side = "sell" if pos["side"] == "long" else "buy"

        logger.info(
            f"[PositionManager] close_position {symbol} qty={qty} side={side} "
            f"(dry_run={CFG.DRY_RUN})"
        )

        if CFG.DRY_RUN:
            return {"symbol": symbol, "side": side, "qty": qty, "dry_run": True, "closed": True}

        try:
            return self._exchange.create_market_order(
                symbol, side, qty, params={"reduceOnly": True}
            )
        except Exception as exc:
            logger.error(f"[PositionManager] close_position failed for {symbol}: {exc}")
            return None

    def close_all(self) -> List[Dict]:
        """全量平所有持仓。返回每笔平仓结果列表。"""
        results = []
        for pos in self.get_positions():
            res = self.close_position(pos["symbol"])
            if res is not None:
                results.append(res)
        return results
