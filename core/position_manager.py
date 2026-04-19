"""下单、平仓、持仓管理"""
from typing import Dict, List, Optional
from loguru import logger
from config import CFG
from core.data_fetcher import DataFetcher
from core.risk_manager import RiskManager


class PositionManager:
    """管理合约账户的开仓 / 平仓 / 持仓查询。"""

    def __init__(self, fetcher: DataFetcher, risk: RiskManager):
        self.fetcher = fetcher
        self.risk = risk
        self._exchange = fetcher.exchange

    # ------------------------------------------------------------------
    # Account helpers
    # ------------------------------------------------------------------
    def _get_equity(self) -> float:
        try:
            balance = self._exchange.fetch_balance()
            return float(balance["total"].get("USDT", 0))
        except Exception as e:
            logger.error(f"[PositionManager] fetch_balance failed: {e}")
            return 0.0

    # ------------------------------------------------------------------
    # Position queries
    # ------------------------------------------------------------------
    def get_positions(self) -> List[Dict]:
        """
        Returns a list of open positions. Each dict contains at minimum:
          symbol (str), side (str), size (float),
          entry_price (float), unrealized_pnl (float),
          unrealized_pnl_pct (float)
        """
        try:
            raw = self._exchange.fetch_positions()
            result = []
            for p in raw:
                contracts = float(p.get("contracts") or 0)
                if contracts == 0:
                    continue
                entry_price = float(p.get("entryPrice") or p.get("entry_price") or 0)
                mark_price = float(p.get("markPrice") or p.get("mark_price") or entry_price)
                pnl = float(p.get("unrealizedPnl") or p.get("unrealized_pnl") or 0)
                side = p.get("side", "long")
                if entry_price > 0:
                    pnl_pct = (mark_price / entry_price - 1) if side == "long" else (entry_price / mark_price - 1)
                else:
                    pnl_pct = 0.0
                result.append({
                    "symbol": p["symbol"],
                    "side": p.get("side", "long"),
                    "size": contracts,
                    "entry_price": entry_price,
                    "mark_price": mark_price,
                    "unrealized_pnl": pnl,
                    "unrealized_pnl_pct": pnl_pct,
                    "leverage": int(p.get("leverage") or CFG.DEFAULT_LEVERAGE),
                })
            return result
        except Exception as e:
            logger.error(f"[PositionManager] get_positions failed: {e}")
            return []

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Return the open position for *symbol*, or None."""
        for p in self.get_positions():
            if p["symbol"] == symbol:
                return p
        return None

    # ------------------------------------------------------------------
    # Open position
    # ------------------------------------------------------------------
    def open_long(self, symbol: str,
                  leverage: Optional[int] = None) -> Optional[Dict]:
        """
        Open a long position on *symbol*.
        Returns the order dict on success, None on failure.
        In DRY_RUN mode, logs intent and returns a synthetic dict.
        """
        leverage = leverage or CFG.DEFAULT_LEVERAGE
        try:
            ticker = self._exchange.fetch_ticker(symbol)
            entry_price = float(ticker["last"])
            equity = self._get_equity()
            if equity <= 0:
                logger.warning(f"[PositionManager] zero equity, skipping {symbol}")
                return None

            sizing = self.risk.calc_position_size(
                symbol, equity, entry_price, leverage
            )
            qty = sizing["quantity"]
            stop_price = sizing["stop_price"]
            effective_leverage = sizing["leverage"]

            logger.info(
                f"[PositionManager] OPEN LONG {symbol} "
                f"qty={qty:.6f} entry≈{entry_price} "
                f"stop={stop_price:.4f} lev={effective_leverage}x "
                f"margin≈{sizing['margin']:.2f} USDT"
            )

            if CFG.DRY_RUN:
                logger.info(f"[DRY_RUN] would open long {symbol}")
                return {
                    "symbol": symbol,
                    "side": "long",
                    "quantity": qty,
                    "entry_price": entry_price,
                    "stop_price": stop_price,
                    "dry_run": True,
                }

            self._exchange.set_leverage(effective_leverage, symbol)
            order = self._exchange.create_market_buy_order(symbol, qty)
            logger.info(f"[PositionManager] order placed: {order.get('id')}")

            # use the actual filled quantity for the stop-loss to avoid mismatches
            filled_qty = float(order.get("filled") or order.get("amount") or qty)

            # place stop-loss order
            try:
                self._exchange.create_order(
                    symbol=symbol,
                    type="stop_market",
                    side="sell",
                    amount=filled_qty,
                    params={"stopPrice": stop_price, "closePosition": True},
                )
            except Exception as e:
                logger.warning(f"[PositionManager] stop-loss order failed: {e}")

            return order
        except Exception as e:
            logger.error(f"[PositionManager] open_long failed for {symbol}: {e}")
            return None

    # ------------------------------------------------------------------
    # Close / scale-out
    # ------------------------------------------------------------------
    def close_position(self, symbol: str,
                       reason: str = "close") -> Optional[Dict]:
        """
        Close the full open long for *symbol*.
        Returns the order dict on success, None on failure / no position.
        """
        position = self.get_position(symbol)
        if not position:
            logger.debug(f"[PositionManager] no open position for {symbol}")
            return None

        qty = position["size"]
        logger.info(
            f"[PositionManager] CLOSE {symbol} "
            f"qty={qty:.6f} reason={reason} "
            f"pnl={position['unrealized_pnl_pct']:.2%}"
        )

        if CFG.DRY_RUN:
            logger.info(f"[DRY_RUN] would close {symbol}")
            return {"symbol": symbol, "closed": True, "dry_run": True}

        try:
            order = self._exchange.create_market_sell_order(
                symbol, qty, params={"reduceOnly": True}
            )
            logger.info(f"[PositionManager] close order: {order.get('id')}")
            return order
        except Exception as e:
            logger.error(f"[PositionManager] close_position failed for {symbol}: {e}")
            return None

    def scale_out(self, symbol: str,
                  fraction: float = 0.5) -> Optional[Dict]:
        """
        Partially close *fraction* (0–1) of the open long for *symbol*.
        Returns the order dict on success, None on failure / no position.
        """
        position = self.get_position(symbol)
        if not position:
            return None

        min_pct, max_pct = CFG.SCALE_OUT_PCT_RANGE
        fraction = max(min_pct, min(max_pct, fraction))
        qty = position["size"] * fraction

        logger.info(
            f"[PositionManager] SCALE_OUT {symbol} "
            f"{fraction:.0%} qty={qty:.6f}"
        )

        if CFG.DRY_RUN:
            logger.info(f"[DRY_RUN] would scale_out {symbol} by {fraction:.0%}")
            return {"symbol": symbol, "scale_out": True, "fraction": fraction, "dry_run": True}

        try:
            order = self._exchange.create_market_sell_order(
                symbol, qty, params={"reduceOnly": True}
            )
            logger.info(f"[PositionManager] scale_out order: {order.get('id')}")
            return order
        except Exception as e:
            logger.error(f"[PositionManager] scale_out failed for {symbol}: {e}")
            return None
