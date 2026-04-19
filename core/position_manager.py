"""下单 / 止损 / 分批止盈 / 全平仓位管理"""
from typing import Dict, List, Optional, Any
from loguru import logger
from config import CFG
from core.data_fetcher import DataFetcher
from core.risk_manager import RiskManager


class PositionManager:
    """通过 ccxt 管理 Binance USDM 合约仓位。"""

    def __init__(self, fetcher: DataFetcher, risk: RiskManager) -> None:
        self.fetcher = fetcher
        self.risk = risk
        self._exchange = fetcher.exchange

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_account_equity(self) -> float:
        try:
            balance = self._exchange.fetch_balance()
            equity = (
                balance.get("info", {}).get("totalWalletBalance")
                or balance.get("USDT", {}).get("total")
                or 0.0
            )
            return float(equity)
        except Exception as exc:
            logger.error(f"[PositionManager] fetch balance failed: {exc}")
            return 0.0

    def _set_leverage(self, symbol: str, leverage: int) -> None:
        try:
            contract_symbol = symbol.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT")
            self._exchange.set_leverage(leverage, contract_symbol)
        except Exception as exc:
            logger.warning(f"[PositionManager] set leverage failed for {symbol}: {exc}")

    # ------------------------------------------------------------------
    # Dry-run mock
    # ------------------------------------------------------------------

    @staticmethod
    def _dry_run_result(symbol: str, action: str, **kwargs: Any) -> Dict:
        result = {"symbol": symbol, "action": action, "dry_run": True, **kwargs}
        logger.info(f"[PositionManager] DRY-RUN {action} {symbol}: {result}")
        return result

    # ------------------------------------------------------------------
    # Position queries
    # ------------------------------------------------------------------

    def get_positions(self) -> List[Dict]:
        """返回当前所有非零仓位的标准化列表。"""
        try:
            raw_positions = self._exchange.fetch_positions()
            positions = []
            for pos in raw_positions:
                amt = float(pos.get("contracts") or pos.get("amount") or 0)
                if amt == 0:
                    continue
                entry = float(pos.get("entryPrice") or 0)
                mark = float(pos.get("markPrice") or pos.get("info", {}).get("markPrice") or entry)
                roi = (mark / entry - 1) if entry > 0 else 0.0
                positions.append({
                    "symbol": pos.get("symbol", ""),
                    "size": amt,
                    "entry_price": entry,
                    "mark_price": mark,
                    "roi": roi,
                    "side": pos.get("side", "long"),
                    "unrealized_pnl": float(pos.get("unrealizedPnl") or 0),
                    "leverage": int(pos.get("leverage") or CFG.DEFAULT_LEVERAGE),
                })
            return positions
        except Exception as exc:
            logger.error(f"[PositionManager] get_positions failed: {exc}")
            return []

    def get_position(self, symbol: str) -> Optional[Dict]:
        """返回指定 symbol 的仓位，不存在返回 None。"""
        for pos in self.get_positions():
            if pos["symbol"] == symbol:
                return pos
        return None

    # ------------------------------------------------------------------
    # Open long
    # ------------------------------------------------------------------

    def open_long(self, symbol: str) -> Optional[Dict]:
        """开多仓。返回订单信息字典，失败返回 None。"""
        try:
            equity = self._get_account_equity()
            if equity <= 0:
                logger.warning("[PositionManager] zero equity, skip open_long")
                return None

            ticker = self._exchange.fetch_ticker(symbol)
            entry_price = float(ticker.get("last") or ticker.get("close") or 0)
            if entry_price <= 0:
                logger.warning(f"[PositionManager] invalid price for {symbol}")
                return None

            is_new = self.fetcher.is_new_listing(symbol, CFG.NEW_COIN_DAYS)
            leverage = CFG.NEW_COIN_LEVERAGE if is_new else CFG.DEFAULT_LEVERAGE
            sizing = self.risk.calc_position_size(
                symbol, equity, entry_price, leverage
            )
            quantity = sizing["quantity"]
            if quantity <= 0:
                logger.warning(f"[PositionManager] zero quantity for {symbol}")
                return None

            if CFG.DRY_RUN:
                return self._dry_run_result(
                    symbol, "open_long",
                    quantity=quantity,
                    entry_price=entry_price,
                    sizing=sizing,
                )

            self._set_leverage(symbol, sizing["leverage"])

            # Market order
            order = self._exchange.create_market_buy_order(symbol, quantity)
            logger.info(
                f"[PositionManager] OPENED LONG {symbol} "
                f"qty={quantity:.6f} entry≈{entry_price} leverage={sizing['leverage']}"
            )

            # Place hard stop-loss
            stop_price = sizing["stop_price"]
            try:
                self._exchange.create_order(
                    symbol,
                    "stop_market",
                    "sell",
                    quantity,
                    None,
                    {"stopPrice": stop_price, "closePosition": True},
                )
                logger.info(
                    f"[PositionManager] stop-loss placed at {stop_price:.6f} for {symbol}"
                )
            except Exception as exc:
                logger.warning(
                    f"[PositionManager] stop-loss placement failed for {symbol}: {exc}"
                )

            return {"symbol": symbol, "order": order, "sizing": sizing}

        except Exception as exc:
            logger.error(f"[PositionManager] open_long failed for {symbol}: {exc}")
            return None

    # ------------------------------------------------------------------
    # Scale out (partial take-profit)
    # ------------------------------------------------------------------

    def scale_out(self, symbol: str, pct: float = 0.5) -> Optional[Dict]:
        """减仓指定比例（pct ∈ (0, 1]）。"""
        pct = min(max(pct, 0.01), 1.0)
        pos = self.get_position(symbol)
        if pos is None:
            logger.warning(f"[PositionManager] no position to scale_out: {symbol}")
            return None
        reduce_qty = pos["size"] * pct

        if CFG.DRY_RUN:
            return self._dry_run_result(symbol, "scale_out", pct=pct, qty=reduce_qty)

        try:
            order = self._exchange.create_market_sell_order(
                symbol, reduce_qty, {"reduceOnly": True}
            )
            logger.info(
                f"[PositionManager] SCALE_OUT {symbol} "
                f"pct={pct:.0%} qty={reduce_qty:.6f}"
            )
            return {"symbol": symbol, "order": order, "pct": pct}
        except Exception as exc:
            logger.error(f"[PositionManager] scale_out failed for {symbol}: {exc}")
            return None

    # ------------------------------------------------------------------
    # Close all
    # ------------------------------------------------------------------

    def close_all(self, symbol: str) -> Optional[Dict]:
        """全平指定 symbol 仓位。"""
        pos = self.get_position(symbol)
        if pos is None:
            logger.warning(f"[PositionManager] no position to close: {symbol}")
            return None

        if CFG.DRY_RUN:
            return self._dry_run_result(symbol, "close_all", size=pos["size"])

        try:
            order = self._exchange.create_market_sell_order(
                symbol, pos["size"], {"reduceOnly": True}
            )
            logger.info(
                f"[PositionManager] CLOSED ALL {symbol} "
                f"size={pos['size']:.6f} roi={pos['roi']:.2%}"
            )
            return {"symbol": symbol, "order": order}
        except Exception as exc:
            logger.error(f"[PositionManager] close_all failed for {symbol}: {exc}")
            return None
