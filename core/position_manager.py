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
            # 防止对同一标的重复开仓
            if self.get_position(symbol) is not None:
                logger.warning(f"[PositionManager] position already open for {symbol}; skipping")
                return None

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
    # Stop-order helpers
    # ------------------------------------------------------------------

    def _cancel_open_stop_orders(self, symbol: str) -> None:
        """撤销指定标的所有挂单中的 stop_market reduceOnly 卖单（止损挂单）。

        在全部平仓后调用，确保不会有残留止损单误触发后续仓位。
        """
        try:
            open_orders = self._exchange.fetch_open_orders(symbol)
            for order in open_orders:
                if (
                    order.get("type") in ("stop_market", "stop")
                    and order.get("reduceOnly")
                    and order.get("side", "").lower() == "sell"
                ):
                    self._exchange.cancel_order(order["id"], symbol)
                    logger.info(
                        f"[PositionManager] cancelled stop order {order['id']} for {symbol}"
                    )
        except Exception as exc:
            logger.error(
                f"[PositionManager] cancel open stop orders failed for {symbol}: {exc}"
            )

    def _adjust_stop_orders(self, symbol: str, remaining_qty: float) -> None:
        """撤销所有旧止损单并按剩余仓位数量重新挂一个止损单（减仓后调用）。

        先批量撤销所有匹配的止损挂单，再统一挂一个新止损单，
        避免多次迭代中重复下单导致超额止损覆盖。

        Parameters
        ----------
        symbol:
            合约标的。
        remaining_qty:
            减仓后剩余的多头数量。
        """
        try:
            open_orders = self._exchange.fetch_open_orders(symbol)
            stop_prices: list = []
            for order in open_orders:
                if (
                    order.get("type") in ("stop_market", "stop")
                    and order.get("reduceOnly")
                    and order.get("side", "").lower() == "sell"
                ):
                    stop_price = float(
                        order.get("stopPrice")
                        or order.get("info", {}).get("stopPrice", 0)
                        or 0
                    )
                    self._exchange.cancel_order(order["id"], symbol)
                    logger.info(
                        f"[PositionManager] cancelled stop order {order['id']} for {symbol}"
                    )
                    if stop_price > 0:
                        stop_prices.append(stop_price)

            if stop_prices:
                # 取最保守（最低）的止损价重新挂单
                stop_price = min(stop_prices)
                new_stop = self._exchange.create_order(
                    symbol=symbol,
                    type="stop_market",
                    side="sell",
                    amount=remaining_qty,
                    params={"stopPrice": stop_price, "reduceOnly": True},
                )
                logger.info(
                    f"[PositionManager] re-placed stop order for {symbol} "
                    f"qty={remaining_qty:.6f} @ {stop_price:.6f} "
                    f"orderId={new_stop.get('id')}"
                )
            else:
                logger.warning(
                    f"[PositionManager] no valid stop price found for {symbol}; "
                    "skipped re-placement – position has no stop protection!"
                )
        except Exception as exc:
            logger.error(
                f"[PositionManager] adjust stop orders failed for {symbol}: {exc}"
            )

    def move_stop_to_breakeven(self, symbol: str, entry_price: float, size: float) -> None:
        """将止损单移至保本价（开仓均价），保护浮盈不变为浮亏。

        遍历当前 reduceOnly stop_market 挂单：
        - 若已有止损挂单价格 ≥ entry_price，说明已在保本或更优位置，跳过不动。
        - 否则撤销所有旧止损，重新以 entry_price 为止损价挂单。

        Parameters
        ----------
        symbol:
            合约标的。
        entry_price:
            开仓均价，作为保本止损价格。
        size:
            当前持仓数量，用于重新挂单。
        """
        if CFG.DRY_RUN:
            logger.info(
                f"[PositionManager][DRY-RUN] would move stop to breakeven "
                f"for {symbol} @ {entry_price:.6f}"
            )
            return
        try:
            open_orders = self._exchange.fetch_open_orders(symbol)
            orders_to_cancel = []
            for order in open_orders:
                if (
                    order.get("type") in ("stop_market", "stop")
                    and order.get("reduceOnly")
                    and order.get("side", "").lower() == "sell"
                ):
                    stop_price = float(
                        order.get("stopPrice")
                        or order.get("info", {}).get("stopPrice", 0)
                        or 0
                    )
                    # 止损已在保本价或更优，无需移动
                    if stop_price >= entry_price:
                        continue
                    orders_to_cancel.append(order["id"])

            for oid in orders_to_cancel:
                self._exchange.cancel_order(oid, symbol)
                logger.info(f"[PositionManager] cancelled stop order {oid} for breakeven move")

            new_stop = self._exchange.create_order(
                symbol=symbol,
                type="stop_market",
                side="sell",
                amount=size,
                params={"stopPrice": entry_price, "reduceOnly": True},
            )
            logger.info(
                f"[PositionManager] BREAKEVEN stop set for {symbol} "
                f"@ {entry_price:.6f} orderId={new_stop.get('id')}"
            )
        except Exception as exc:
            logger.error(f"[PositionManager] move_stop_to_breakeven failed for {symbol}: {exc}")

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def close_long(self, symbol: str, quantity: Optional[float] = None) -> Optional[Dict]:
        """平多单（全部或部分）。

        平仓成功后自动撤销或调整对应的 stop_market reduceOnly 止损挂单，
        避免残留止损单误触发后续仓位。

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

            # 撤销或调整止损挂单，防止残留单误触发
            remaining_qty = pos["size"] - close_qty
            if remaining_qty <= 0:
                if remaining_qty < 0:
                    logger.warning(
                        f"[PositionManager] close qty ({close_qty:.6f}) exceeds recorded "
                        f"position size ({pos['size']:.6f}) for {symbol}; "
                        f"cancelling all stop orders"
                    )
                self._cancel_open_stop_orders(symbol)
            else:
                self._adjust_stop_orders(symbol, remaining_qty)

            return order

        except Exception as exc:
            logger.error(f"[PositionManager] close_long failed for {symbol}: {exc}")
            return None

    def close_all(self) -> None:
        """平掉所有持仓（紧急清仓）。"""
        for pos in self.get_positions():
            self.close_long(pos["symbol"])
