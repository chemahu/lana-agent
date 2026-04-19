"""追踪持仓评估器：对每个持仓周期性调用 AI 判断是否继续持有、减仓或平仓。"""
from typing import Optional

from loguru import logger

from config import CFG
from core.ai_evaluator import AIEvaluator
from core.data_fetcher import DataFetcher
from core.position_manager import PositionManager
from core.risk_manager import RiskManager
from utils.notifier import notify


class TrailingEvaluator:
    """对所有当前持仓逐一抓取快照、调用 AI 评估，并执行相应操作。

    操作逻辑
    --------
    * ``close``      → 立即全部平仓
    * ``scale_out``  → 按 scale_out_pct 部分减仓
    * ``hold``       → 继续持有（什么都不做）

    同时内置黑天鹅检测（闪跌超阈值时强制平仓）和高盈利止盈逻辑。
    """

    def __init__(
        self,
        fetcher: DataFetcher,
        position: PositionManager,
        evaluator: AIEvaluator,
        risk: RiskManager,
    ) -> None:
        self.fetcher = fetcher
        self.position = position
        self.evaluator = evaluator
        self.risk = risk

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_roi(self, entry_price: float, current_price: float) -> float:
        """计算简单收益率（未考虑杠杆）。"""
        if entry_price <= 0:
            return 0.0
        return (current_price - entry_price) / entry_price

    def _reconcile_position(self, symbol: str, stale_pos: dict) -> Optional[dict]:
        """执行平仓/减仓前的二次校验：重新拉取最新持仓，确保操作基于最新状态。

        Parameters
        ----------
        symbol:
            合约标的。
        stale_pos:
            本轮开始时获取的"快照"持仓信息（可能已过期）。

        Returns
        -------
        dict | None
            返回最新持仓字典；若持仓已不存在则返回 None（调用方应跳过该操作）。
        """
        fresh = self.position.get_position(symbol)
        if fresh is None:
            logger.warning(
                f"[Trailing] reconcile: {symbol} position no longer exists on exchange; "
                "skipping action to avoid phantom close"
            )
            return None

        stale_size = stale_pos.get("size", 0)
        fresh_size = fresh.get("size", 0)
        if stale_size > 0:
            size_diff_pct = abs(fresh_size - stale_size) / stale_size
            if size_diff_pct > 0.05:
                logger.warning(
                    f"[Trailing] reconcile: {symbol} stale_size={stale_size:.6f} "
                    f"fresh_size={fresh_size:.6f} (diff={size_diff_pct:.1%}) – "
                    "using fresh data"
                )
        return fresh

    def _handle_black_swan(
        self,
        symbol: str,
        entry_price: float = 0,
        current_price: float = 0,
        pos: Optional[dict] = None,
    ) -> bool:
        """检测黑天鹅并执行紧急平仓。返回 True 表示已触发。

        Parameters
        ----------
        pos:
            可选——调用方已有的最新持仓字典，用于在判定触发后二次确认仓位存在，
            避免对已平仓标的发出幻影平仓指令。
        """
        if self.risk.check_black_swan(symbol, entry_price, current_price):
            # 二次校验：确认持仓确实存在，避免幻影平仓
            live = self._reconcile_position(symbol, pos or {})
            if live is None:
                logger.warning(
                    f"[Trailing] BLACK SWAN on {symbol} but position already gone; skipping"
                )
                return True  # 已无仓位，视为已处理
            logger.warning(f"[Trailing] BLACK SWAN detected on {symbol}, closing position")
            self.position.close_long(symbol)
            notify(f"⚠️ BLACK SWAN: force-closed {symbol}")
            return True
        return False

    def _handle_high_roi(self, symbol: str, pos: dict, current_price: float) -> bool:
        """若 ROI 超过高盈利阈值，先进行持仓二次校验再执行部分减仓（50%）。返回 True 表示已操作。"""
        roi = self._get_roi(pos.get("entry_price", 0), current_price)
        if roi >= CFG.HIGH_ROI_THRESHOLD:
            # 二次校验：确认持仓最新状态，防止基于过期快照下单
            fresh_pos = self._reconcile_position(symbol, pos)
            if fresh_pos is None:
                return True  # 仓位已消失，视为已处理
            scale_qty = fresh_pos["size"] * 0.5
            logger.info(
                f"[Trailing] high ROI {roi:.2%} on {symbol}, scaling out 50% "
                f"({scale_qty:.6f})"
            )
            self.position.close_long(symbol, quantity=scale_qty)
            notify(f"💰 HIGH ROI {roi:.2%}: scaled out 50% of {symbol}")
            return True
        return False

    def _handle_breakeven(self, symbol: str, pos: dict, current_price: float) -> None:
        """若浮盈达到保本触发阈值，将止损单移至开仓均价，保护浮盈不变为浮亏。"""
        entry_price = pos.get("entry_price", 0)
        if entry_price <= 0:
            return
        roi = self._get_roi(entry_price, current_price)
        if roi >= CFG.BREAKEVEN_ROI_TRIGGER:
            logger.info(
                f"[Trailing] breakeven triggered for {symbol} "
                f"(ROI={roi:.2%}), moving stop to entry {entry_price:.6f}"
            )
            self.position.move_stop_to_breakeven(symbol, entry_price, pos["size"])
            notify(
                f"🔒 BREAKEVEN: moved stop to entry {entry_price:.6f} "
                f"for {symbol} (ROI={roi:.2%})"
            )

    # ------------------------------------------------------------------
    # Per-position logic
    # ------------------------------------------------------------------

    def _evaluate_one(self, pos: dict) -> None:
        symbol = pos["symbol"]
        entry_price = pos.get("entry_price", 0)
        try:
            # 1. 抓取快照（顺便拿到现价，给黑天鹅检测复用，省掉每轮的 1m K 线请求）
            snapshot = self.fetcher.snapshot(symbol)
            current_price = snapshot.get("price", {}).get("current_price", 0)

            # 2. 黑天鹅检测（仍是第一道平仓判断；优先用 current_price 直比 entry_price，
            #    snapshot 失败拿不到现价时回退到 K 线兜底路径；内含持仓二次校验）
            if self._handle_black_swan(symbol, entry_price, current_price, pos):
                return

            if not current_price:
                return

            # 二次校验：在执行后续高频操作前，重新拉取最新持仓，确保所有动作
            # 基于交易所实际状态（而非本轮开始时的快照）
            live_pos = self._reconcile_position(symbol, pos)
            if live_pos is None:
                return  # 持仓已消失（止损/外部平仓等），跳过本次评估

            # 3. 高盈利止盈（使用最新持仓数据）
            if self._handle_high_roi(symbol, live_pos, current_price):
                return

            # 4. 保本止损（浮盈达阈值则将止损移至开仓均价，保护浮盈不变浮亏）
            self._handle_breakeven(symbol, live_pos, current_price)

            # 5. AI 评估
            decision = self.evaluator.evaluate_hold(symbol, snapshot)
            action = decision.get("action", "hold")
            p_up = decision.get("p_up", 0.5)
            reason = decision.get("key_reason", "")

            if action == "close":
                logger.info(f"[Trailing] AI says CLOSE {symbol}: {reason}")
                self.position.close_long(symbol)
                notify(f"📉 CLOSE {symbol} (AI): {reason}")

            elif action == "scale_out":
                scale_pct = float(decision.get("scale_out_pct", 0))
                # 约束在合理范围内
                scale_pct = max(CFG.SCALE_OUT_PCT_RANGE[0],
                                min(scale_pct, CFG.SCALE_OUT_PCT_RANGE[1]))
                # 以最新持仓数量计算减仓绝对数量，避免基于过期 pos 超额平仓
                scale_qty = live_pos.get("size", 0) * scale_pct
                logger.info(
                    f"[Trailing] AI says SCALE_OUT {symbol} {scale_pct:.0%}: {reason}"
                )
                self.position.close_long(symbol, quantity=scale_qty)
                notify(f"📊 SCALE OUT {scale_pct:.0%} of {symbol} (AI): {reason}")

            else:  # hold
                logger.debug(
                    f"[Trailing] HOLD {symbol} p_up={p_up:.2f}: {reason}"
                )

        except Exception as exc:
            logger.error(f"[Trailing] _evaluate_one crashed for {symbol}: {exc}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_all(self) -> None:
        """遍历所有当前持仓并执行追踪逻辑。"""
        positions = self.position.get_positions()
        if not positions:
            logger.debug("[Trailing] no open positions to evaluate")
            return
        logger.info(f"[Trailing] evaluating {len(positions)} position(s)")
        for pos in positions:
            self._evaluate_one(pos)
