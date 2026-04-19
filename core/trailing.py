"""Cron 止盈循环（Trailing Evaluator）——系统核心灵魂。

每隔 ``CFG.EVALUATION_INTERVAL_MINUTES`` 分钟对所有持仓执行一次多维评估，
根据 AI 决策自动止盈、减仓或平仓。

评估逻辑（优先级从高到低）：
1. 黑天鹅检测 → 立即全平
2. AI 模型返回 ``close`` → 全平
3. AI 模型返回 ``scale_out`` → 按比例减仓
4. AI 模型返回 ``add`` 且置信度 > 阈值 → 加仓（受仓位上限约束）
5. 其余情况 → 持有，记录日志

对于模糊信号（``p_up`` 接近阈值），会以较短周期
``CFG.AMBIGUOUS_INTERVAL_MINUTES`` 重新评估。
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Set

from loguru import logger

from config import CFG
from core.ai_evaluator import AIEvaluator
from core.data_fetcher import DataFetcher
from core.position_manager import PositionManager
from core.risk_manager import RiskManager
from utils.notifier import notify

#: p_up 接近 0.5 ± 此范围视为模糊信号
_AMBIGUOUS_MARGIN = 0.08


class TrailingEvaluator:
    """持仓 Cron 评估器。

    Parameters
    ----------
    fetcher:
        行情数据获取器。
    position:
        仓位管理器。
    evaluator:
        AI 评估器。
    risk:
        风控管理器。
    """

    def __init__(
        self,
        fetcher: DataFetcher,
        position: PositionManager,
        evaluator: AIEvaluator,
        risk: RiskManager,
    ) -> None:
        self._fetcher = fetcher
        self._position = position
        self._evaluator = evaluator
        self._risk = risk
        # symbol -> unix timestamp of next allowed re-evaluation
        self._next_eval: Dict[str, float] = {}
        # symbols currently in "ambiguous" fast-poll mode
        self._ambiguous: Set[str] = set()

    # ------------------------------------------------------------------
    # Public interface (called by main.py scheduler)
    # ------------------------------------------------------------------

    def evaluate_all(self) -> None:
        """对所有当前持仓执行一轮评估。"""
        positions = self._position.get_positions()
        if not positions:
            logger.debug("[Trailing] no open positions")
            return

        logger.info(f"[Trailing] evaluating {len(positions)} positions")
        for pos in positions:
            symbol = pos["symbol"]
            try:
                self._evaluate_one(symbol, pos)
            except Exception as exc:
                logger.error(f"[Trailing] error evaluating {symbol}: {exc}")

    # ------------------------------------------------------------------
    # Per-symbol evaluation
    # ------------------------------------------------------------------

    def _evaluate_one(self, symbol: str, position: Dict) -> None:
        now = time.time()
        next_due = self._next_eval.get(symbol, 0)
        if now < next_due:
            logger.debug(f"[Trailing] {symbol} skipped (next eval in {next_due - now:.0f}s)")
            return

        # ---- black swan check (highest priority) ----------------------
        if self._risk.check_black_swan(symbol):
            logger.warning(f"[Trailing] BLACK SWAN {symbol} — closing immediately")
            self._close(symbol, "black swan detected")
            return

        # ---- fetch latest snapshot ------------------------------------
        try:
            snapshot = self._fetcher.snapshot(symbol)
        except Exception as exc:
            logger.warning(f"[Trailing] snapshot failed for {symbol}: {exc}")
            return

        # ---- AI hold evaluation ---------------------------------------
        decision = self._evaluator.evaluate_hold(symbol, snapshot, position)
        action = decision.get("action", "hold")
        p_up = decision.get("p_up", 0.5)
        confidence = decision.get("confidence", 0.0)
        reason = decision.get("key_reason", "")
        scale_pct = decision.get("scale_out_pct", CFG.SCALE_OUT_PCT_RANGE[0])

        logger.info(
            f"[Trailing] {symbol} action={action} p_up={p_up:.2f} "
            f"conf={confidence:.2f} | {reason}"
        )

        # ---- execute decision -----------------------------------------
        if action == "close":
            self._close(symbol, reason)

        elif action == "scale_out":
            self._scale_out(symbol, scale_pct, reason)
            # schedule next eval sooner if still in ambiguous zone
            interval_sec = self._pick_interval(p_up)
            self._next_eval[symbol] = now + interval_sec

        elif action == "add":
            if confidence >= CFG.ENTRY_CONFIDENCE_THRESHOLD:
                self._add(symbol, reason)
            else:
                logger.info(
                    f"[Trailing] {symbol} add signal but low confidence={confidence:.2f}, skip"
                )
            interval_sec = self._pick_interval(p_up)
            self._next_eval[symbol] = now + interval_sec

        else:  # hold
            interval_sec = self._pick_interval(p_up)
            self._next_eval[symbol] = now + interval_sec
            if symbol in self._ambiguous and abs(p_up - 0.5) > _AMBIGUOUS_MARGIN:
                logger.info(f"[Trailing] {symbol} exiting ambiguous mode")
                self._ambiguous.discard(symbol)

    # ------------------------------------------------------------------
    # Action helpers
    # ------------------------------------------------------------------

    def _close(self, symbol: str, reason: str) -> None:
        result = self._position.close_position(symbol)
        if result is not None:
            notify(f"CLOSE {symbol} | {reason}")
            logger.info(f"[Trailing] closed {symbol}: {reason}")
        self._next_eval.pop(symbol, None)
        self._ambiguous.discard(symbol)

    def _scale_out(self, symbol: str, pct: float, reason: str) -> None:
        pct = max(CFG.SCALE_OUT_PCT_RANGE[0], min(CFG.SCALE_OUT_PCT_RANGE[1], pct))
        result = self._position.scale_out(symbol, pct)
        if result is not None:
            notify(f"SCALE_OUT {symbol} {pct:.0%} | {reason}")
            logger.info(f"[Trailing] scaled out {symbol} {pct:.0%}: {reason}")

    def _add(self, symbol: str, reason: str) -> None:
        result = self._position.open_long(symbol)
        if result is not None:
            notify(f"ADD {symbol} | {reason}")
            logger.info(f"[Trailing] added to {symbol}: {reason}")

    # ------------------------------------------------------------------
    # Interval selection
    # ------------------------------------------------------------------

    def _pick_interval(self, p_up: float) -> float:
        """根据 p_up 决定下次评估间隔（秒）。

        若信号模糊（接近 0.5），使用更短的 AMBIGUOUS 间隔。
        """
        if abs(p_up - 0.5) <= _AMBIGUOUS_MARGIN:
            return CFG.AMBIGUOUS_INTERVAL_MINUTES * 60.0
        return CFG.EVALUATION_INTERVAL_MINUTES * 60.0

    # ------------------------------------------------------------------
    # Convenience: evaluate specific symbols (for external callers)
    # ------------------------------------------------------------------

    def evaluate_symbols(self, symbols: List[str]) -> None:
        """只评估给定 symbol 子集（不依赖当前持仓列表）。"""
        positions = self._position.get_positions()
        pos_map = {p["symbol"]: p for p in positions}
        for symbol in symbols:
            pos = pos_map.get(symbol)
            if pos is None:
                logger.debug(f"[Trailing] {symbol} not in positions, skip")
                continue
            try:
                self._evaluate_one(symbol, pos)
            except Exception as exc:
                logger.error(f"[Trailing] error evaluating {symbol}: {exc}")
