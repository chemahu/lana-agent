"""Cron 止盈 / 持仓再评估循环（系统灵魂）"""
from typing import Optional
from loguru import logger
from config import CFG
from core.data_fetcher import DataFetcher
from core.position_manager import PositionManager
from core.ai_evaluator import AIEvaluator
from core.risk_manager import RiskManager
from utils.notifier import notify


class TrailingEvaluator:
    """每隔 EVALUATION_INTERVAL_MINUTES 分钟对所有持仓做再评估。

    决策树
    ------
    1. 黑天鹅检测 → 触发则立即全平
    2. AI 评估 (evaluate_hold):
       - CLOSE_ALL  → 全平
       - SCALE_OUT  → 按建议比例减仓
       - HOLD       → 检查 p_up 是否低于持仓维持阈值，是则减仓
    3. 低置信度 / 模糊区间 → 5 分钟后再看（由 schedule 保证）
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
    # Public entry point
    # ------------------------------------------------------------------

    def evaluate_all(self) -> None:
        """对当前所有持仓依次执行再评估。"""
        positions = self.position.get_positions()
        if not positions:
            logger.debug("[Trailing] no open positions")
            return

        logger.info(f"[Trailing] evaluating {len(positions)} position(s)")
        for pos in positions:
            try:
                self._evaluate_one(pos)
            except Exception as exc:
                logger.error(
                    f"[Trailing] error evaluating {pos.get('symbol')}: {exc}"
                )

    # ------------------------------------------------------------------
    # Single-position evaluation
    # ------------------------------------------------------------------

    def _evaluate_one(self, pos: dict) -> None:
        symbol = pos["symbol"]
        roi = pos.get("roi", 0.0)
        logger.info(
            f"[Trailing] {symbol}  roi={roi:.2%}  "
            f"entry={pos.get('entry_price')}"
        )

        # --- 1. Black swan check ---
        if self.risk.check_black_swan(symbol):
            logger.warning(f"[Trailing] BLACK SWAN → close all {symbol}")
            result = self.position.close_all(symbol)
            if result:
                notify(
                    f"🚨 BLACK SWAN close {symbol} roi={roi:.2%}"
                )
            return

        # --- 2. AI hold evaluation ---
        try:
            snapshot = self.fetcher.snapshot(symbol)
        except Exception as exc:
            logger.warning(f"[Trailing] snapshot failed for {symbol}: {exc}")
            return

        decision = self.evaluator.evaluate_hold(symbol, snapshot, roi)
        action = decision.get("action", "HOLD")
        reason = decision.get("key_reason", "")
        scale_pct = float(decision.get("scale_out_pct", 0.5))

        if action == "CLOSE_ALL":
            result = self.position.close_all(symbol)
            if result:
                notify(
                    f"✅ CLOSE ALL {symbol} roi={roi:.2%} reason={reason}"
                )

        elif action == "SCALE_OUT":
            # Clamp scale_out_pct to configured range
            lo, hi = CFG.SCALE_OUT_PCT_RANGE
            clamped = min(max(scale_pct, lo), hi)
            result = self.position.scale_out(symbol, pct=clamped)
            if result:
                notify(
                    f"📉 SCALE OUT {symbol} {clamped:.0%} roi={roi:.2%} reason={reason}"
                )

        elif action == "HOLD":
            p_up = decision.get("p_up", 0.5)
            # If p_up dropped significantly below threshold, do a small scale-out
            if p_up < (CFG.ENTRY_P_UP_THRESHOLD - CFG.HOLD_P_UP_MARGIN):
                result = self.position.scale_out(symbol, pct=0.25)
                if result:
                    notify(
                        f"⚠️ CAUTIOUS SCALE OUT {symbol} 25%  "
                        f"p_up={p_up:.2f} roi={roi:.2%}"
                    )
            else:
                logger.info(f"[Trailing] HOLD {symbol}  p_up={p_up:.2f}")
