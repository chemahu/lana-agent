"""Trailing 止盈评估器（系统灵魂）。

每隔 ``CFG.EVALUATION_INTERVAL_MINUTES`` 分钟对所有持仓做一轮评估：

1. 先检查黑天鹅（闪崩） → 立即全平。
2. 调用 ``AIEvaluator.evaluate_hold`` 获取处置建议。
3. 若主评估置信度低（模糊区间），触发 ``evaluate_ambiguous`` 二次确认。
4. 根据建议执行：``hold`` 不动 / ``scale_out`` 分批减仓 / ``close_all`` 全平。
5. 全程日志 + Telegram 通知。
"""
from __future__ import annotations

from typing import Dict, List

from loguru import logger

from config import CFG
from core.ai_evaluator import AIEvaluator
from core.data_fetcher import DataFetcher
from core.position_manager import PositionManager
from core.risk_manager import RiskManager
from utils.notifier import notify

# 置信度低于此值时触发二次确认
_AMBIGUOUS_CONFIDENCE = 0.55


class TrailingEvaluator:
    """周期性评估所有持仓，执行动态止盈 / 止损策略。

    Parameters
    ----------
    fetcher:
        行情 / 仓位数据源。
    position:
        仓位管理器（执行开 / 平仓）。
    evaluator:
        AI 评估器（给出持仓建议）。
    risk:
        风险管理器（黑天鹅检测）。
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _handle_position(self, pos: Dict) -> None:
        """对单个持仓执行一次完整的评估 + 处置流程。"""
        symbol: str = pos["symbol"]
        unrealized_pnl: float = float(pos.get("unrealizedPnl") or 0)
        entry_price: float = float(pos.get("entryPrice") or 0)

        logger.info(
            f"[Trailing] evaluating {symbol} "
            f"pnl={unrealized_pnl:+.2f} USDT entry={entry_price}"
        )

        # 1. 黑天鹅检测
        if self._risk.check_black_swan(symbol):
            logger.warning(f"[Trailing] BLACK SWAN detected on {symbol} → CLOSE ALL")
            self._position.close_position(symbol)
            notify(f"⚠️ BLACK SWAN {symbol}: 紧急全平")
            return

        # 2. 获取市场快照
        try:
            snapshot = self._fetcher.snapshot(symbol)
        except Exception as exc:
            logger.error(f"[Trailing] snapshot failed for {symbol}: {exc}")
            return

        # 3. 主评估
        hold_result = self._evaluator.evaluate_hold(symbol, snapshot)
        action: str = hold_result.get("action", "hold")
        confidence: float = float(hold_result.get("confidence", 0.0))
        key_reason: str = hold_result.get("key_reason", "")
        scale_out_pct: float = float(hold_result.get("scale_out_pct") or 0.0)

        logger.info(
            f"[Trailing] {symbol} primary → action={action} "
            f"confidence={confidence:.2f} reason={key_reason!r}"
        )

        # 4. 模糊区间二次确认
        if confidence < _AMBIGUOUS_CONFIDENCE:
            logger.info(
                f"[Trailing] {symbol} confidence={confidence:.2f} < "
                f"{_AMBIGUOUS_CONFIDENCE} → ambiguous cross-check"
            )
            ambiguous_result = self._evaluator.evaluate_ambiguous(
                symbol, snapshot, primary_result=hold_result
            )
            # 以二次结果覆盖（evaluate_ambiguous 已做 consensus 逻辑）
            action = ambiguous_result.get("action", action)
            confidence = float(ambiguous_result.get("confidence", confidence))
            key_reason = ambiguous_result.get("key_reason", key_reason)
            scale_out_pct = float(ambiguous_result.get("scale_out_pct") or scale_out_pct)
            logger.info(
                f"[Trailing] {symbol} after cross-check → action={action} "
                f"confidence={confidence:.2f}"
            )

        # 5. 执行建议
        if action == "close_all":
            logger.info(f"[Trailing] {symbol} → CLOSE ALL (reason={key_reason!r})")
            self._position.close_position(symbol)
            notify(
                f"🔴 CLOSE ALL {symbol}\n"
                f"reason: {key_reason}\n"
                f"pnl: {unrealized_pnl:+.2f} USDT"
            )

        elif action == "scale_out":
            # clamp to configured range
            min_pct, max_pct = CFG.SCALE_OUT_PCT_RANGE
            pct = max(min_pct, min(max_pct, scale_out_pct or min_pct))
            logger.info(
                f"[Trailing] {symbol} → SCALE_OUT {pct:.0%} "
                f"(reason={key_reason!r})"
            )
            self._position.scale_out(symbol, pct)
            notify(
                f"🟡 SCALE OUT {symbol} {pct:.0%}\n"
                f"reason: {key_reason}\n"
                f"pnl: {unrealized_pnl:+.2f} USDT"
            )

        else:  # hold
            logger.info(
                f"[Trailing] {symbol} → HOLD "
                f"(reason={key_reason!r} confidence={confidence:.2f})"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_all(self) -> None:
        """对所有当前持仓执行一轮 trailing 评估。

        由 ``main.py`` 中的 ``schedule`` 任务周期调用。
        出现异常时记录日志但不中断整轮循环。
        """
        positions: List[Dict] = self._position.get_positions()
        if not positions:
            logger.debug("[Trailing] no open positions")
            return

        logger.info(f"[Trailing] evaluating {len(positions)} positions")
        for pos in positions:
            try:
                self._handle_position(pos)
            except Exception as exc:
                symbol = pos.get("symbol", "?")
                logger.error(
                    f"[Trailing] unhandled error for {symbol}: {exc}",
                    exc_info=True,
                )
