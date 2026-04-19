"""双模型 AI 评估器（Claude + GPT）。

对每个候选标的构造结构化 prompt，并行请求两个模型，
以多数票机制决定是否入场/持仓/出场。

主要接口
--------
- ``evaluate_entry(symbol, snapshot)``  → 入场决策
- ``evaluate_hold(symbol, snapshot, position)``  → 持仓决策（加仓/减仓/持有/平仓）
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from loguru import logger

from config import CFG

# --------------------------------------------------------------------------
# Optional SDK imports — gracefully degrade when keys are absent
# --------------------------------------------------------------------------

try:
    import anthropic as _anthropic_sdk  # type: ignore
except ImportError:
    _anthropic_sdk = None  # type: ignore

try:
    import openai as _openai_sdk  # type: ignore
except ImportError:
    _openai_sdk = None  # type: ignore

# --------------------------------------------------------------------------
# Prompt templates
# --------------------------------------------------------------------------

_ENTRY_SYSTEM = (
    "You are Lana, an expert quantitative crypto futures trader. "
    "Analyse the provided market snapshot and return ONLY valid JSON with keys: "
    '"should_enter" (bool), "p_up" (float 0-1), "confidence" (float 0-1), '
    '"key_reason" (str ≤ 80 chars). No markdown, no extra text.'
)

_HOLD_SYSTEM = (
    "You are Lana, managing an open futures position. "
    "Analyse the snapshot and current position, then return ONLY valid JSON with keys: "
    '"action" (one of: hold/scale_out/close/add), '
    '"scale_out_pct" (float 0-1, relevant only for scale_out action), '
    '"p_up" (float 0-1), "confidence" (float 0-1), '
    '"key_reason" (str ≤ 80 chars). No markdown, no extra text.'
)

_CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
_GPT_MODEL = "gpt-4o-mini"


# --------------------------------------------------------------------------
# AIEvaluator
# --------------------------------------------------------------------------


class AIEvaluator:
    """双模型评估器。若 API key 缺失则自动降级为规则基线。"""

    def __init__(self) -> None:
        self._claude = self._init_claude()
        self._gpt = self._init_gpt()

    # ------------------------------------------------------------------
    # SDK initialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _init_claude() -> Optional[Any]:
        if _anthropic_sdk is None or not CFG.ANTHROPIC_API_KEY:
            return None
        try:
            return _anthropic_sdk.Anthropic(api_key=CFG.ANTHROPIC_API_KEY)
        except Exception as exc:
            logger.warning(f"[AIEvaluator] Claude init failed: {exc}")
            return None

    @staticmethod
    def _init_gpt() -> Optional[Any]:
        if _openai_sdk is None or not CFG.OPENAI_API_KEY:
            return None
        try:
            return _openai_sdk.OpenAI(api_key=CFG.OPENAI_API_KEY)
        except Exception as exc:
            logger.warning(f"[AIEvaluator] GPT init failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # LLM call helpers
    # ------------------------------------------------------------------

    def _call_claude(self, system: str, user_msg: str) -> Optional[Dict]:
        if self._claude is None:
            return None
        try:
            resp = self._claude.messages.create(
                model=_CLAUDE_MODEL,
                max_tokens=256,
                system=system,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = resp.content[0].text.strip()
            return json.loads(raw)
        except Exception as exc:
            logger.warning(f"[AIEvaluator] Claude call failed: {exc}")
            return None

    def _call_gpt(self, system: str, user_msg: str) -> Optional[Dict]:
        if self._gpt is None:
            return None
        try:
            resp = self._gpt.chat.completions.create(
                model=_GPT_MODEL,
                max_tokens=256,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
            )
            raw = resp.choices[0].message.content or "{}"
            return json.loads(raw)
        except Exception as exc:
            logger.warning(f"[AIEvaluator] GPT call failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Merge & baseline
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_entry(results: list) -> Dict:
        """Merge multiple model outputs into one entry decision."""
        valid = [r for r in results if r and "p_up" in r]
        if not valid:
            return {
                "should_enter": False,
                "p_up": 0.5,
                "confidence": 0.0,
                "key_reason": "no AI response, defaulting to skip",
            }
        avg_p_up = sum(r["p_up"] for r in valid) / len(valid)
        avg_conf = sum(r.get("confidence", 0.5) for r in valid) / len(valid)
        reasons = [r.get("key_reason", "") for r in valid if r.get("key_reason")]
        key_reason = reasons[0] if reasons else "models agree"
        should_enter = (
            avg_p_up >= CFG.ENTRY_P_UP_THRESHOLD
            and avg_conf >= CFG.ENTRY_CONFIDENCE_THRESHOLD
        )
        return {
            "should_enter": should_enter,
            "p_up": avg_p_up,
            "confidence": avg_conf,
            "key_reason": key_reason,
        }

    @staticmethod
    def _merge_hold(results: list) -> Dict:
        """Merge multiple model outputs into one hold decision."""
        valid = [r for r in results if r and "action" in r]
        if not valid:
            return {
                "action": "hold",
                "scale_out_pct": 0.0,
                "p_up": 0.5,
                "confidence": 0.0,
                "key_reason": "no AI response, defaulting to hold",
            }
        # majority vote on action
        from collections import Counter
        action_votes = Counter(r["action"] for r in valid)
        action = action_votes.most_common(1)[0][0]
        avg_p_up = sum(r.get("p_up", 0.5) for r in valid) / len(valid)
        avg_conf = sum(r.get("confidence", 0.5) for r in valid) / len(valid)
        scale_out_pct = sum(r.get("scale_out_pct", 0.0) for r in valid) / len(valid)
        reasons = [r.get("key_reason", "") for r in valid if r.get("key_reason")]
        key_reason = reasons[0] if reasons else "models agree"
        return {
            "action": action,
            "scale_out_pct": max(
                CFG.SCALE_OUT_PCT_RANGE[0],
                min(CFG.SCALE_OUT_PCT_RANGE[1], scale_out_pct),
            ),
            "p_up": avg_p_up,
            "confidence": avg_conf,
            "key_reason": key_reason,
        }

    # ------------------------------------------------------------------
    # Rule-based baseline (fallback when no AI keys are configured)
    # ------------------------------------------------------------------

    @staticmethod
    def _rule_entry(symbol: str, snapshot: Dict) -> Dict:
        """Simple rule-based entry decision used when AI is unavailable."""
        price = snapshot.get("price", {})
        social = snapshot.get("social", {})
        derivatives = snapshot.get("derivatives", {})

        change_1h = price.get("change_1h", 0)
        oi_change_4h = derivatives.get("oi_change_4h", 0)
        posts_1h = social.get("posts_1h", 0)
        unique_authors = social.get("unique_authors", 0)
        growth = social.get("posts_growth_rate", 0)
        bullish_ratio = social.get("bullish_tag_ratio", 0.5)

        score = 0.0
        if change_1h > CFG.MIN_PRICE_CHANGE_1H:
            score += 0.25
        if oi_change_4h > CFG.MIN_OI_CHANGE_4H:
            score += 0.25
        if posts_1h >= CFG.MIN_SOCIAL_POSTS_1H and unique_authors >= CFG.MIN_UNIQUE_AUTHORS:
            score += 0.25
        if growth >= CFG.MIN_POSTS_GROWTH:
            score += 0.15
        if bullish_ratio > 0.6:
            score += 0.1

        should_enter = score >= CFG.ENTRY_CONFIDENCE_THRESHOLD
        return {
            "should_enter": should_enter,
            "p_up": score,
            "confidence": score,
            "key_reason": f"rule-based score={score:.2f}",
        }

    @staticmethod
    def _rule_hold(symbol: str, snapshot: Dict, position: Dict) -> Dict:
        """Simple rule-based hold decision used when AI is unavailable."""
        entry_price = position.get("entry_price", 0)
        current_price = snapshot.get("price", {}).get("current_price", entry_price or 1)
        roi = (current_price / entry_price - 1) if entry_price else 0
        p_up = snapshot.get("price", {}).get("change_1h", 0) + 0.5

        if roi >= CFG.HIGH_ROI_THRESHOLD:
            return {
                "action": "scale_out",
                "scale_out_pct": CFG.SCALE_OUT_PCT_RANGE[1],
                "p_up": p_up,
                "confidence": 0.7,
                "key_reason": f"high ROI={roi:.2%}, scaling out",
            }
        if roi < -0.04:
            return {
                "action": "close",
                "scale_out_pct": 0.0,
                "p_up": p_up,
                "confidence": 0.8,
                "key_reason": f"stop-loss triggered ROI={roi:.2%}",
            }
        return {
            "action": "hold",
            "scale_out_pct": 0.0,
            "p_up": p_up,
            "confidence": 0.5,
            "key_reason": f"ROI={roi:.2%}, holding",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_entry(self, symbol: str, snapshot: Dict) -> Dict:
        """判断是否入场。

        Parameters
        ----------
        symbol:
            交易对符号，如 ``"BTC/USDT:USDT"``。
        snapshot:
            ``DataFetcher.snapshot()`` 返回的完整行情快照字典。

        Returns
        -------
        dict
            包含 ``should_enter`` (bool), ``p_up`` (float),
            ``confidence`` (float), ``key_reason`` (str)。
        """
        if self._claude is None and self._gpt is None:
            return self._rule_entry(symbol, snapshot)

        user_msg = json.dumps({"symbol": symbol, "snapshot": snapshot}, ensure_ascii=False)

        results = []
        if self._claude is not None:
            results.append(self._call_claude(_ENTRY_SYSTEM, user_msg))
        if self._gpt is not None:
            results.append(self._call_gpt(_ENTRY_SYSTEM, user_msg))

        decision = self._merge_entry(results)
        logger.debug(
            f"[AIEvaluator] entry {symbol}: "
            f"p_up={decision['p_up']:.2f} conf={decision['confidence']:.2f} "
            f"enter={decision['should_enter']}"
        )
        return decision

    def evaluate_hold(self, symbol: str, snapshot: Dict, position: Dict) -> Dict:
        """判断持仓应如何操作。

        Parameters
        ----------
        symbol:
            交易对符号。
        snapshot:
            当前行情快照。
        position:
            当前持仓字典，至少包含 ``entry_price`` 字段。

        Returns
        -------
        dict
            包含 ``action`` (hold/scale_out/close/add),
            ``scale_out_pct`` (float), ``p_up`` (float),
            ``confidence`` (float), ``key_reason`` (str)。
        """
        if self._claude is None and self._gpt is None:
            return self._rule_hold(symbol, snapshot, position)

        user_msg = json.dumps(
            {"symbol": symbol, "snapshot": snapshot, "position": position},
            ensure_ascii=False,
        )

        results = []
        if self._claude is not None:
            results.append(self._call_claude(_HOLD_SYSTEM, user_msg))
        if self._gpt is not None:
            results.append(self._call_gpt(_HOLD_SYSTEM, user_msg))

        decision = self._merge_hold(results)
        logger.debug(
            f"[AIEvaluator] hold {symbol}: action={decision['action']} "
            f"p_up={decision['p_up']:.2f} conf={decision['confidence']:.2f}"
        )
        return decision
