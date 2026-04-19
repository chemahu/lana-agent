"""双模型 AI 评估层（Claude + GPT）"""
import json
from typing import Dict, Any, Optional
from loguru import logger
from config import CFG


def _fmt_snapshot(symbol: str, snapshot: Dict) -> str:
    """将 snapshot 格式化为简洁文本供模型阅读。"""
    p = snapshot.get("price", {})
    d = snapshot.get("derivatives", {})
    s = snapshot.get("social", {})
    r = snapshot.get("relative", {})
    return (
        f"Symbol: {symbol}\n"
        f"Price change 1h: {p.get('change_1h', 0):.2%}  "
        f"4h: {p.get('change_4h', 0):.2%}  "
        f"24h: {p.get('change_24h', 0):.2%}\n"
        f"Consecutive green candles: {p.get('consecutive_green_candles', 0)}\n"
        f"Upper wick ratio (avg 3 bars): {p.get('upper_wick_ratio_avg', 0):.2f}\n"
        f"OI change 1h: {d.get('oi_change_1h', 0):.2%}  "
        f"4h: {d.get('oi_change_4h', 0):.2%}\n"
        f"Funding rate: {d.get('funding_rate', 0):.4%}\n"
        f"Social posts 1h: {s.get('posts_1h', 0)}  "
        f"24h: {s.get('posts_24h', 0)}  "
        f"growth: {s.get('posts_growth_rate', 0):.2f}x\n"
        f"Unique authors 1h: {s.get('unique_authors', 0)}\n"
        f"Bullish tag ratio: {s.get('bullish_tag_ratio', 0.5):.2f}  "
        f"KOL mentioned: {s.get('kol_mentioned', False)}\n"
        f"Trade widget count: {s.get('trade_widget_count', 0)}\n"
        f"Rank in top gainers: {r.get('rank_in_gainers', 999)}"
    )


_ENTRY_SYSTEM = (
    "You are an algorithmic crypto futures trading analyst. "
    "Given a market snapshot, decide whether to open a LONG position. "
    "Reply ONLY with a JSON object: "
    "{\"should_enter\": bool, \"p_up\": float 0-1, \"confidence\": float 0-1, "
    "\"key_reason\": str}. "
    "No extra text."
)

_HOLD_SYSTEM = (
    "You are an algorithmic crypto futures trading analyst managing an open LONG. "
    "Decide: HOLD, SCALE_OUT (partial take-profit), or CLOSE_ALL. "
    "Reply ONLY with JSON: "
    "{\"action\": \"HOLD\"|\"SCALE_OUT\"|\"CLOSE_ALL\", "
    "\"p_up\": float 0-1, \"confidence\": float 0-1, "
    "\"scale_out_pct\": float 0-1, \"key_reason\": str}. "
    "No extra text."
)


class AIEvaluator:
    """双模型入场/持仓评估器（Claude 主力 + GPT 校验）。"""

    def __init__(self) -> None:
        self._claude_client: Any = None
        self._openai_client: Any = None
        self._init_clients()

    # ------------------------------------------------------------------
    # Client initialisation
    # ------------------------------------------------------------------

    def _init_clients(self) -> None:
        if CFG.ANTHROPIC_API_KEY:
            try:
                import anthropic  # type: ignore
                self._claude_client = anthropic.Anthropic(
                    api_key=CFG.ANTHROPIC_API_KEY
                )
            except Exception as exc:
                logger.warning(f"[AIEvaluator] Claude init failed: {exc}")

        if CFG.OPENAI_API_KEY:
            try:
                from openai import OpenAI  # type: ignore
                self._openai_client = OpenAI(api_key=CFG.OPENAI_API_KEY)
            except Exception as exc:
                logger.warning(f"[AIEvaluator] OpenAI init failed: {exc}")

    # ------------------------------------------------------------------
    # Internal LLM helpers
    # ------------------------------------------------------------------

    def _ask_claude(self, system: str, user_msg: str) -> Optional[Dict]:
        if self._claude_client is None:
            return None
        try:
            response = self._claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=256,
                system=system,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = response.content[0].text.strip()
            return json.loads(text)
        except Exception as exc:
            logger.warning(f"[AIEvaluator] Claude call failed: {exc}")
            return None

    def _ask_gpt(self, system: str, user_msg: str) -> Optional[Dict]:
        if self._openai_client is None:
            return None
        try:
            response = self._openai_client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=256,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_object"},
            )
            text = response.choices[0].message.content.strip()
            return json.loads(text)
        except Exception as exc:
            logger.warning(f"[AIEvaluator] GPT call failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Fallback heuristic (no LLM available)
    # ------------------------------------------------------------------

    @staticmethod
    def _heuristic_entry(snapshot: Dict) -> Dict:
        p = snapshot.get("price", {})
        d = snapshot.get("derivatives", {})
        s = snapshot.get("social", {})
        change_1h = p.get("change_1h", 0)
        oi_4h = d.get("oi_change_4h", 0)
        posts_1h = s.get("posts_1h", 0)
        funding = d.get("funding_rate", 0)
        score = 0.0
        if change_1h >= CFG.MIN_PRICE_CHANGE_1H:
            score += 0.3
        if oi_4h >= CFG.MIN_OI_CHANGE_4H:
            score += 0.25
        if posts_1h >= CFG.MIN_SOCIAL_POSTS_1H:
            score += 0.2
        if s.get("kol_mentioned"):
            score += 0.1
        if s.get("bullish_tag_ratio", 0.5) > 0.6:
            score += 0.1
        if funding > 0.001:
            score -= 0.15
        p_up = min(max(score, 0.0), 1.0)
        confidence = 0.55
        should_enter = (
            p_up >= CFG.ENTRY_P_UP_THRESHOLD
            and confidence >= CFG.ENTRY_CONFIDENCE_THRESHOLD
        )
        return {
            "should_enter": should_enter,
            "p_up": p_up,
            "confidence": confidence,
            "key_reason": "heuristic fallback",
        }

    @staticmethod
    def _heuristic_hold(symbol: str, snapshot: Dict, roi: float) -> Dict:
        p = snapshot.get("price", {})
        wick = p.get("upper_wick_ratio_avg", 0)
        change_1h = p.get("change_1h", 0)
        if roi >= CFG.HIGH_ROI_THRESHOLD:
            return {
                "action": "CLOSE_ALL",
                "p_up": 0.4,
                "confidence": 0.6,
                "scale_out_pct": 1.0,
                "key_reason": f"ROI {roi:.1%} hit hard take-profit",
            }
        if wick > 2.0 or change_1h < -0.03:
            return {
                "action": "SCALE_OUT",
                "p_up": 0.45,
                "confidence": 0.6,
                "scale_out_pct": 0.5,
                "key_reason": "large upper wick or reversal",
            }
        return {
            "action": "HOLD",
            "p_up": 0.55,
            "confidence": 0.55,
            "scale_out_pct": 0.0,
            "key_reason": "heuristic hold",
        }

    # ------------------------------------------------------------------
    # Dual-model consensus helper
    # ------------------------------------------------------------------

    def _consensus_entry(self, claude_res: Optional[Dict],
                         gpt_res: Optional[Dict],
                         fallback: Dict) -> Dict:
        """合并两个模型的结果；任一有效则使用，两者都有时取均值。"""
        results = [r for r in (claude_res, gpt_res) if r is not None]
        if not results:
            return fallback
        if len(results) == 1:
            res = results[0]
        else:
            res = {
                "p_up": (results[0].get("p_up", 0.5) + results[1].get("p_up", 0.5)) / 2,
                "confidence": (
                    results[0].get("confidence", 0.5) + results[1].get("confidence", 0.5)
                ) / 2,
                "key_reason": results[0].get("key_reason", ""),
                "should_enter": None,
            }
        p_up = float(res.get("p_up", 0.5))
        confidence = float(res.get("confidence", 0.5))
        should_enter = (
            p_up >= CFG.ENTRY_P_UP_THRESHOLD
            and confidence >= CFG.ENTRY_CONFIDENCE_THRESHOLD
        )
        return {
            "should_enter": should_enter,
            "p_up": p_up,
            "confidence": confidence,
            "key_reason": str(res.get("key_reason", "")),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_entry(self, symbol: str, snapshot: Dict) -> Dict:
        """评估是否开多仓。返回 {should_enter, p_up, confidence, key_reason}。"""
        user_msg = _fmt_snapshot(symbol, snapshot)
        fallback = self._heuristic_entry(snapshot)
        claude_res = self._ask_claude(_ENTRY_SYSTEM, user_msg)
        gpt_res = self._ask_gpt(_ENTRY_SYSTEM, user_msg)
        result = self._consensus_entry(claude_res, gpt_res, fallback)
        logger.info(
            f"[AIEvaluator] entry {symbol}: "
            f"enter={result['should_enter']}  "
            f"p_up={result['p_up']:.2f}  "
            f"conf={result['confidence']:.2f}  "
            f"reason={result['key_reason']!r}"
        )
        return result

    def evaluate_hold(self, symbol: str, snapshot: Dict, roi: float) -> Dict:
        """评估持仓动作。返回 {action, p_up, confidence, scale_out_pct, key_reason}。"""
        user_msg = (
            f"{_fmt_snapshot(symbol, snapshot)}\n"
            f"Current ROI: {roi:.4f}"
        )
        fallback = self._heuristic_hold(symbol, snapshot, roi)
        claude_res = self._ask_claude(_HOLD_SYSTEM, user_msg)
        gpt_res = self._ask_gpt(_HOLD_SYSTEM, user_msg)

        results = [r for r in (claude_res, gpt_res) if r is not None]
        if not results:
            return fallback

        if len(results) == 1:
            best = results[0]
        else:
            # If either model says CLOSE_ALL, respect it
            actions = [r.get("action", "HOLD") for r in results]
            if "CLOSE_ALL" in actions:
                best = next(r for r in results if r.get("action") == "CLOSE_ALL")
            elif "SCALE_OUT" in actions:
                best = next(r for r in results if r.get("action") == "SCALE_OUT")
            else:
                best = results[0]

        action = str(best.get("action", "HOLD")).upper()
        if action not in ("HOLD", "SCALE_OUT", "CLOSE_ALL"):
            action = "HOLD"
        result = {
            "action": action,
            "p_up": float(best.get("p_up", 0.5)),
            "confidence": float(best.get("confidence", 0.5)),
            "scale_out_pct": float(best.get("scale_out_pct", 0.0)),
            "key_reason": str(best.get("key_reason", "")),
        }
        logger.info(
            f"[AIEvaluator] hold {symbol}: "
            f"action={result['action']}  "
            f"roi={roi:.2%}  "
            f"reason={result['key_reason']!r}"
        )
        return result
