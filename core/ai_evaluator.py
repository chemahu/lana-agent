"""AI 决策评估器：调用 LLM（Claude 或 OpenAI）对标的做入场 / 持仓判断。"""
import json
from typing import Dict, Optional

from loguru import logger

from config import CFG


def _build_prompt(symbol: str, snapshot: Dict, mode: str = "entry") -> str:
    price = snapshot.get("price", {})
    deriv = snapshot.get("derivatives", {})
    social = snapshot.get("social", {})
    relative = snapshot.get("relative", {})

    lines = [
        f"Symbol: {symbol}",
        f"Timestamp: {snapshot.get('timestamp', 'unknown')}",
        "",
        "=== Price Features ===",
        f"  current_price          : {price.get('current_price', 'N/A')}",
        f"  change_1h              : {price.get('change_1h', 0):.2%}",
        f"  change_4h              : {price.get('change_4h', 0):.2%}",
        f"  change_24h             : {price.get('change_24h', 0):.2%}",
        f"  distance_from_24h_high : {price.get('distance_from_24h_high', 0):.2%}",
        f"  consecutive_green      : {price.get('consecutive_green_candles', 0)}",
        f"  upper_wick_ratio_avg   : {price.get('upper_wick_ratio_avg', 0):.3f}",
        "",
        "=== Derivatives ===",
        f"  oi_change_1h  : {deriv.get('oi_change_1h', 0):.2%}",
        f"  oi_change_4h  : {deriv.get('oi_change_4h', 0):.2%}",
        f"  funding_rate  : {deriv.get('funding_rate', 0):.4%}",
        "",
        "=== Social ===",
        f"  posts_1h          : {social.get('posts_1h', 0)}",
        f"  posts_24h         : {social.get('posts_24h', 0)}",
        f"  posts_growth_rate : {social.get('posts_growth_rate', 0):.2f}",
        f"  unique_authors    : {social.get('unique_authors', 0)}",
        f"  bullish_tag_ratio : {social.get('bullish_tag_ratio', 0.5):.2f}",
        f"  kol_mentioned     : {social.get('kol_mentioned', False)}",
        f"  trade_widget_cnt  : {social.get('trade_widget_count', 0)}",
        "",
        "=== Relative ===",
        f"  rank_in_gainers : {relative.get('rank', relative.get('rank_in_gainers', 'N/A'))}",
    ]
    context = "\n".join(lines)

    if mode == "entry":
        return (
            "You are a professional crypto futures trader. "
            "Analyze the following market snapshot and decide whether to open a LONG position.\n\n"
            f"{context}\n\n"
            "Reply ONLY with a JSON object with these exact keys:\n"
            '  "should_enter": bool,\n'
            '  "p_up": float (0-1, estimated probability price rises >3% in 4h),\n'
            '  "confidence": float (0-1, your confidence in the estimate),\n'
            '  "key_reason": str (≤80 chars, the single most important reason),\n'
            '  "risks": str (≤80 chars, main risk)\n'
            "No explanation outside the JSON."
        )
    else:  # hold / exit
        return (
            "You are a professional crypto futures trader managing an open LONG position. "
            "Analyze the following market snapshot and decide whether to HOLD or EXIT.\n\n"
            f"{context}\n\n"
            "Reply ONLY with a JSON object with these exact keys:\n"
            '  "action": "hold" | "scale_out" | "close",\n'
            '  "scale_out_pct": float (0-1, portion to close, 0 if action is not scale_out),\n'
            '  "p_up": float (0-1, estimated probability price keeps rising),\n'
            '  "confidence": float (0-1),\n'
            '  "key_reason": str (≤80 chars)\n'
            "No explanation outside the JSON."
        )


def _parse_json_response(text: str) -> Dict:
    """Extract and parse the first JSON object found in *text*."""
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in response: {text!r}")
    return json.loads(text[start:end])


class AIEvaluator:
    """调用 Anthropic Claude 或 OpenAI GPT 对标的做评估。

    优先尝试 Anthropic；若未配置 ANTHROPIC_API_KEY 则回落到 OpenAI；
    两者都未配置则使用内置规则引擎作为兜底。
    """

    def __init__(self) -> None:
        self._anthropic_client: Optional[object] = None
        self._openai_client: Optional[object] = None

        if CFG.ANTHROPIC_API_KEY:
            try:
                import anthropic  # type: ignore
                self._anthropic_client = anthropic.Anthropic(api_key=CFG.ANTHROPIC_API_KEY)
                logger.info("[AIEvaluator] using Anthropic Claude")
            except Exception as exc:
                logger.warning(f"[AIEvaluator] Anthropic init failed: {exc}")

        if not self._anthropic_client and CFG.OPENAI_API_KEY:
            try:
                from openai import OpenAI  # type: ignore
                self._openai_client = OpenAI(api_key=CFG.OPENAI_API_KEY)
                logger.info("[AIEvaluator] using OpenAI GPT")
            except Exception as exc:
                logger.warning(f"[AIEvaluator] OpenAI init failed: {exc}")

        if not self._anthropic_client and not self._openai_client:
            logger.warning("[AIEvaluator] no LLM configured – using rule-based fallback")

    # ------------------------------------------------------------------
    # Internal: call LLM
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        if self._anthropic_client:
            resp = self._anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text

        if self._openai_client:
            resp = self._openai_client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content

        raise RuntimeError("No LLM client available")

    # ------------------------------------------------------------------
    # Rule-based fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _rule_entry(snapshot: Dict) -> Dict:
        price = snapshot.get("price", {})
        deriv = snapshot.get("derivatives", {})
        social = snapshot.get("social", {})

        p_up = 0.5
        p_up += min(price.get("change_1h", 0) * 3, 0.15)
        p_up += min(deriv.get("oi_change_4h", 0) * 0.5, 0.1)
        if social.get("kol_mentioned"):
            p_up += 0.05
        p_up += (social.get("bullish_tag_ratio", 0.5) - 0.5) * 0.1
        p_up = max(0.0, min(1.0, p_up))

        should_enter = (p_up >= CFG.ENTRY_P_UP_THRESHOLD
                        and price.get("upper_wick_ratio_avg", 0) < 2.0)
        return {
            "should_enter": should_enter,
            "p_up": round(p_up, 3),
            "confidence": 0.5,
            "key_reason": "rule-based: momentum + OI growth",
            "risks": "no LLM configured; rule-based only",
        }

    @staticmethod
    def _rule_hold(snapshot: Dict) -> Dict:
        price = snapshot.get("price", {})
        p_up = 0.5 + min(price.get("change_1h", 0) * 2, 0.2)
        p_up = max(0.0, min(1.0, p_up))
        action = "hold" if p_up >= 0.5 else "close"
        return {
            "action": action,
            "scale_out_pct": 0.0,
            "p_up": round(p_up, 3),
            "confidence": 0.5,
            "key_reason": "rule-based momentum check",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_entry(self, symbol: str, snapshot: Dict) -> Dict:
        """判断是否入场。返回包含 should_enter / p_up / key_reason 的字典。"""
        try:
            prompt = _build_prompt(symbol, snapshot, mode="entry")
            raw = self._call_llm(prompt)
            result = _parse_json_response(raw)
            # ensure required keys exist
            result.setdefault("should_enter", False)
            result.setdefault("p_up", 0.5)
            result.setdefault("confidence", 0.5)
            result.setdefault("key_reason", "")
            result.setdefault("risks", "")
            # enforce thresholds
            if (result["p_up"] < CFG.ENTRY_P_UP_THRESHOLD
                    or result["confidence"] < CFG.ENTRY_CONFIDENCE_THRESHOLD):
                result["should_enter"] = False
            logger.info(
                f"[AIEvaluator] entry {symbol}: should_enter={result['should_enter']} "
                f"p_up={result['p_up']:.2f} conf={result['confidence']:.2f} "
                f"reason={result['key_reason']!r}"
            )
            return result
        except RuntimeError:
            return self._rule_entry(snapshot)
        except Exception as exc:
            logger.error(f"[AIEvaluator] evaluate_entry failed for {symbol}: {exc}")
            return self._rule_entry(snapshot)

    def evaluate_hold(self, symbol: str, snapshot: Dict) -> Dict:
        """判断持仓标的是继续持有、部分减仓还是全部平仓。"""
        try:
            prompt = _build_prompt(symbol, snapshot, mode="hold")
            raw = self._call_llm(prompt)
            result = _parse_json_response(raw)
            result.setdefault("action", "hold")
            result.setdefault("scale_out_pct", 0.0)
            result.setdefault("p_up", 0.5)
            result.setdefault("confidence", 0.5)
            result.setdefault("key_reason", "")
            logger.info(
                f"[AIEvaluator] hold {symbol}: action={result['action']} "
                f"p_up={result['p_up']:.2f} reason={result['key_reason']!r}"
            )
            return result
        except RuntimeError:
            return self._rule_hold(snapshot)
        except Exception as exc:
            logger.error(f"[AIEvaluator] evaluate_hold failed for {symbol}: {exc}")
            return self._rule_hold(snapshot)
