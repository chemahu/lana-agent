"""AI 决策评估器：并行调用 Claude 和 GPT 对标的做入场 / 持仓判断，交叉验证后合并结果。"""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, Tuple

from loguru import logger

from config import CFG


def _build_prompt(symbol: str, snapshot: Dict, mode: str = "entry") -> str:
    price = snapshot.get("price", {})
    deriv = snapshot.get("derivatives", {})
    volume = snapshot.get("volume", {})
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
        "=== Volume ===",
        f"  volume_24h_usdt         : {volume.get('volume_24h_usdt', 0):.0f}",
        f"  volume_change_1h        : {volume.get('volume_change_1h', 0):.2%}",
        f"  avg_hourly_volume_usdt  : {volume.get('avg_hourly_volume_usdt', 0):.0f}",
        f"  buy_volume_ratio        : {volume.get('buy_volume_ratio', 0.5):.2f}",
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
    """并行调用 Anthropic Claude 和 OpenAI GPT 对标的做评估，交叉验证后合并结果。

    决策逻辑
    --------
    * 两个模型都可用时：并行调用，合并结果（保守合并策略）。
    * 只有一个模型可用时：使用单模型结果。
    * 两者都未配置时使用内置规则引擎作为兜底。
    """

    def __init__(self) -> None:
        self._anthropic_client: Optional[object] = None
        self._openai_client: Optional[object] = None

        if CFG.ANTHROPIC_API_KEY:
            try:
                import anthropic  # type: ignore
                self._anthropic_client = anthropic.Anthropic(api_key=CFG.ANTHROPIC_API_KEY)
                logger.info("[AIEvaluator] Anthropic Claude client initialised")
            except Exception as exc:
                logger.warning(f"[AIEvaluator] Anthropic init failed: {exc}")

        if CFG.OPENAI_API_KEY:
            try:
                from openai import OpenAI  # type: ignore
                self._openai_client = OpenAI(api_key=CFG.OPENAI_API_KEY)
                logger.info("[AIEvaluator] OpenAI GPT client initialised")
            except Exception as exc:
                logger.warning(f"[AIEvaluator] OpenAI init failed: {exc}")

        if self._anthropic_client and self._openai_client:
            logger.info("[AIEvaluator] dual-model mode: Claude + GPT will vote in parallel")
        elif self._anthropic_client:
            logger.info("[AIEvaluator] single-model mode: Anthropic Claude only")
        elif self._openai_client:
            logger.info("[AIEvaluator] single-model mode: OpenAI GPT only")
        else:
            logger.warning("[AIEvaluator] no LLM configured – using rule-based fallback")

    # ------------------------------------------------------------------
    # Internal: individual model callers
    # ------------------------------------------------------------------

    def _call_anthropic(self, prompt: str) -> str:
        resp = self._anthropic_client.messages.create(  # type: ignore[union-attr]
            model="claude-3-5-sonnet-20241022",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    def _call_openai(self, prompt: str) -> str:
        resp = self._openai_client.chat.completions.create(  # type: ignore[union-attr]
            model="gpt-4o-mini",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content

    # ------------------------------------------------------------------
    # Internal: dual-model parallel call
    # ------------------------------------------------------------------

    def _call_both_models(self, prompt: str) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Call Claude and GPT concurrently; return (claude_result, gpt_result).

        Each element is either a parsed Dict or None if that model failed.
        """
        callers = {
            "claude": self._call_anthropic,
            "gpt": self._call_openai,
        }
        results: Dict[str, Optional[Dict]] = {"claude": None, "gpt": None}

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_name = {
                executor.submit(callers[name], prompt): name
                for name in callers
            }
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    raw = future.result()
                    results[name] = _parse_json_response(raw)
                    logger.debug(f"[AIEvaluator] {name} responded: {results[name]}")
                except Exception as exc:
                    logger.warning(f"[AIEvaluator] {name} call failed: {exc}")

        return results["claude"], results["gpt"]

    # ------------------------------------------------------------------
    # Internal: result merging
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_entry(r1: Dict, r2: Dict) -> Dict:
        """Conservatively merge two entry-evaluation results.

        Both models must agree to enter; p_up and confidence are averaged.
        """
        p_up = (r1.get("p_up", 0.5) + r2.get("p_up", 0.5)) / 2
        confidence = (r1.get("confidence", 0.5) + r2.get("confidence", 0.5)) / 2
        both_enter = r1.get("should_enter", False) and r2.get("should_enter", False)
        # Pick the reason from whichever model was more confident
        primary = r1 if r1.get("confidence", 0) >= r2.get("confidence", 0) else r2
        risks = "; ".join(
            filter(None, [r1.get("risks", ""), r2.get("risks", "")])
        )[:80]
        return {
            "should_enter": both_enter,
            "p_up": round(p_up, 3),
            "confidence": round(confidence, 3),
            "key_reason": primary.get("key_reason", ""),
            "risks": risks,
            "model_votes": {
                "claude_enter": r1.get("should_enter", False),
                "gpt_enter": r2.get("should_enter", False),
            },
        }

    @staticmethod
    def _merge_hold(r1: Dict, r2: Dict) -> Dict:
        """Conservatively merge two hold-evaluation results.

        Takes the more cautious action; p_up and confidence are averaged.
        """
        action_priority = {"close": 0, "scale_out": 1, "hold": 2}
        a1 = r1.get("action", "hold")
        a2 = r2.get("action", "hold")
        # Pick the more cautious (lower priority number) action
        action = a1 if action_priority.get(a1, 2) <= action_priority.get(a2, 2) else a2
        p_up = (r1.get("p_up", 0.5) + r2.get("p_up", 0.5)) / 2
        confidence = (r1.get("confidence", 0.5) + r2.get("confidence", 0.5)) / 2
        # Average scale_out_pct only when the merged action is scale_out
        if action == "scale_out":
            scale_out_pct = (
                r1.get("scale_out_pct", 0.0) + r2.get("scale_out_pct", 0.0)
            ) / 2
        else:
            scale_out_pct = 0.0
        primary = r1 if action_priority.get(a1, 2) <= action_priority.get(a2, 2) else r2
        return {
            "action": action,
            "scale_out_pct": round(scale_out_pct, 3),
            "p_up": round(p_up, 3),
            "confidence": round(confidence, 3),
            "key_reason": primary.get("key_reason", ""),
            "model_votes": {"claude_action": a1, "gpt_action": a2},
        }

    # ------------------------------------------------------------------
    # Rule-based fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _rule_entry(snapshot: Dict) -> Dict:
        price = snapshot.get("price", {})
        deriv = snapshot.get("derivatives", {})
        social = snapshot.get("social", {})
        volume = snapshot.get("volume", {})

        p_up = 0.5
        p_up += min(price.get("change_1h", 0) * 3, 0.15)
        p_up += min(deriv.get("oi_change_4h", 0) * 0.5, 0.1)
        if social.get("kol_mentioned"):
            p_up += 0.05
        p_up += (social.get("bullish_tag_ratio", 0.5) - 0.5) * 0.1
        # Volume confirmation: above-average 1h volume boosts confidence slightly
        p_up += min(volume.get("volume_change_1h", 0) * 0.1, 0.05)
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
        """判断是否入场。返回包含 should_enter / p_up / key_reason 的字典。

        当 Claude 和 GPT 均可用时并行调用，交叉验证后保守合并；
        只有一个模型时使用该模型；两者均无则降级为规则引擎。
        """
        try:
            prompt = _build_prompt(symbol, snapshot, mode="entry")

            if self._anthropic_client and self._openai_client:
                # Dual-model parallel evaluation
                claude_res, gpt_res = self._call_both_models(prompt)
                if claude_res is not None and gpt_res is not None:
                    result = self._merge_entry(claude_res, gpt_res)
                    logger.info(
                        f"[AIEvaluator] entry {symbol} dual-model: "
                        f"claude={claude_res.get('should_enter')} "
                        f"gpt={gpt_res.get('should_enter')} "
                        f"merged should_enter={result['should_enter']} "
                        f"p_up={result['p_up']:.2f} conf={result['confidence']:.2f}"
                    )
                elif claude_res is not None:
                    result = claude_res
                elif gpt_res is not None:
                    result = gpt_res
                else:
                    return self._rule_entry(snapshot)
            elif self._anthropic_client:
                raw = self._call_anthropic(prompt)
                result = _parse_json_response(raw)
            elif self._openai_client:
                raw = self._call_openai(prompt)
                result = _parse_json_response(raw)
            else:
                return self._rule_entry(snapshot)

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
        except Exception as exc:
            logger.error(f"[AIEvaluator] evaluate_entry failed for {symbol}: {exc}")
            return self._rule_entry(snapshot)

    def evaluate_hold(self, symbol: str, snapshot: Dict) -> Dict:
        """判断持仓标的是继续持有、部分减仓还是全部平仓。

        当 Claude 和 GPT 均可用时并行调用，取更保守的操作合并；
        只有一个模型时使用该模型；两者均无则降级为规则引擎。
        """
        try:
            prompt = _build_prompt(symbol, snapshot, mode="hold")

            if self._anthropic_client and self._openai_client:
                # Dual-model parallel evaluation
                claude_res, gpt_res = self._call_both_models(prompt)
                if claude_res is not None and gpt_res is not None:
                    result = self._merge_hold(claude_res, gpt_res)
                    logger.info(
                        f"[AIEvaluator] hold {symbol} dual-model: "
                        f"claude={claude_res.get('action')} "
                        f"gpt={gpt_res.get('action')} "
                        f"merged action={result['action']}"
                    )
                elif claude_res is not None:
                    result = claude_res
                elif gpt_res is not None:
                    result = gpt_res
                else:
                    return self._rule_hold(snapshot)
            elif self._anthropic_client:
                raw = self._call_anthropic(prompt)
                result = _parse_json_response(raw)
            elif self._openai_client:
                raw = self._call_openai(prompt)
                result = _parse_json_response(raw)
            else:
                return self._rule_hold(snapshot)

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
        except Exception as exc:
            logger.error(f"[AIEvaluator] evaluate_hold failed for {symbol}: {exc}")
            return self._rule_hold(snapshot)
