"""AI 决策评估器：调用最多 3 个 LLM（Anthropic Claude / OpenAI GPT / Google Gemini）
并通过集成投票（少数服从多数 + 多数派概率均值）形成最终交易判断。

集成思路（更容易根据结果判断交易操作）：
  - 每个已配置的 LLM 都会被调用一次，得到独立的 JSON 决策；
  - 入场 should_enter / 持仓 action 采用**少数服从多数**：得票最多的取胜，平票时按
    第一个出现的取胜（行为稳定，便于回放）；
  - p_up、confidence、scale_out_pct **只在多数派内部取均值**，少数派意见仅作记录、
    不稀释多数派的把握度；
  - key_reason / risks 拼接所有模型的简要理由（多数派在前），方便人工复盘；
  - 全部 LLM 调用失败时回退到内置规则引擎；
  - **每次决策**都会向 ``CFG.CONSULTATION_LOG_PATH`` 追加一行 JSON 会商记录，
    包含各模型原始输出、投票明细、最终结论，用于事后复盘。
"""
import json
import os
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger

from config import CFG


def _build_prompt(symbol: str, snapshot: Dict, mode: str = "entry") -> str:
    price = snapshot.get("price", {})
    deriv = snapshot.get("derivatives", {})
    social = snapshot.get("social", {})
    volume = snapshot.get("volume", {})
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
        "=== Volume ===",
        f"  volume_1h           : {volume.get('volume_1h', 0):.0f}",
        f"  volume_change_1h    : {volume.get('volume_change_1h', 0):.2%}",
        f"  volume_vs_avg_24h   : {volume.get('volume_vs_avg_24h', 1.0):.2f}x",
        f"  volume_24h_total    : {volume.get('volume_24h_total', 0):.0f}",
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
    """调用最多 3 个 LLM（Anthropic Claude / OpenAI GPT / Google Gemini）
    对标的做评估，并通过集成投票生成最终结果。

    每个 LLM 互相独立调用、互相对照；
    至少配置任一 API Key 即启用对应模型，全部缺失则使用规则引擎兜底。
    """

    def __init__(self) -> None:
        self._anthropic_client: Optional[object] = None
        self._openai_client: Optional[object] = None
        self._gemini_client: Optional[object] = None

        if CFG.ANTHROPIC_API_KEY:
            try:
                import anthropic  # type: ignore
                self._anthropic_client = anthropic.Anthropic(api_key=CFG.ANTHROPIC_API_KEY)
                logger.info("[AIEvaluator] Anthropic Claude enabled")
            except Exception as exc:
                logger.warning(f"[AIEvaluator] Anthropic init failed: {exc}")

        if CFG.OPENAI_API_KEY:
            try:
                from openai import OpenAI  # type: ignore
                self._openai_client = OpenAI(api_key=CFG.OPENAI_API_KEY)
                logger.info("[AIEvaluator] OpenAI GPT enabled")
            except Exception as exc:
                logger.warning(f"[AIEvaluator] OpenAI init failed: {exc}")

        if CFG.GEMINI_API_KEY:
            try:
                import google.generativeai as genai  # type: ignore
                genai.configure(api_key=CFG.GEMINI_API_KEY)
                self._gemini_client = genai.GenerativeModel("gemini-1.5-flash")
                logger.info("[AIEvaluator] Google Gemini enabled")
            except Exception as exc:
                logger.warning(f"[AIEvaluator] Gemini init failed: {exc}")

        active = sum(1 for c in (self._anthropic_client,
                                  self._openai_client,
                                  self._gemini_client) if c)
        if active == 0:
            logger.warning("[AIEvaluator] no LLM configured – using rule-based fallback")
        else:
            logger.info(f"[AIEvaluator] ensemble active with {active} LLM(s)")

    # ------------------------------------------------------------------
    # Internal: per-model callers
    # ------------------------------------------------------------------

    def _call_anthropic(self, prompt: str) -> str:
        resp = self._anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    def _call_openai(self, prompt: str) -> str:
        resp = self._openai_client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content

    def _call_gemini(self, prompt: str) -> str:
        resp = self._gemini_client.generate_content(prompt)
        # Gemini SDK exposes .text on success; empty/missing means a refusal or
        # safety block — surface it as an error so _gather_results drops this
        # vote instead of producing a JSON-parse failure downstream.
        text = getattr(resp, "text", "") or ""
        if not text.strip():
            raise RuntimeError("gemini returned empty response")
        return text

    def _active_callers(self) -> List[Tuple[str, Callable[[str], str]]]:
        """返回所有已配置的 (模型名, 调用函数) 列表。"""
        callers: List[Tuple[str, Callable[[str], str]]] = []
        if self._anthropic_client:
            callers.append(("anthropic", self._call_anthropic))
        if self._openai_client:
            callers.append(("openai", self._call_openai))
        if self._gemini_client:
            callers.append(("gemini", self._call_gemini))
        return callers

    def _gather_results(self, prompt: str) -> List[Tuple[str, Dict]]:
        """依次调用所有已配置的 LLM，返回 [(模型名, 解析后的 dict), ...]。"""
        results: List[Tuple[str, Dict]] = []
        for name, caller in self._active_callers():
            try:
                raw = caller(prompt)
                parsed = _parse_json_response(raw)
                results.append((name, parsed))
            except Exception as exc:
                logger.warning(f"[AIEvaluator] {name} call failed: {exc}")
        return results

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _avg(values: List[float], default: float) -> float:
        nums = [float(v) for v in values if isinstance(v, (int, float))]
        return sum(nums) / len(nums) if nums else default

    @staticmethod
    def _majority_vote(values: List, default):
        """简单多数投票。

        Parameters
        ----------
        values:
            待投票的取值列表（每个元素代表一个模型的选择）。
        default:
            ``values`` 为空时返回的默认值。

        Returns
        -------
        (winner, win_count, total): tuple
            * ``winner`` —— 票数最多的取值；并列时取第一个出现的值（稳定）。
            * ``win_count`` —— 获胜取值的票数。
            * ``total`` —— 参与投票的总票数。
        """
        counts: Dict = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1
        if not counts:
            return default, 0, 0
        winner = max(counts.items(), key=lambda kv: kv[1])
        return winner[0], winner[1], len(values)

    @classmethod
    def _aggregate_entry(cls, results: List[Tuple[str, Dict]]) -> Dict:
        """集成入场决策：should_enter 按少数服从多数，p_up/confidence 仅在多数派内取均值。"""
        if not results:
            raise RuntimeError("no LLM results to aggregate")

        votes = [bool(r.get("should_enter", False)) for _, r in results]
        winner, win_count, total = cls._majority_vote(votes, False)

        # 只对多数派取均值，少数票不稀释多数派的 p_up / confidence
        winners = [(name, r) for (name, r), v in zip(results, votes) if v == winner]
        dissenters = [(name, r) for (name, r), v in zip(results, votes) if v != winner]

        p_up = cls._avg([r.get("p_up", 0.5) for _, r in winners], 0.5)
        confidence = cls._avg([r.get("confidence", 0.5) for _, r in winners], 0.5)

        # 多数派在前，少数派在后，便于人工对照
        ordered = winners + dissenters
        reason = " | ".join(
            f"{name}:{(r.get('key_reason') or '')[:60]}" for name, r in ordered
        )
        risks = " | ".join(
            f"{name}:{(r.get('risks') or '')[:60]}" for name, r in ordered
        )
        return {
            "should_enter": bool(winner),
            "p_up": round(max(0.0, min(1.0, p_up)), 3),
            "confidence": round(max(0.0, min(1.0, confidence)), 3),
            "key_reason": reason[:240],
            "risks": risks[:240],
            "votes": {
                "winner": bool(winner),
                "win_count": win_count,
                "total": total,
                "models": [name for name, _ in results],
                "majority": [name for name, _ in winners],
                "dissenters": [name for name, _ in dissenters],
            },
        }

    @classmethod
    def _aggregate_hold(cls, results: List[Tuple[str, Dict]]) -> Dict:
        """集成持仓决策：action 按少数服从多数，scale_out_pct/p_up/confidence 仅在多数派内取均值。"""
        if not results:
            raise RuntimeError("no LLM results to aggregate")

        actions = [str(r.get("action", "hold")).lower() for _, r in results]
        winner, win_count, total = cls._majority_vote(actions, "hold")

        winners = [(name, r) for (name, r), a in zip(results, actions) if a == winner]
        dissenters = [(name, r) for (name, r), a in zip(results, actions) if a != winner]

        scale_out_pct = cls._avg(
            [r.get("scale_out_pct", 0.0) for _, r in winners], 0.0
        )
        p_up = cls._avg([r.get("p_up", 0.5) for _, r in winners], 0.5)
        confidence = cls._avg([r.get("confidence", 0.5) for _, r in winners], 0.5)

        ordered = winners + dissenters
        reason = " | ".join(
            f"{name}:{(r.get('key_reason') or '')[:60]}" for name, r in ordered
        )
        return {
            "action": winner,
            "scale_out_pct": round(max(0.0, min(1.0, scale_out_pct)), 3),
            "p_up": round(max(0.0, min(1.0, p_up)), 3),
            "confidence": round(max(0.0, min(1.0, confidence)), 3),
            "key_reason": reason[:240],
            "votes": {
                "winner": winner,
                "win_count": win_count,
                "total": total,
                "models": [name for name, _ in results],
                "majority": [name for name, _ in winners],
                "dissenters": [name for name, _ in dissenters],
                "actions": dict(zip([n for n, _ in results], actions)),
            },
        }

    # ------------------------------------------------------------------
    # Consultation log (会商记录)
    # ------------------------------------------------------------------

    @staticmethod
    def _record_consultation(
        symbol: str,
        mode: str,
        results: List[Tuple[str, Dict]],
        final: Dict,
        snapshot_summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        """将一次 LLM 集成会商完整落盘为一行 JSON，便于事后复盘。

        失败不抛异常 —— 记录写入失败不应影响交易主流程。
        """
        path = getattr(CFG, "CONSULTATION_LOG_PATH", "") or ""
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            record = {
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "symbol": symbol,
                "mode": mode,
                "models": [
                    {"name": name, "response": resp}
                    for name, resp in results
                ],
                "final": final,
            }
            if snapshot_summary:
                record["snapshot"] = snapshot_summary
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:  # pragma: no cover - best-effort logging
            logger.warning(f"[AIEvaluator] consultation log write failed: {exc}")

    @staticmethod
    def _snapshot_summary(snapshot: Dict) -> Dict[str, Any]:
        """从原始 snapshot 中抽取关键特征作为复盘上下文。"""
        price = snapshot.get("price", {}) or {}
        deriv = snapshot.get("derivatives", {}) or {}
        social = snapshot.get("social", {}) or {}
        volume = snapshot.get("volume", {}) or {}
        return {
            "current_price": price.get("current_price"),
            "change_1h": price.get("change_1h"),
            "change_4h": price.get("change_4h"),
            "change_24h": price.get("change_24h"),
            "oi_change_4h": deriv.get("oi_change_4h"),
            "funding_rate": deriv.get("funding_rate"),
            "posts_1h": social.get("posts_1h"),
            "kol_mentioned": social.get("kol_mentioned"),
            "volume_vs_avg_24h": volume.get("volume_vs_avg_24h"),
        }


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
            if not self._active_callers():
                return self._rule_entry(snapshot)
            prompt = _build_prompt(symbol, snapshot, mode="entry")
            results = self._gather_results(prompt)
            if not results:
                logger.error(
                    f"[AIEvaluator] all LLMs failed for entry {symbol}; "
                    "falling back to rule engine"
                )
                return self._rule_entry(snapshot)
            result = self._aggregate_entry(results)
            # ensure required keys exist (defensive)
            result.setdefault("should_enter", False)
            result.setdefault("p_up", 0.5)
            result.setdefault("confidence", 0.5)
            result.setdefault("key_reason", "")
            result.setdefault("risks", "")
            # 少数服从多数后再过一次入场阈值（confidence 已是多数派均值，不再缩放）
            if (result["p_up"] < CFG.ENTRY_P_UP_THRESHOLD
                    or result["confidence"] < CFG.ENTRY_CONFIDENCE_THRESHOLD):
                result["should_enter"] = False
            votes = result.get("votes", {})
            logger.info(
                f"[AIEvaluator] entry {symbol}: should_enter={result['should_enter']} "
                f"p_up={result['p_up']:.2f} conf={result['confidence']:.2f} "
                f"vote={votes.get('win_count', '?')}/{votes.get('total', '?')} "
                f"majority={votes.get('majority', [])} "
                f"dissenters={votes.get('dissenters', [])} "
                f"reason={result['key_reason']!r}"
            )
            self._record_consultation(
                symbol, "entry", results, result,
                snapshot_summary=self._snapshot_summary(snapshot),
            )
            return result
        except Exception as exc:
            logger.error(f"[AIEvaluator] evaluate_entry failed for {symbol}: {exc}")
            return self._rule_entry(snapshot)

    def evaluate_hold(self, symbol: str, snapshot: Dict) -> Dict:
        """判断持仓标的是继续持有、部分减仓还是全部平仓。"""
        try:
            if not self._active_callers():
                return self._rule_hold(snapshot)
            prompt = _build_prompt(symbol, snapshot, mode="hold")
            results = self._gather_results(prompt)
            if not results:
                logger.error(
                    f"[AIEvaluator] all LLMs failed for hold {symbol}; "
                    "falling back to rule engine"
                )
                return self._rule_hold(snapshot)
            result = self._aggregate_hold(results)
            result.setdefault("action", "hold")
            result.setdefault("scale_out_pct", 0.0)
            result.setdefault("p_up", 0.5)
            result.setdefault("confidence", 0.5)
            result.setdefault("key_reason", "")
            votes = result.get("votes", {})
            logger.info(
                f"[AIEvaluator] hold {symbol}: action={result['action']} "
                f"p_up={result['p_up']:.2f} conf={result['confidence']:.2f} "
                f"vote={votes.get('win_count', '?')}/{votes.get('total', '?')} "
                f"majority={votes.get('majority', [])} "
                f"dissenters={votes.get('dissenters', [])} "
                f"reason={result['key_reason']!r}"
            )
            self._record_consultation(
                symbol, "hold", results, result,
                snapshot_summary=self._snapshot_summary(snapshot),
            )
            return result
        except Exception as exc:
            logger.error(f"[AIEvaluator] evaluate_hold failed for {symbol}: {exc}")
            return self._rule_hold(snapshot)
