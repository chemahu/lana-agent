"""双模型 AI 评估器（Claude + GPT 互备）。

评估维度：
- ``evaluate_entry``：给定 symbol + 市场快照，输出是否进场及置信度。
- ``evaluate_hold``：给定已持仓 symbol，输出继续持仓 / 分批止盈 / 全平建议。
- ``evaluate_ambiguous``：在主评估结果模糊时触发备用模型交叉验证。
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from loguru import logger

from config import CFG

# ---------------------------------------------------------------------------
# 提示词模板
# ---------------------------------------------------------------------------

_ENTRY_SYSTEM = (
    "You are Lana, a quantitative crypto futures analyst. "
    "Evaluate whether to open a long position on {symbol}. "
    "Reply ONLY with a JSON object (no markdown, no explanation) in this schema: "
    '{"should_enter": bool, "p_up": float (0-1), "confidence": float (0-1), '
    '"key_reason": str, "risk_flags": list[str]}'
)

_HOLD_SYSTEM = (
    "You are Lana, a quantitative crypto futures risk manager. "
    "Evaluate what to do with the current long position on {symbol}. "
    "Reply ONLY with a JSON object: "
    '{"action": "hold"|"scale_out"|"close_all", '
    '"p_up": float (0-1), "confidence": float (0-1), '
    '"scale_out_pct": float (0-1) or null, "key_reason": str}'
)

_SNAPSHOT_TEMPLATE = "Market snapshot:\n{snapshot}"


def _fmt_snapshot(snapshot: Dict) -> str:
    try:
        return json.dumps(snapshot, ensure_ascii=False, indent=2)
    except Exception:
        return str(snapshot)


# ---------------------------------------------------------------------------
# Helpers: parse LLM response
# ---------------------------------------------------------------------------

def _parse_json_safe(text: str) -> Dict:
    """从 LLM 响应中提取 JSON 对象，容忍 markdown 代码块包裹。"""
    text = text.strip()
    # 去掉可能的 ```json ... ``` 包裹
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [l for l in lines if not l.startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned non-JSON response: {text!r}") from exc


# ---------------------------------------------------------------------------
# Provider wrappers
# ---------------------------------------------------------------------------

def _call_claude(system: str, user_content: str) -> str:
    """调用 Anthropic Claude，返回文本响应。"""
    import anthropic  # type: ignore

    client = anthropic.Anthropic(api_key=CFG.ANTHROPIC_API_KEY)
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": user_content}],
    )
    return message.content[0].text


def _call_openai(system: str, user_content: str) -> str:
    """调用 OpenAI GPT，返回文本响应。"""
    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=CFG.OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=512,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
    )
    return response.choices[0].message.content or ""


def _call_llm(system: str, user_content: str) -> str:
    """优先 Claude，失败自动降级到 GPT。"""
    if CFG.ANTHROPIC_API_KEY:
        try:
            return _call_claude(system, user_content)
        except Exception as exc:
            logger.warning(f"[AIEvaluator] Claude failed, falling back to GPT: {exc}")
    if CFG.OPENAI_API_KEY:
        return _call_openai(system, user_content)
    raise RuntimeError(
        "[AIEvaluator] No LLM API key configured "
        "(ANTHROPIC_API_KEY / OPENAI_API_KEY)"
    )


# ---------------------------------------------------------------------------
# Default safe responses (used when LLM is unavailable / parse fails)
# ---------------------------------------------------------------------------

_DEFAULT_ENTRY: Dict[str, Any] = {
    "should_enter": False,
    "p_up": 0.5,
    "confidence": 0.0,
    "key_reason": "AI evaluation unavailable",
    "risk_flags": [],
}

_DEFAULT_HOLD: Dict[str, Any] = {
    "action": "hold",
    "p_up": 0.5,
    "confidence": 0.0,
    "scale_out_pct": None,
    "key_reason": "AI evaluation unavailable",
}


# ---------------------------------------------------------------------------
# Main evaluator class
# ---------------------------------------------------------------------------

class AIEvaluator:
    """双模型 AI 评估器。

    所有方法在 LLM 不可用或返回格式错误时均返回保守默认值（不进场 / 持仓不动），
    保证下游不会因 AI 故障而产生意外交易。
    """

    # ------------------------------------------------------------------
    # Entry evaluation
    # ------------------------------------------------------------------

    def evaluate_entry(self, symbol: str, snapshot: Dict) -> Dict[str, Any]:
        """评估是否开多仓。

        Parameters
        ----------
        symbol:
            交易对符号，如 ``"BTC/USDT:USDT"``。
        snapshot:
            ``DataFetcher.snapshot`` 返回的完整市场快照字典。

        Returns
        -------
        dict
            包含 ``should_enter``, ``p_up``, ``confidence``,
            ``key_reason``, ``risk_flags`` 字段。
        """
        system = _ENTRY_SYSTEM.format(symbol=symbol)
        user_content = _SNAPSHOT_TEMPLATE.format(snapshot=_fmt_snapshot(snapshot))
        try:
            raw = _call_llm(system, user_content)
            result = _parse_json_safe(raw)
            # 阈值校验：仅当两个指标同时达标才允许进场
            should_enter = (
                result.get("p_up", 0) >= CFG.ENTRY_P_UP_THRESHOLD
                and result.get("confidence", 0) >= CFG.ENTRY_CONFIDENCE_THRESHOLD
                and result.get("should_enter", False)
            )
            result["should_enter"] = should_enter
            return result
        except Exception as exc:
            logger.error(f"[AIEvaluator] evaluate_entry failed for {symbol}: {exc}")
            return dict(_DEFAULT_ENTRY)

    # ------------------------------------------------------------------
    # Hold / exit evaluation
    # ------------------------------------------------------------------

    def evaluate_hold(self, symbol: str, snapshot: Dict) -> Dict[str, Any]:
        """评估已持仓头寸的处置方式。

        Returns
        -------
        dict
            包含 ``action``（``hold`` / ``scale_out`` / ``close_all``）、
            ``p_up``、``confidence``、``scale_out_pct``、``key_reason``。
        """
        system = _HOLD_SYSTEM.format(symbol=symbol)
        user_content = _SNAPSHOT_TEMPLATE.format(snapshot=_fmt_snapshot(snapshot))
        try:
            raw = _call_llm(system, user_content)
            return _parse_json_safe(raw)
        except Exception as exc:
            logger.error(f"[AIEvaluator] evaluate_hold failed for {symbol}: {exc}")
            return dict(_DEFAULT_HOLD)

    # ------------------------------------------------------------------
    # Ambiguous cross-validation (second opinion from backup model)
    # ------------------------------------------------------------------

    def evaluate_ambiguous(
        self,
        symbol: str,
        snapshot: Dict,
        primary_result: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """主评估结果模糊时，调用备用模型（GPT）二次确认。

        若主模型为 Claude，则直接用 GPT；反之亦然。
        主 / 备结论一致则置信度加成，否则返回保守默认值。
        该方法在持仓评估（evaluate_hold）模糊时调用，因此使用 _HOLD_SYSTEM 提示词。
        """
        user_content = _SNAPSHOT_TEMPLATE.format(snapshot=_fmt_snapshot(snapshot))
        system = _HOLD_SYSTEM.format(symbol=symbol)

        # 始终尝试备用模型
        backup_result: Optional[Dict] = None
        if CFG.OPENAI_API_KEY:
            try:
                raw = _call_openai(system, user_content)
                backup_result = _parse_json_safe(raw)
            except Exception as exc:
                logger.warning(f"[AIEvaluator] backup model failed: {exc}")
        elif CFG.ANTHROPIC_API_KEY:
            try:
                raw = _call_claude(system, user_content)
                backup_result = _parse_json_safe(raw)
            except Exception as exc:
                logger.warning(f"[AIEvaluator] backup model failed: {exc}")

        if backup_result is None:
            return dict(_DEFAULT_HOLD)

        if primary_result is None:
            return backup_result

        # 两模型结论一致 → 置信度取平均并稍加权
        if primary_result.get("action") == backup_result.get("action"):
            merged = dict(backup_result)
            merged["confidence"] = (
                primary_result.get("confidence", 0) + backup_result.get("confidence", 0)
            ) / 2
            merged["key_reason"] = (
                f"[consensus] {primary_result.get('key_reason', '')} | "
                f"{backup_result.get('key_reason', '')}"
            )
            return merged

        # 两模型结论不一致 → 保守持仓不动
        logger.info(
            f"[AIEvaluator] ambiguous: primary={primary_result.get('action')}, "
            f"backup={backup_result.get('action')} → hold"
        )
        return dict(_DEFAULT_HOLD)
