"""AI 决策评估器 —— 调用大模型对入场和持仓进行打分"""
import json
from typing import Dict
from loguru import logger
from config import CFG


def _build_entry_prompt(symbol: str, snapshot: Dict) -> str:
    price = snapshot.get("price", {})
    deriv = snapshot.get("derivatives", {})
    social = snapshot.get("social", {})
    relative = snapshot.get("relative", {})
    return (
        f"You are a crypto futures trading assistant.\n"
        f"Analyze the following market snapshot for {symbol} and decide whether to open a LONG position.\n\n"
        f"Price features: {json.dumps(price, indent=2)}\n"
        f"Derivatives: {json.dumps(deriv, indent=2)}\n"
        f"Social signals: {json.dumps(social, indent=2)}\n"
        f"Relative rank: {json.dumps(relative, indent=2)}\n\n"
        f"Respond with a JSON object containing:\n"
        f"  p_up (float 0-1): probability price goes up in next 4h\n"
        f"  confidence (float 0-1): your confidence in the prediction\n"
        f"  key_reason (str): one-line reason\n"
        f"  should_enter (bool): true if recommended to enter\n"
        f"Return ONLY the raw JSON object, no markdown."
    )


def _build_hold_prompt(symbol: str, snapshot: Dict, unrealized_pnl_pct: float) -> str:
    price = snapshot.get("price", {})
    deriv = snapshot.get("derivatives", {})
    social = snapshot.get("social", {})
    return (
        f"You are a crypto futures trading assistant managing an open LONG on {symbol}.\n"
        f"Unrealized PnL: {unrealized_pnl_pct:.2%}\n\n"
        f"Price features: {json.dumps(price, indent=2)}\n"
        f"Derivatives: {json.dumps(deriv, indent=2)}\n"
        f"Social signals: {json.dumps(social, indent=2)}\n\n"
        f"Respond with a JSON object containing:\n"
        f"  p_up (float 0-1): probability price continues up in next 4h\n"
        f"  confidence (float 0-1): your confidence\n"
        f"  action (str): one of 'hold', 'close', 'scale_out'\n"
        f"  scale_out_pct (float 0-1): fraction to close if action is scale_out\n"
        f"  key_reason (str): one-line reason\n"
        f"Return ONLY the raw JSON object, no markdown."
    )


def _call_llm(prompt: str) -> str:
    """Try Anthropic first, fall back to OpenAI."""
    if CFG.ANTHROPIC_API_KEY:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=CFG.ANTHROPIC_API_KEY)
            msg = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        except Exception as e:
            logger.warning(f"Anthropic call failed: {e}")

    if CFG.OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=CFG.OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.warning(f"OpenAI call failed: {e}")

    raise RuntimeError("No LLM API key configured (ANTHROPIC_API_KEY / OPENAI_API_KEY)")


def _parse_json_response(raw: str) -> Dict:
    """Extract JSON from LLM response, stripping markdown fences if present."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    return json.loads(text)


class AIEvaluator:
    """调用大模型对入场 / 持仓信号进行评估。"""

    # ------------------------------------------------------------------
    # Entry evaluation
    # ------------------------------------------------------------------
    def evaluate_entry(self, symbol: str, snapshot: Dict) -> Dict:
        """
        Returns a dict with keys:
          should_enter (bool), p_up (float), confidence (float), key_reason (str)
        """
        default = {
            "should_enter": False,
            "p_up": 0.5,
            "confidence": 0.0,
            "key_reason": "evaluation skipped",
        }
        try:
            prompt = _build_entry_prompt(symbol, snapshot)
            raw = _call_llm(prompt)
            result = _parse_json_response(raw)
            p_up = float(result.get("p_up", 0.5))
            confidence = float(result.get("confidence", 0.0))
            should_enter = (
                bool(result.get("should_enter", False))
                and p_up >= CFG.ENTRY_P_UP_THRESHOLD
                and confidence >= CFG.ENTRY_CONFIDENCE_THRESHOLD
            )
            return {
                "should_enter": should_enter,
                "p_up": p_up,
                "confidence": confidence,
                "key_reason": str(result.get("key_reason", "")),
            }
        except Exception as e:
            logger.error(f"[AIEvaluator] evaluate_entry failed for {symbol}: {e}")
            return default

    # ------------------------------------------------------------------
    # Hold / exit evaluation
    # ------------------------------------------------------------------
    def evaluate_hold(self, symbol: str, snapshot: Dict,
                      unrealized_pnl_pct: float = 0.0) -> Dict:
        """
        Returns a dict with keys:
          action (str), p_up (float), confidence (float),
          scale_out_pct (float), key_reason (str)
        """
        default = {
            "action": "hold",
            "p_up": 0.5,
            "confidence": 0.0,
            "scale_out_pct": 0.0,
            "key_reason": "evaluation skipped",
        }
        try:
            prompt = _build_hold_prompt(symbol, snapshot, unrealized_pnl_pct)
            raw = _call_llm(prompt)
            result = _parse_json_response(raw)
            action = str(result.get("action", "hold"))
            if action not in ("hold", "close", "scale_out"):
                action = "hold"
            scale_out_pct = float(result.get("scale_out_pct", 0.0))
            min_pct, max_pct = CFG.SCALE_OUT_PCT_RANGE
            scale_out_pct = max(min_pct, min(max_pct, scale_out_pct))
            return {
                "action": action,
                "p_up": float(result.get("p_up", 0.5)),
                "confidence": float(result.get("confidence", 0.0)),
                "scale_out_pct": scale_out_pct,
                "key_reason": str(result.get("key_reason", "")),
            }
        except Exception as e:
            logger.error(f"[AIEvaluator] evaluate_hold failed for {symbol}: {e}")
            return default
