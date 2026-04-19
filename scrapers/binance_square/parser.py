"""帖子原始 JSON → 标准化记录的解析器。"""
import re
from datetime import datetime, timezone
from typing import Dict, Optional, Set

SYMBOL_RE = re.compile(r"\$?([A-Z0-9]{2,15})(?:USDT|/USDT|币|TOKEN)?")

BULL_TAGS: Set[str] = {
    # English
    "bullish", "bull", "long", "buy", "breakout", "pump", "moon", "mooning",
    "uptrend", "rally", "surge", "gain", "gains", "up", "green", "ath",
    "accumulate", "accumulation", "hodl", "hold", "strong", "support",
    # Chinese
    "看涨", "多头", "做多", "看多", "涨", "拉升", "突破", "加仓", "囤币",
}

BEAR_TAGS: Set[str] = {
    # English
    "bearish", "bear", "short", "sell", "dump", "crash", "drop", "downtrend",
    "falling", "correction", "resistance", "breakdown", "red", "loss", "losses",
    "down", "rekt", "liquidation",
    # Chinese
    "看跌", "空头", "做空", "看空", "跌", "下跌", "崩盘", "减仓", "清仓",
}

KOL_UIDS: Set[str] = set()

_SYMBOL_BLACKLIST: Set[str] = {
    "THE", "AND", "FOR", "YOU", "BTC", "USD", "USDT", "NEW",
    "BUY", "SELL", "NOW", "LONG", "SHORT", "PUMP", "DUMP",
    "BULL", "BEAR", "OUT", "ALL", "GET",
}

_DEFAULT_NAME_RE = re.compile(r"User-?[A-Fa-f0-9]{4,16}")


def _extract_symbols(text: str) -> list:
    """从文本中提取代币符号并过滤黑名单与过短项。"""
    raw = SYMBOL_RE.findall(text.upper())
    unique_symbols: list = []
    for sym in raw:
        if sym in _SYMBOL_BLACKLIST:
            continue
        if len(sym) < 2:
            continue
        if sym not in unique_symbols:
            unique_symbols.append(sym)
    return unique_symbols


def _sentiment(text: str) -> str:
    """根据文本中的关键词返回 'bullish' / 'bearish' / 'neutral'。"""
    lower = text.lower()
    bull_hits = sum(1 for tag in BULL_TAGS if tag in lower)
    bear_hits = sum(1 for tag in BEAR_TAGS if tag in lower)
    if bull_hits > bear_hits:
        return "bullish"
    if bear_hits > bull_hits:
        return "bearish"
    return "neutral"


def parse_post(raw: Dict) -> Dict:
    """将币安广场单条帖子原始 JSON 解析为标准化记录。

    Parameters
    ----------
    raw:
        API 返回的单帖原始字典。

    Returns
    -------
    dict
        标准化字段集合（见字段说明）。
    """
    # ------------------------------------------------------------------
    # Author fields
    # ------------------------------------------------------------------
    author: Dict = raw.get("author") or raw.get("userInfo") or {}
    author_id: str = str(
        author.get("userId") or author.get("uid") or raw.get("userId", "")
    )
    author_nickname: str = str(
        author.get("nickName") or author.get("nickname") or author.get("name", "")
    )
    author_is_default: bool = bool(_DEFAULT_NAME_RE.fullmatch(author_nickname))
    author_is_kol: bool = author_id in KOL_UIDS

    # ------------------------------------------------------------------
    # Post metadata
    # ------------------------------------------------------------------
    post_id: str = str(raw.get("id") or raw.get("postId") or raw.get("feedId", ""))

    content: str = str(raw.get("content") or raw.get("text") or "")

    # Timestamps – try milliseconds first, then seconds
    created_at_ms: Optional[int] = None
    raw_ts = raw.get("createTime") or raw.get("publishTime") or raw.get("createdAt")
    if raw_ts is not None:
        try:
            raw_ts_int = int(raw_ts)
            # heuristic: >1e12 → milliseconds
            if raw_ts_int > 1_000_000_000_000:
                created_at_ms = raw_ts_int
            else:
                created_at_ms = raw_ts_int * 1000
        except (ValueError, TypeError):
            pass

    created_at: Optional[str] = None
    if created_at_ms is not None:
        try:
            created_at = datetime.fromtimestamp(
                created_at_ms / 1000, tz=timezone.utc
            ).isoformat()
        except (OSError, OverflowError, ValueError):
            pass

    # ------------------------------------------------------------------
    # Engagement counters
    # ------------------------------------------------------------------
    stats: Dict = raw.get("postStats") or raw.get("stats") or {}
    likes: int = int(raw.get("likeCount") or stats.get("likeCount", 0) or 0)
    comments: int = int(raw.get("commentCount") or stats.get("commentCount", 0) or 0)
    shares: int = int(raw.get("shareCount") or stats.get("shareCount", 0) or 0)
    views: int = int(raw.get("viewCount") or stats.get("viewCount", 0) or 0)

    # ------------------------------------------------------------------
    # Derived fields
    # ------------------------------------------------------------------
    symbols: list = _extract_symbols(content)
    post_sentiment: str = _sentiment(content)
    has_trade_widget: bool = bool(raw.get("tradeWidget") or raw.get("hasTrade"))

    return {
        "id": post_id,
        "author_id": author_id,
        "author_nickname": author_nickname,
        "author_is_default": author_is_default,
        "author_is_kol": author_is_kol,
        "content": content,
        "created_at": created_at,
        "created_at_ms": created_at_ms,
        "likes": likes,
        "comments": comments,
        "shares": shares,
        "views": views,
        "symbols": symbols,
        "sentiment": post_sentiment,
        "has_trade_widget": has_trade_widget,
        "raw": raw,
    }
