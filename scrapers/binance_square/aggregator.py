"""币安广场数据聚合器：用 SquareClient 拉取帖子并用 parser 解析，
汇总成下游 DataFetcher 所需的社会热度特征向量。"""
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

from loguru import logger

from .client import SquareClient
from .parser import parse_post


class SquareAggregator:
    """高层聚合器：将原始帖子列表转换为特征字典。"""

    def __init__(
        self,
        client: Optional[SquareClient] = None,
        pages_per_query: int = 3,
        page_size: int = 20,
    ) -> None:
        self._client = client or SquareClient()
        self._pages = pages_per_query
        self._page_size = page_size

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_posts_for_symbol(self, symbol: str) -> List[Dict]:
        """拉取与 symbol 相关的帖子列表（已解析）。"""
        posts: List[Dict] = []
        for page in range(1, self._pages + 1):
            try:
                data = self._client.search(
                    keyword=symbol, page=page, page_size=self._page_size
                )
                items: List[Dict] = []
                if isinstance(data, dict):
                    items = data.get("list") or data.get("items") or []
                elif isinstance(data, list):
                    items = data
                for raw in items:
                    try:
                        posts.append(parse_post(raw))
                    except Exception as exc:
                        logger.debug(f"[SquareAggregator] parse_post error: {exc}")
                if not items:
                    break
            except Exception as exc:
                logger.warning(
                    f"[SquareAggregator] search failed for {symbol} page {page}: {exc}"
                )
                break
        return posts

    @staticmethod
    def _now_ms() -> int:
        return int(datetime.now(tz=timezone.utc).timestamp() * 1000)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def features_for(self, symbol: str) -> Dict:
        """返回 DataFetcher.get_social_features 所需的特征字典。"""
        posts = self._fetch_posts_for_symbol(symbol)

        now_ms = self._now_ms()
        ms_1h = 3_600_000
        ms_24h = 86_400_000

        posts_1h = 0
        posts_24h = 0
        authors_1h: Set[str] = set()
        bull_count = 0
        bear_count = 0
        kol_mentioned = False
        trade_widget_count = 0

        for p in posts:
            created_ms = p.get("created_at_ms")
            if created_ms is None:
                # skip posts with unknown timestamps for age-based counters
                sentiment = p.get("sentiment", "neutral")
                if sentiment == "bullish":
                    bull_count += 1
                elif sentiment == "bearish":
                    bear_count += 1
                if p.get("author_is_kol"):
                    kol_mentioned = True
                if p.get("has_trade_widget"):
                    trade_widget_count += 1
                continue

            age_ms = now_ms - created_ms
            if age_ms <= ms_24h:
                posts_24h += 1
            if age_ms <= ms_1h:
                posts_1h += 1
                authors_1h.add(p["author_id"])

            sentiment = p.get("sentiment", "neutral")
            if sentiment == "bullish":
                bull_count += 1
            elif sentiment == "bearish":
                bear_count += 1

            if p.get("author_is_kol"):
                kol_mentioned = True
            if p.get("has_trade_widget"):
                trade_widget_count += 1

        total_sentiment = bull_count + bear_count
        bullish_tag_ratio = (
            bull_count / total_sentiment if total_sentiment > 0 else 0.5
        )

        # growth rate: posts in last 1 h vs expected baseline from 24 h window
        baseline_1h = posts_24h / 24 if posts_24h > 0 else 0
        posts_growth_rate = (
            (posts_1h / baseline_1h - 1) if baseline_1h > 0 else 0.0
        )

        return {
            "posts_1h": posts_1h,
            "posts_24h": posts_24h,
            "posts_growth_rate": posts_growth_rate,
            "unique_authors": len(authors_1h),
            "bullish_tag_ratio": bullish_tag_ratio,
            "kol_mentioned": kol_mentioned,
            "trade_widget_count": trade_widget_count,
        }
