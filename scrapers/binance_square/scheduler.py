"""常驻后台进程：每隔 INTERVAL_MIN 分钟拉取一次币安广场数据并持久化。

用法
----
直接运行::

    python -m scrapers.binance_square.scheduler

或在代码中调用::

    from scrapers.binance_square.scheduler import SquareScheduler
    scheduler = SquareScheduler(symbols=["BTC", "ETH"])
    scheduler.run()          # 阻塞运行
    scheduler.start()        # 后台线程运行
"""
from __future__ import annotations

import threading
import time
from typing import Iterable, List, Optional

from loguru import logger

from .aggregator import SquareAggregator
from .bot_filter import filter_posts
from .client import SquareClient
from .parser import parse_post
from .storage import PostStorage

# 默认抓取间隔（分钟）
_DEFAULT_INTERVAL_MIN = 2
# 默认监控的代币列表（可被构造参数覆盖）
_DEFAULT_SYMBOLS: List[str] = [
    "BTC", "ETH", "BNB", "SOL", "ARB", "OP", "DOGE", "PEPE",
]


class SquareScheduler:
    """周期性拉取币安广场帖子并写入 SQLite 的调度器。

    Parameters
    ----------
    symbols:
        需要监控的代币符号列表。
    interval_min:
        拉取间隔，单位分钟，默认 2。
    storage:
        ``PostStorage`` 实例；为 None 时自动创建（使用默认路径）。
    client:
        ``SquareClient`` 实例；为 None 时自动创建。
    pages_per_query:
        每次搜索拉取的页数。
    page_size:
        每页帖子数。
    """

    def __init__(
        self,
        symbols: Optional[Iterable[str]] = None,
        interval_min: float = _DEFAULT_INTERVAL_MIN,
        storage: Optional[PostStorage] = None,
        client: Optional[SquareClient] = None,
        pages_per_query: int = 3,
        page_size: int = 20,
    ) -> None:
        self._symbols: List[str] = list(symbols or _DEFAULT_SYMBOLS)
        self._interval_sec = interval_min * 60
        self._storage = storage or PostStorage()
        self._client = client or SquareClient()
        self._pages = pages_per_query
        self._page_size = page_size
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_prune_ts: float = 0.0  # Unix timestamp of last prune

    # ------------------------------------------------------------------
    # Internal fetch logic
    # ------------------------------------------------------------------

    def _fetch_symbol(self, symbol: str) -> int:
        """拉取 symbol 相关帖子，过滤 bot，写入存储，返回新增数量。"""
        raw_posts: List[dict] = []
        for page in range(1, self._pages + 1):
            try:
                data = self._client.search(
                    keyword=symbol, page=page, page_size=self._page_size
                )
                items: List[dict] = []
                if isinstance(data, dict):
                    items = data.get("list") or data.get("items") or []
                elif isinstance(data, list):
                    items = data
                for raw in items:
                    try:
                        raw_posts.append(parse_post(raw))
                    except Exception as exc:
                        logger.debug(
                            f"[Scheduler] parse_post error ({symbol}): {exc}"
                        )
                if not items:
                    break
            except Exception as exc:
                logger.warning(
                    f"[Scheduler] search failed for {symbol} page {page}: {exc}"
                )
                break

        # 过滤 bot 后写入存储
        clean_posts = filter_posts(raw_posts)
        added = self._storage.save_posts(clean_posts, symbol)
        return added

    def _cycle(self) -> None:
        """执行一轮所有 symbol 的抓取。"""
        total_added = 0
        for symbol in self._symbols:
            if self._stop_event.is_set():
                break
            try:
                added = self._fetch_symbol(symbol)
                total_added += added
                logger.info(
                    f"[Scheduler] {symbol}: +{added} new posts"
                )
            except Exception as exc:
                logger.error(
                    f"[Scheduler] cycle error for {symbol}: {exc}"
                )
        logger.info(
            f"[Scheduler] cycle complete — total new posts: {total_added}"
        )
        # 仅每 24 小时执行一次旧数据清理，避免每次循环都触发 DELETE
        now = time.monotonic()
        if now - self._last_prune_ts >= 86400:
            try:
                pruned = self._storage.prune_old(keep_days=7)
                self._last_prune_ts = now
                if pruned:
                    logger.debug(f"[Scheduler] pruned {pruned} old posts")
            except Exception as exc:
                logger.warning(f"[Scheduler] prune failed: {exc}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """阻塞运行直到调用 :py:meth:`stop` 或收到 KeyboardInterrupt。"""
        logger.info(
            f"[Scheduler] starting, symbols={self._symbols}, "
            f"interval={self._interval_sec / 60:.1f}min"
        )
        try:
            while not self._stop_event.is_set():
                self._cycle()
                self._stop_event.wait(timeout=self._interval_sec)
        except KeyboardInterrupt:
            logger.info("[Scheduler] interrupted, shutting down")
        finally:
            self._storage.close()
            self._client.close()

    def start(self) -> None:
        """以后台线程启动，立即返回。"""
        if self._thread and self._thread.is_alive():
            logger.warning("[Scheduler] already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self.run,
            name="SquareScheduler",
            daemon=True,
        )
        self._thread.start()
        logger.info("[Scheduler] background thread started")

    def stop(self) -> None:
        """优雅停止（最多等待一个 interval）。"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self._interval_sec + 5)
        logger.info("[Scheduler] stopped")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Binance Square scraper scheduler")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=_DEFAULT_SYMBOLS,
        help="Space-separated list of symbols to monitor",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=_DEFAULT_INTERVAL_MIN,
        help="Fetch interval in minutes (default: 2)",
    )
    args = parser.parse_args()

    scheduler = SquareScheduler(
        symbols=args.symbols,
        interval_min=args.interval,
    )
    scheduler.run()
