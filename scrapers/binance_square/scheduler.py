"""币安广场爬虫调度器：定时拉取指定关键词的帖子并持久化存储。"""
import threading
import time
from typing import Callable, List, Optional

from loguru import logger

from .aggregator import SquareAggregator
from .bot_filter import BotFilter
from .client import SquareClient
from .parser import parse_post
from .storage import PostStorage

_DEFAULT_SYMBOLS = ["BTC", "ETH", "SOL", "BNB"]
_DEFAULT_INTERVAL_SEC = 300   # 每 5 分钟抓一次


class SquareScheduler:
    """后台线程调度器：周期性爬取币安广场帖子并写入 PostStorage。

    Parameters
    ----------
    symbols:
        要监控的标的列表（用作搜索关键词）。
    interval_sec:
        两次爬取间隔秒数。
    client:
        可选的自定义 SquareClient（不传则自动创建）。
    storage:
        可选的自定义 PostStorage（不传则自动创建）。
    on_new_posts:
        每次有新帖子入库时的回调，接收 ``(symbol, new_post_count)``。
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        interval_sec: int = _DEFAULT_INTERVAL_SEC,
        client: Optional[SquareClient] = None,
        storage: Optional[PostStorage] = None,
        on_new_posts: Optional[Callable[[str, int], None]] = None,
    ) -> None:
        self._symbols = symbols or list(_DEFAULT_SYMBOLS)
        self._interval = interval_sec
        self._client = client or SquareClient()
        self._storage = storage or PostStorage()
        self._bot_filter = BotFilter()
        self._on_new_posts = on_new_posts

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Core fetch logic
    # ------------------------------------------------------------------

    def _fetch_and_store(self, symbol: str) -> int:
        """爬取 symbol 相关帖子，过滤后存库。返回新增帖子数。"""
        raw_posts = []
        for page in range(1, 4):  # 最多抓 3 页
            try:
                data = self._client.search(keyword=symbol, page=page, page_size=20)
                items: List = []
                if isinstance(data, dict):
                    items = data.get("list") or data.get("items") or []
                elif isinstance(data, list):
                    items = data
                for raw in items:
                    try:
                        raw_posts.append(parse_post(raw))
                    except Exception as exc:
                        logger.debug(f"[Scheduler] parse_post error: {exc}")
                if not items:
                    break
            except Exception as exc:
                logger.warning(f"[Scheduler] fetch failed {symbol} page {page}: {exc}")
                break

        clean = self._bot_filter.filter(raw_posts)
        inserted = self._storage.save_posts(clean)
        return inserted

    # ------------------------------------------------------------------
    # Scheduler loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        logger.info(
            f"[SquareScheduler] started, symbols={self._symbols}, "
            f"interval={self._interval}s"
        )
        while not self._stop_event.is_set():
            for symbol in self._symbols:
                if self._stop_event.is_set():
                    break
                try:
                    n = self._fetch_and_store(symbol)
                    logger.info(f"[SquareScheduler] {symbol}: +{n} new posts")
                    if n and self._on_new_posts:
                        try:
                            self._on_new_posts(symbol, n)
                        except Exception as cb_exc:
                            logger.warning(f"[SquareScheduler] on_new_posts error: {cb_exc}")
                except Exception as exc:
                    logger.error(f"[SquareScheduler] error for {symbol}: {exc}")

            self._stop_event.wait(timeout=self._interval)

        logger.info("[SquareScheduler] stopped")

    # ------------------------------------------------------------------
    # Public control API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """在后台线程中启动调度循环（幂等）。"""
        if self._thread and self._thread.is_alive():
            logger.warning("[SquareScheduler] already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="SquareScheduler",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        """优雅停止调度循环。"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    def run_once(self) -> None:
        """手动触发一次全量抓取（同步阻塞）。"""
        for symbol in self._symbols:
            try:
                n = self._fetch_and_store(symbol)
                logger.info(f"[SquareScheduler] run_once {symbol}: +{n} new posts")
            except Exception as exc:
                logger.error(f"[SquareScheduler] run_once error for {symbol}: {exc}")

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())
