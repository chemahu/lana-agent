"""币安广场常驻拉取进程：每 FETCH_INTERVAL_SEC 秒抓一轮热门帖子并写入 SQLite。"""
import signal
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional

from loguru import logger

from .aggregator import SquareAggregator
from .client import SquareClient
from .parser import parse_post
from .bot_filter import BotFilter
from .storage import PostStorage

# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------

FETCH_INTERVAL_SEC: int = 120        # 2 分钟拉取一次
PRUNE_OLDER_THAN_HOURS: int = 48     # 48 小时前的帖子定期清理
_PRUNE_EVERY_N_CYCLES: int = 30      # 每 30 个周期做一次清理


class SquareScheduler:
    """常驻后台 scheduler，循环拉取广场帖子并持久化。

    Usage
    -----
    >>> sched = SquareScheduler()
    >>> sched.start()           # 启动后台线程
    >>> ...
    >>> sched.stop()            # 优雅关闭
    """

    def __init__(
        self,
        interval_sec: int = FETCH_INTERVAL_SEC,
        db_path: Optional[Path] = None,
        pages_per_query: int = 5,
        page_size: int = 20,
    ) -> None:
        self._interval = interval_sec
        self._client = SquareClient()
        self._bot_filter = BotFilter()
        self._storage = PostStorage(db_path=db_path)
        self._pages = pages_per_query
        self._page_size = page_size

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cycle_count = 0

    # ------------------------------------------------------------------
    # Start / stop
    # ------------------------------------------------------------------

    def start(self, daemon: bool = True) -> None:
        """在后台线程中启动调度循环。"""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("[SquareScheduler] already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="square-scheduler",
            daemon=daemon,
        )
        self._thread.start()
        logger.info(
            f"[SquareScheduler] started (interval={self._interval}s)"
        )

    def stop(self, timeout: float = 10.0) -> None:
        """发送停止信号并等待线程退出。"""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        logger.info("[SquareScheduler] stopped")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        logger.info("[SquareScheduler] loop started")
        while not self._stop_event.is_set():
            try:
                self._fetch_cycle()
            except Exception as exc:
                logger.error(f"[SquareScheduler] fetch cycle error: {exc}")
            self._stop_event.wait(timeout=self._interval)
        logger.info("[SquareScheduler] loop exited")

    def _fetch_cycle(self) -> None:
        self._cycle_count += 1
        self._bot_filter.reset()

        posts: List[dict] = []

        # Fetch feed list (general hot posts)
        for page in range(1, self._pages + 1):
            if self._stop_event.is_set():
                break
            try:
                data = self._client.fetch_feed_list(
                    page=page, page_size=self._page_size
                )
                items: List[dict] = []
                if isinstance(data, dict):
                    items = data.get("list") or data.get("items") or []
                elif isinstance(data, list):
                    items = data
                for raw in items:
                    try:
                        posts.append(parse_post(raw))
                    except Exception as exc:
                        logger.debug(f"[SquareScheduler] parse error: {exc}")
                if not items:
                    break
            except Exception as exc:
                logger.warning(
                    f"[SquareScheduler] feed_list page {page} failed: {exc}"
                )
                break

        # Filter bots
        clean = self._bot_filter.filter(posts)

        # Persist
        new_count = self._storage.save_posts(clean, symbol="")
        logger.debug(
            f"[SquareScheduler] cycle {self._cycle_count}: "
            f"fetched={len(posts)} clean={len(clean)} new={new_count}"
        )

        # Periodic prune
        if self._cycle_count % _PRUNE_EVERY_N_CYCLES == 0:
            cutoff_ms = int(
                (
                    datetime.now(tz=timezone.utc)
                    - timedelta(hours=PRUNE_OLDER_THAN_HOURS)
                ).timestamp()
                * 1000
            )
            pruned = self._storage.prune_old(cutoff_ms)
            if pruned:
                logger.info(f"[SquareScheduler] pruned {pruned} old records")


# ------------------------------------------------------------------
# Standalone entry point
# ------------------------------------------------------------------

def _handle_signal(signum, frame):  # pragma: no cover
    raise SystemExit(0)


def run_forever(interval_sec: int = FETCH_INTERVAL_SEC) -> None:  # pragma: no cover
    """作为独立进程运行：python -m scrapers.binance_square.scheduler"""
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    sched = SquareScheduler(interval_sec=interval_sec)
    sched.start(daemon=False)
    try:
        while sched.is_running():
            time.sleep(1)
    except SystemExit:
        sched.stop()


if __name__ == "__main__":  # pragma: no cover
    run_forever()
