"""币安广场常驻拉取进程。

每隔 ``POLL_INTERVAL_SEC``（默认 120 秒）拉取一轮所有关注 symbol 的帖子，
经 BotFilter 过滤后写入 PostStorage。

可直接运行::

    python -m scrapers.binance_square.scheduler

也可在主进程中通过 ``SquareScheduler`` 类嵌入。
"""
from __future__ import annotations

import signal
import time
from typing import List, Optional

from loguru import logger

from .aggregator import SquareAggregator
from .bot_filter import BotFilter
from .client import SquareClient
from .storage import PostStorage

#: 默认轮询间隔（秒）
POLL_INTERVAL_SEC: int = 120

#: 每轮拉取的 symbol 列表（可由外部注入或读取配置）
DEFAULT_SYMBOLS: List[str] = []


class SquareScheduler:
    """常驻进程调度器：定期从币安广场拉取帖子并存库。

    Parameters
    ----------
    symbols:
        需要追踪的交易对列表，例如 ``["BTC", "ETH", "SOL"]``。
        若为空则仅拉取广场首页热门信息流。
    interval:
        轮询间隔（秒），默认 120。
    storage:
        ``PostStorage`` 实例，默认自动创建。
    client:
        ``SquareClient`` 实例，默认自动创建。
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        interval: int = POLL_INTERVAL_SEC,
        storage: Optional[PostStorage] = None,
        client: Optional[SquareClient] = None,
    ) -> None:
        self._symbols: List[str] = list(symbols or DEFAULT_SYMBOLS)
        self._interval = interval
        self._storage = storage or PostStorage()
        self._client = client or SquareClient()
        self._aggregator = SquareAggregator(client=self._client)
        self._bot_filter = BotFilter()
        self._running = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add_symbol(self, symbol: str) -> None:
        """动态添加追踪 symbol。"""
        sym = symbol.upper()
        if sym not in self._symbols:
            self._symbols.append(sym)

    def remove_symbol(self, symbol: str) -> None:
        """动态移除追踪 symbol。"""
        sym = symbol.upper()
        if sym in self._symbols:
            self._symbols.remove(sym)

    def run_once(self) -> int:
        """执行单轮拉取，返回本轮新增帖子总数。"""
        total_new = 0
        symbols = list(self._symbols) if self._symbols else ["BTC"]
        for symbol in symbols:
            try:
                posts = self._aggregator._fetch_posts_for_symbol(symbol)
                clean, bots = self._bot_filter.filter(posts)
                logger.debug(
                    f"[Scheduler] {symbol}: {len(posts)} fetched, "
                    f"{len(bots)} bots filtered, {len(clean)} clean"
                )
                new_count = self._storage.bulk_insert(clean)
                total_new += new_count
                logger.info(
                    f"[Scheduler] {symbol}: +{new_count} new posts stored"
                )
            except Exception as exc:
                logger.warning(f"[Scheduler] error processing {symbol}: {exc}")
        self._bot_filter.reset()
        return total_new

    def start(self) -> None:
        """阻塞运行，直到接收到 SIGINT/SIGTERM 或调用 ``stop()``。"""
        self._running = True
        self._register_signals()
        logger.info(
            f"[Scheduler] started — symbols={self._symbols}, "
            f"interval={self._interval}s"
        )
        while self._running:
            try:
                new = self.run_once()
                logger.info(f"[Scheduler] round done, total new posts: {new}")
            except Exception as exc:
                logger.error(f"[Scheduler] unexpected error in run loop: {exc}")
            for _ in range(self._interval):
                if not self._running:
                    break
                time.sleep(1)
        logger.info("[Scheduler] stopped.")

    def stop(self) -> None:
        """停止运行循环。"""
        self._running = False

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def _register_signals(self) -> None:
        def _handler(signum, frame):  # noqa: ANN001
            logger.info(f"[Scheduler] received signal {signum}, stopping…")
            self.stop()

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Binance Square 常驻拉取进程")
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=[],
        help="追踪的 symbol 列表，例如 BTC ETH SOL",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=POLL_INTERVAL_SEC,
        help=f"轮询间隔（秒），默认 {POLL_INTERVAL_SEC}",
    )
    args = parser.parse_args()

    scheduler = SquareScheduler(
        symbols=args.symbols or DEFAULT_SYMBOLS,
        interval=args.interval,
    )
    scheduler.start()
