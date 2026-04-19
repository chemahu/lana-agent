"""带重试 / 限速 / 代理轮换的币安广场 HTTP 客户端。"""
import time
import asyncio
import threading
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger

from .endpoints import ENDPOINTS

DEFAULT_HEADERS: Dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Content-Type": "application/json",
    "Origin": "https://www.binance.com",
    "Referer": "https://www.binance.com/en/square",
    "clienttype": "web",
    "lang": "en",
}

_RETRIABLE_STATUS = {429, 418, 403}
_MAX_RETRIES = 5
_BASE_BACKOFF = 1.0  # seconds


class _TokenBucket:
    """令牌桶限速器（线程安全）。"""

    def __init__(self, rate: float = 1.0, burst: int = 5) -> None:
        self._rate = rate
        self._burst = burst
        self._tokens = float(burst)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._last = now
            self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
            if self._tokens >= 1.0:
                self._tokens -= 1.0
            else:
                wait = (1.0 - self._tokens) / self._rate
                self._tokens = 0.0
                time.sleep(wait)

    async def async_acquire(self) -> None:
        loop = asyncio.get_event_loop()
        # offload blocking token refill to thread pool to keep event loop clean
        await loop.run_in_executor(None, self.acquire)


class SquareClient:
    """币安广场 HTTP 客户端（同步 + 异步双模）。"""

    def __init__(
        self,
        proxies: Optional[List[str]] = None,
        rate: float = 1.0,
        burst: int = 5,
        timeout: float = 15.0,
    ) -> None:
        self._proxies = proxies or []
        self._proxy_index = 0
        self._proxy_lock = threading.Lock()
        self._bucket = _TokenBucket(rate=rate, burst=burst)
        self._timeout = timeout
        self._sync_client = httpx.Client(
            headers=DEFAULT_HEADERS,
            timeout=timeout,
            follow_redirects=True,
        )

    # ------------------------------------------------------------------
    # Proxy helpers
    # ------------------------------------------------------------------

    def _next_proxy(self) -> Optional[str]:
        if not self._proxies:
            return None
        with self._proxy_lock:
            proxy = self._proxies[self._proxy_index % len(self._proxies)]
            self._proxy_index += 1
        return proxy

    # ------------------------------------------------------------------
    # Low-level POST (sync)
    # ------------------------------------------------------------------

    def _post(self, url: str, payload: Dict) -> Dict:
        last_exc: Optional[Exception] = None
        for attempt in range(1, _MAX_RETRIES + 1):
            self._bucket.acquire()
            proxy = self._next_proxy()
            client = self._sync_client
            if proxy:
                client = httpx.Client(
                    headers=DEFAULT_HEADERS,
                    proxies=proxy,
                    timeout=self._timeout,
                    follow_redirects=True,
                )
            try:
                resp = client.post(url, json=payload)
                if resp.status_code in _RETRIABLE_STATUS:
                    backoff = _BASE_BACKOFF * (2 ** (attempt - 1))
                    logger.warning(
                        f"[SquareClient] HTTP {resp.status_code} on {url}, "
                        f"retry {attempt}/{_MAX_RETRIES} in {backoff:.1f}s"
                    )
                    time.sleep(backoff)
                    continue
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if exc.response.status_code not in _RETRIABLE_STATUS:
                    raise
                backoff = _BASE_BACKOFF * (2 ** (attempt - 1))
                logger.warning(
                    f"[SquareClient] HTTPStatusError {exc.response.status_code}, "
                    f"retry {attempt}/{_MAX_RETRIES} in {backoff:.1f}s"
                )
                time.sleep(backoff)
            except httpx.RequestError as exc:
                last_exc = exc
                backoff = _BASE_BACKOFF * (2 ** (attempt - 1))
                logger.warning(
                    f"[SquareClient] RequestError {exc}, "
                    f"retry {attempt}/{_MAX_RETRIES} in {backoff:.1f}s"
                )
                time.sleep(backoff)
            finally:
                if proxy and client is not self._sync_client:
                    client.close()
        raise RuntimeError(
            f"[SquareClient] all {_MAX_RETRIES} retries failed for {url}"
        ) from last_exc

    # ------------------------------------------------------------------
    # Low-level POST (async)
    # ------------------------------------------------------------------

    async def _async_post(self, url: str, payload: Dict) -> Dict:
        last_exc: Optional[Exception] = None
        for attempt in range(1, _MAX_RETRIES + 1):
            await self._bucket.async_acquire()
            proxy = self._next_proxy()
            async_client_kwargs: Dict[str, Any] = dict(
                headers=DEFAULT_HEADERS,
                timeout=self._timeout,
                follow_redirects=True,
            )
            if proxy:
                async_client_kwargs["proxies"] = proxy
            try:
                async with httpx.AsyncClient(**async_client_kwargs) as client:
                    resp = await client.post(url, json=payload)
                    if resp.status_code in _RETRIABLE_STATUS:
                        backoff = _BASE_BACKOFF * (2 ** (attempt - 1))
                        logger.warning(
                            f"[SquareClient] async HTTP {resp.status_code} on {url}, "
                            f"retry {attempt}/{_MAX_RETRIES} in {backoff:.1f}s"
                        )
                        await asyncio.sleep(backoff)
                        continue
                    resp.raise_for_status()
                    return resp.json()
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if exc.response.status_code not in _RETRIABLE_STATUS:
                    raise
                backoff = _BASE_BACKOFF * (2 ** (attempt - 1))
                logger.warning(
                    f"[SquareClient] async HTTPStatusError {exc.response.status_code}, "
                    f"retry {attempt}/{_MAX_RETRIES} in {backoff:.1f}s"
                )
                await asyncio.sleep(backoff)
            except httpx.RequestError as exc:
                last_exc = exc
                backoff = _BASE_BACKOFF * (2 ** (attempt - 1))
                logger.warning(
                    f"[SquareClient] async RequestError {exc}, "
                    f"retry {attempt}/{_MAX_RETRIES} in {backoff:.1f}s"
                )
                await asyncio.sleep(backoff)
        raise RuntimeError(
            f"[SquareClient] all {_MAX_RETRIES} async retries failed for {url}"
        ) from last_exc

    # ------------------------------------------------------------------
    # Response validation
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_data(body: Dict) -> Any:
        success = body.get("success", False)
        code = str(body.get("code", ""))
        if success or code in ("000000", "0"):
            return body.get("data")
        raise ValueError(
            f"[SquareClient] API error: code={code!r}, "
            f"message={body.get('message')!r}"
        )

    # ------------------------------------------------------------------
    # High-level semantic methods (sync)
    # ------------------------------------------------------------------

    def fetch_feed_list(self, page: int = 1, page_size: int = 20) -> Any:
        body = self._post(
            ENDPOINTS["feed_list"],
            {"pageNo": page, "pageSize": page_size},
        )
        return self._extract_data(body)

    def fetch_topic_feed(
        self, topic_id: str, page: int = 1, page_size: int = 20
    ) -> Any:
        body = self._post(
            ENDPOINTS["topic_feed"],
            {"topicId": topic_id, "pageNo": page, "pageSize": page_size},
        )
        return self._extract_data(body)

    def search(self, keyword: str, page: int = 1, page_size: int = 20) -> Any:
        body = self._post(
            ENDPOINTS["search"],
            {"keyword": keyword, "pageNo": page, "pageSize": page_size},
        )
        return self._extract_data(body)

    def fetch_user_profile(self, user_id: str) -> Any:
        body = self._post(
            ENDPOINTS["user_profile"],
            {"userId": user_id},
        )
        return self._extract_data(body)

    def fetch_hot_topics(self) -> Any:
        body = self._post(ENDPOINTS["hot_topics"], {})
        return self._extract_data(body)

    # ------------------------------------------------------------------
    # High-level semantic methods (async)
    # ------------------------------------------------------------------

    async def async_fetch_feed_list(self, page: int = 1, page_size: int = 20) -> Any:
        body = await self._async_post(
            ENDPOINTS["feed_list"],
            {"pageNo": page, "pageSize": page_size},
        )
        return self._extract_data(body)

    async def async_fetch_topic_feed(
        self, topic_id: str, page: int = 1, page_size: int = 20
    ) -> Any:
        body = await self._async_post(
            ENDPOINTS["topic_feed"],
            {"topicId": topic_id, "pageNo": page, "pageSize": page_size},
        )
        return self._extract_data(body)

    async def async_search(
        self, keyword: str, page: int = 1, page_size: int = 20
    ) -> Any:
        body = await self._async_post(
            ENDPOINTS["search"],
            {"keyword": keyword, "pageNo": page, "pageSize": page_size},
        )
        return self._extract_data(body)

    async def async_fetch_user_profile(self, user_id: str) -> Any:
        body = await self._async_post(
            ENDPOINTS["user_profile"],
            {"userId": user_id},
        )
        return self._extract_data(body)

    async def async_fetch_hot_topics(self) -> Any:
        body = await self._async_post(ENDPOINTS["hot_topics"], {})
        return self._extract_data(body)

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._sync_client.close()

    def __enter__(self) -> "SquareClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
