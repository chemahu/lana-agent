"""Lana 启发式 Bot 过滤器。

根据以下规则将疑似机器人帖子标记出来，供聚合器在统计时剔除：
1. 作者昵称为默认格式（User-xxxxxxxx）
2. 内容极短（< 10 个字符）
3. 帖子内容与已知 bot 模板高度匹配（正则）
4. 单个作者在滑动窗口内连续发帖超过阈值（高频刷屏）
"""
from __future__ import annotations

import re
import time
from collections import defaultdict
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------
# 可配置常量
# --------------------------------------------------------------------------

#: 默认昵称正则（如 "User-3f2a" 或 "User3f2a1b"）
_DEFAULT_NICKNAME_RE = re.compile(r"^User-?[A-Fa-f0-9]{4,16}$", re.IGNORECASE)

#: 帖子内容最低字符数阈值（低于此视为无效内容）
MIN_CONTENT_LENGTH: int = 10

#: 单作者在 `RATE_WINDOW_SEC` 秒内最多允许的帖子数
MAX_POSTS_IN_WINDOW: int = 10

#: 滑动窗口时长（秒）
RATE_WINDOW_SEC: int = 3600

#: 已知 bot 内容模板片段（正则，任意一条命中即视为 bot）
_BOT_PATTERNS: List[re.Pattern] = [
    re.compile(r"(follow\s+me|follow\s+back|dm\s+me\s+for)", re.IGNORECASE),
    re.compile(r"(free\s+signal|vip\s+signal|join\s+my\s+channel)", re.IGNORECASE),
    re.compile(r"(t\.me/|telegram\.me/)", re.IGNORECASE),
    re.compile(r"(click\s+here|visit\s+now|limited\s+offer)", re.IGNORECASE),
    re.compile(r"(\d{3,}\s*%\s*(profit|return|gain))", re.IGNORECASE),
]


# --------------------------------------------------------------------------
# 核心类
# --------------------------------------------------------------------------


class BotFilter:
    """有状态的 bot 过滤器，需在聚合器中实例化并复用。

    Examples
    --------
    >>> bf = BotFilter()
    >>> clean_posts = [p for p in raw_posts if not bf.is_bot(p)]
    """

    def __init__(
        self,
        max_posts_in_window: int = MAX_POSTS_IN_WINDOW,
        rate_window_sec: int = RATE_WINDOW_SEC,
    ) -> None:
        self._max_posts = max_posts_in_window
        self._window = rate_window_sec
        # author_id -> list of post timestamps (ms)
        self._author_ts: Dict[str, List[int]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def is_bot(self, post: Dict) -> bool:
        """判断单条帖子是否来自 bot。

        Parameters
        ----------
        post:
            由 ``parser.parse_post`` 返回的标准化帖子字典。

        Returns
        -------
        bool
            ``True`` 表示疑似 bot，``False`` 表示正常帖子。
        """
        return (
            self._is_default_author(post)
            or self._is_too_short(post)
            or self._matches_bot_template(post)
            or self._is_high_frequency(post)
        )

    def filter(self, posts: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """批量过滤，返回 ``(clean, bots)`` 两个列表。"""
        clean: List[Dict] = []
        bots: List[Dict] = []
        for p in posts:
            (bots if self.is_bot(p) else clean).append(p)
        return clean, bots

    def reset(self) -> None:
        """清除速率统计缓存（可在每轮抓取结束后调用）。"""
        self._author_ts.clear()

    # ------------------------------------------------------------------
    # Private rule implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _is_default_author(post: Dict) -> bool:
        nickname: str = post.get("author_nickname", "")
        return bool(_DEFAULT_NICKNAME_RE.match(nickname))

    @staticmethod
    def _is_too_short(post: Dict) -> bool:
        return len(post.get("content", "")) < MIN_CONTENT_LENGTH

    @staticmethod
    def _matches_bot_template(post: Dict) -> bool:
        content: str = post.get("content", "")
        return any(pat.search(content) for pat in _BOT_PATTERNS)

    def _is_high_frequency(self, post: Dict) -> bool:
        author_id: str = post.get("author_id", "")
        if not author_id:
            return False
        now_ms: int = post.get("created_at_ms") or int(time.time() * 1000)
        cutoff_ms: int = now_ms - self._window * 1000
        ts_list = self._author_ts[author_id]
        # keep only timestamps within the sliding window
        ts_list[:] = [t for t in ts_list if t >= cutoff_ms]
        ts_list.append(now_ms)
        return len(ts_list) > self._max_posts
