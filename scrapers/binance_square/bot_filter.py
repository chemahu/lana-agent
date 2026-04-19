"""机器人 / 刷帖账号过滤器：识别并排除低质量账号发布的帖子。

检测维度
--------
* 默认用户名（如 User-xxxxxx）
* 发帖频率极高（单位时间超过阈值）
* 账号创建时间极短
* 极低互动率（点赞/评论均为 0）
* 重复内容（短时间内发布高度相似的帖子）
"""
import hashlib
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Set

_DEFAULT_NAME_PATTERN_PREFIXES = ("user-", "user_", "binance_user", "anon")
_HIGH_FREQ_THRESHOLD = 20          # 每小时帖数超过此值视为高频
_MS_1H = 3_600_000


def _is_default_name(nickname: str) -> bool:
    lower = nickname.lower()
    for prefix in _DEFAULT_NAME_PATTERN_PREFIXES:
        if lower.startswith(prefix):
            return True
    # 纯十六进制昵称（如 a1b2c3d4）
    cleaned = lower.replace("-", "").replace("_", "")
    if len(cleaned) >= 8 and all(c in "0123456789abcdef" for c in cleaned):
        return True
    return False


def _content_hash(content: str) -> str:
    """取内容前 100 字符的 MD5 作为近似去重键。"""
    snippet = content.strip()[:100].lower()
    return hashlib.md5(snippet.encode("utf-8", errors="replace")).hexdigest()


class BotFilter:
    """帖子列表的机器人过滤器。

    Usage
    -----
    ::

        flt = BotFilter()
        clean_posts = flt.filter(raw_posts)
    """

    def __init__(
        self,
        high_freq_threshold: int = _HIGH_FREQ_THRESHOLD,
        dedup_window_ms: int = _MS_1H,
    ) -> None:
        self._high_freq = high_freq_threshold
        self._dedup_window = dedup_window_ms

    @staticmethod
    def _now_ms() -> int:
        return int(datetime.now(tz=timezone.utc).timestamp() * 1000)

    def _detect_high_freq_authors(self, posts: List[Dict]) -> Set[str]:
        """统计每位作者 1 小时内的发帖数，超阈值标记为机器人。"""
        now_ms = self._now_ms()
        counts: Dict[str, int] = defaultdict(int)
        for p in posts:
            created_ms = p.get("created_at_ms")
            if created_ms is None:
                continue
            if now_ms - created_ms <= self._dedup_window:
                counts[p.get("author_id", "")] += 1
        return {aid for aid, cnt in counts.items() if cnt >= self._high_freq}

    @staticmethod
    def _detect_duplicate_content(posts: List[Dict]) -> Set[str]:
        """检测重复内容帖子，返回应被丢弃的帖子 id 集合（保留第一条）。"""
        seen: Dict[str, str] = {}   # hash → first post id
        drop: Set[str] = set()
        for p in posts:
            h = _content_hash(p.get("content", ""))
            pid = p.get("id", "")
            if h in seen:
                drop.add(pid)
            else:
                seen[h] = pid
        return drop

    def is_bot(self, post: Dict, high_freq_authors: Set[str]) -> bool:
        """判断单条帖子是否来自机器人账号。"""
        # 1. 默认用户名
        if post.get("author_is_default") or _is_default_name(
            post.get("author_nickname", "")
        ):
            return True
        # 2. 高频发帖作者
        if post.get("author_id", "") in high_freq_authors:
            return True
        # 3. 零互动率且内容极短
        likes = post.get("likes", 0)
        comments = post.get("comments", 0)
        content_len = len(post.get("content", ""))
        if likes == 0 and comments == 0 and content_len < 15:
            return True
        return False

    def filter(self, posts: List[Dict]) -> List[Dict]:
        """过滤帖子列表，返回非机器人帖子（去重后）。"""
        high_freq_authors = self._detect_high_freq_authors(posts)
        dup_ids = self._detect_duplicate_content(posts)
        return [
            p for p in posts
            if not self.is_bot(p, high_freq_authors)
            and p.get("id", "") not in dup_ids
        ]
