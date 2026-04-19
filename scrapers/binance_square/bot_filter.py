"""启发式 Bot 过滤器：从帖子列表中剔除可疑机器人账号的帖子。"""
import logging
import re
from typing import Dict, List, Set

# ------------------------------------------------------------------
# Heuristic constants
# ------------------------------------------------------------------

# Patterns that strongly indicate auto-generated / default usernames
_DEFAULT_NAME_PATTERNS = [
    re.compile(r"^User[-_]?[A-Fa-f0-9]{4,16}$"),
    re.compile(r"^user\d{6,}$", re.IGNORECASE),
    re.compile(r"^[Bb]inance\w{6,}$"),
    re.compile(r"^[A-Za-z]{2,4}\d{8,}$"),
]

# Minimum engagement to be considered non-bot
_MIN_LIKES_FOR_UNVERIFIED = 0  # allow zero-engagement real users
_BOT_ENGAGEMENT_THRESHOLD = 50  # if all engagement == 0 AND post count high → suspicious

# Content repetition: if same content hash seen too many times, mark as bot
_MAX_DUPLICATE_CONTENT = 3


def _is_default_username(nickname: str) -> bool:
    """判断昵称是否符合自动生成的模式。"""
    for pattern in _DEFAULT_NAME_PATTERNS:
        if pattern.match(nickname):
            return True
    return False


def _content_fingerprint(content: str) -> str:
    """返回内容的简化指纹（去除空白和大小写）。"""
    return re.sub(r"\s+", "", content).lower()[:120]


class BotFilter:
    """从解析后的帖子列表中剔除机器人帖子。

    过滤规则（满足其中一条即被丢弃）
    ----------------------------------
    1. 作者昵称匹配默认用户名模式，且帖子零互动（likes+comments+shares == 0）
    2. 帖子内容完全重复超过 _MAX_DUPLICATE_CONTENT 次
    3. 帖子内容极短（< 5 个字符）且零互动
    """

    def __init__(self) -> None:
        self._seen_fingerprints: dict = {}  # fingerprint → count
        self._known_bot_ids: Set[str] = set()

    def reset(self) -> None:
        """清除批次内的去重状态（每次抓取开始前调用）。"""
        self._seen_fingerprints.clear()

    def _is_bot(self, post: Dict) -> bool:
        author_id = post.get("author_id", "")
        if author_id in self._known_bot_ids:
            return True

        nickname = post.get("author_nickname", "")
        is_default = post.get("author_is_default", False) or _is_default_username(nickname)

        likes = post.get("likes", 0)
        comments = post.get("comments", 0)
        shares = post.get("shares", 0)
        engagement = likes + comments + shares

        content = post.get("content", "")

        # Rule 1: default username + zero engagement
        if is_default and engagement == 0:
            return True

        # Rule 2: content duplicate spam
        fp = _content_fingerprint(content)
        if fp:
            count = self._seen_fingerprints.get(fp, 0) + 1
            self._seen_fingerprints[fp] = count
            if count > _MAX_DUPLICATE_CONTENT:
                return True

        # Rule 3: trivially short content + zero engagement
        if len(content.strip()) < 5 and engagement == 0:
            return True

        return False

    def filter(self, posts: List[Dict]) -> List[Dict]:
        """过滤帖子列表，返回非机器人帖子。"""
        clean: List[Dict] = []
        bot_count = 0
        for post in posts:
            if self._is_bot(post):
                bot_count += 1
            else:
                clean.append(post)
        if bot_count:
            logging.getLogger(__name__).debug(
                f"[BotFilter] removed {bot_count}/{len(posts)} suspected bot posts"
            )
        return clean

    def mark_bot(self, author_id: str) -> None:
        """手动标记某 author_id 为已知机器人。"""
        self._known_bot_ids.add(author_id)
