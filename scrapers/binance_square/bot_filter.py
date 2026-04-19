"""启发式 bot / 水军过滤器。

过滤规则（满足任意一条即标记为 bot）：
1. 默认用户名（User-xxxxxx 或 Userxxxx 格式，连字符可选）且互动数为零
2. 内容长度 < 5 字符（纯占位帖）
3. 连续重复 token 超过阈值（复读帖）
4. 内容完全是 URL 或标签，无实质文字
5. 发帖时间间隔异常短（同 author 连续帖时间差 < MIN_INTERVAL_SEC）
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List


# 最短帖子字符数（不含空白）
_MIN_CONTENT_LEN = 5
# 认定为重复 token 的占比阈值
_REPEAT_TOKEN_RATIO = 0.6
# 同一作者相邻帖子最小时间间隔（毫秒）
_MIN_INTERVAL_MS = 3_000

_URL_RE = re.compile(
    r"https?://\S+|www\.\S+",
    re.IGNORECASE,
)
_HASHTAG_RE = re.compile(r"#\S+")
_WHITESPACE_RE = re.compile(r"\s+")


def _is_default_name(nickname: str) -> bool:
    return bool(re.fullmatch(r"User-?[A-Fa-f0-9]{4,16}", nickname))


def _token_repeat_ratio(text: str) -> float:
    """计算最高频 token 在全部 token 中的占比，衡量复读程度。"""
    tokens = _WHITESPACE_RE.split(text.strip().lower())
    tokens = [t for t in tokens if t]
    if not tokens:
        return 0.0
    freq = Counter(tokens)
    most_common_count = freq.most_common(1)[0][1]
    return most_common_count / len(tokens)


def _is_substance_empty(content: str) -> bool:
    """去掉 URL 和 hashtag 后是否还有实质文字。"""
    stripped = _URL_RE.sub("", content)
    stripped = _HASHTAG_RE.sub("", stripped)
    stripped = stripped.strip()
    return len(stripped) < _MIN_CONTENT_LEN


def is_bot_post(post: Dict) -> bool:
    """判断单条帖子是否疑似 bot 发出。

    Parameters
    ----------
    post:
        由 ``parser.parse_post`` 返回的标准化帖子字典。

    Returns
    -------
    bool
        True 表示疑似 bot，应从统计中剔除。
    """
    content: str = post.get("content", "")
    likes: int = post.get("likes", 0)
    comments: int = post.get("comments", 0)
    shares: int = post.get("shares", 0)
    views: int = post.get("views", 0)
    author_is_default: bool = post.get("author_is_default", False)

    # 规则 1：默认名 + 零互动
    if author_is_default and (likes + comments + shares + views) == 0:
        return True

    # 规则 2：内容太短
    if len(content.strip()) < _MIN_CONTENT_LEN:
        return True

    # 规则 3：复读 token
    if _token_repeat_ratio(content) >= _REPEAT_TOKEN_RATIO:
        return True

    # 规则 4：内容无实质文字
    if _is_substance_empty(content):
        return True

    return False


def filter_posts(posts: List[Dict]) -> List[Dict]:
    """从帖子列表中移除疑似 bot 发出的帖子，并剔除同一作者过于频繁的帖子。

    Parameters
    ----------
    posts:
        已按 ``created_at_ms`` **升序**排列的标准化帖子列表。

    Returns
    -------
    list
        过滤后的干净帖子列表（保持原始顺序）。
    """
    # 先做单帖规则过滤
    cleaned = [p for p in posts if not is_bot_post(p)]

    # 再做同 author 时间间隔过滤（按时间升序遍历）
    last_ms_by_author: Dict[str, int] = {}
    result: List[Dict] = []
    for p in cleaned:
        aid = p.get("author_id", "")
        ts = p.get("created_at_ms")
        if ts is not None and aid:
            prev = last_ms_by_author.get(aid)
            if prev is not None and (ts - prev) < _MIN_INTERVAL_MS:
                continue  # 发帖太频繁，视作刷屏 bot
            last_ms_by_author[aid] = ts
        result.append(p)

    return result
