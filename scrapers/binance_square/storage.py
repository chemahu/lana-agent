"""SQLite 持久化层：帖子去重与增量存储。

职责：
- 每条帖子以 ``post_id`` 为主键去重。
- 支持按 symbol 或时间窗口查询已存帖子。
- 只写入新帖（增量），避免重复处理。

数据库文件默认位于项目根目录 ``data/square_posts.db``。
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional

_DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "square_posts.db"
)

_DDL = """
CREATE TABLE IF NOT EXISTS posts (
    post_id         TEXT PRIMARY KEY,
    author_id       TEXT NOT NULL,
    author_nickname TEXT,
    author_is_kol   INTEGER DEFAULT 0,
    content         TEXT,
    created_at      TEXT,
    created_at_ms   INTEGER,
    likes           INTEGER DEFAULT 0,
    comments        INTEGER DEFAULT 0,
    shares          INTEGER DEFAULT 0,
    views           INTEGER DEFAULT 0,
    sentiment       TEXT,
    has_trade_widget INTEGER DEFAULT 0,
    symbols         TEXT,
    raw             TEXT,
    inserted_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_posts_created ON posts (created_at_ms);
CREATE INDEX IF NOT EXISTS idx_posts_author  ON posts (author_id);
"""


class PostStorage:
    """线程安全的帖子 SQLite 存储。

    Examples
    --------
    >>> storage = PostStorage()
    >>> new_count = storage.bulk_insert(parsed_posts)
    >>> recent = storage.query_recent(hours=1)
    """

    def __init__(self, db_path: str = _DEFAULT_DB_PATH) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._db_path = db_path
        self._local = threading.local()
        self._init_db()

    # ------------------------------------------------------------------
    # Connection management (one connection per thread)
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        if not getattr(self._local, "conn", None):
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn  # type: ignore[return-value]

    def _init_db(self) -> None:
        conn = self._conn()
        conn.executescript(_DDL)
        conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def insert(self, post: Dict) -> bool:
        """插入单条帖子，若已存在则跳过。返回是否成功插入。"""
        post_id: str = post.get("id", "")
        if not post_id:
            return False
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        symbols_json = json.dumps(post.get("symbols", []), ensure_ascii=False)
        raw_json = json.dumps(post.get("raw", {}), ensure_ascii=False)
        sql = """
        INSERT OR IGNORE INTO posts
            (post_id, author_id, author_nickname, author_is_kol,
             content, created_at, created_at_ms,
             likes, comments, shares, views,
             sentiment, has_trade_widget, symbols, raw, inserted_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """
        conn = self._conn()
        cursor = conn.execute(
            sql,
            (
                post_id,
                post.get("author_id", ""),
                post.get("author_nickname", ""),
                int(bool(post.get("author_is_kol"))),
                post.get("content", ""),
                post.get("created_at"),
                post.get("created_at_ms"),
                int(post.get("likes", 0)),
                int(post.get("comments", 0)),
                int(post.get("shares", 0)),
                int(post.get("views", 0)),
                post.get("sentiment", "neutral"),
                int(bool(post.get("has_trade_widget"))),
                symbols_json,
                raw_json,
                now_iso,
            ),
        )
        conn.commit()
        return cursor.rowcount > 0

    def bulk_insert(self, posts: List[Dict]) -> int:
        """批量插入，返回实际插入（新增）的条数。"""
        return sum(1 for p in posts if self.insert(p))

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def exists(self, post_id: str) -> bool:
        """检查帖子是否已存在。"""
        row = self._conn().execute(
            "SELECT 1 FROM posts WHERE post_id = ?", (post_id,)
        ).fetchone()
        return row is not None

    def query_recent(self, hours: int = 1) -> List[Dict]:
        """查询最近 N 小时内的帖子，按创建时间降序。"""
        cutoff_ms = int(
            (datetime.now(tz=timezone.utc).timestamp() - hours * 3600) * 1000
        )
        rows = self._conn().execute(
            "SELECT * FROM posts WHERE created_at_ms >= ? ORDER BY created_at_ms DESC",
            (cutoff_ms,),
        ).fetchall()
        return [dict(r) for r in rows]

    def query_by_symbol(
        self,
        symbol: str,
        hours: Optional[int] = None,
        limit: int = 200,
    ) -> List[Dict]:
        """查询包含指定 symbol 的帖子（JSON 字符串匹配）。"""
        pattern = f'%"{symbol}"%'
        if hours is not None:
            cutoff_ms = int(
                (datetime.now(tz=timezone.utc).timestamp() - hours * 3600) * 1000
            )
            rows = self._conn().execute(
                "SELECT * FROM posts WHERE symbols LIKE ? AND created_at_ms >= ?"
                " ORDER BY created_at_ms DESC LIMIT ?",
                (pattern, cutoff_ms, limit),
            ).fetchall()
        else:
            rows = self._conn().execute(
                "SELECT * FROM posts WHERE symbols LIKE ?"
                " ORDER BY created_at_ms DESC LIMIT ?",
                (pattern, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        """返回数据库中帖子总数。"""
        row = self._conn().execute("SELECT COUNT(*) FROM posts").fetchone()
        return row[0] if row else 0

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def purge_old(self, keep_days: int = 7) -> int:
        """删除 keep_days 天前的记录，返回删除行数。"""
        cutoff_ms = int(
            (datetime.now(tz=timezone.utc).timestamp() - keep_days * 86400) * 1000
        )
        conn = self._conn()
        cursor = conn.execute(
            "DELETE FROM posts WHERE created_at_ms < ?", (cutoff_ms,)
        )
        conn.commit()
        return cursor.rowcount

    def close(self) -> None:
        """关闭当前线程的数据库连接。"""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
