"""帖子持久化存储：将解析后的帖子写入本地 SQLite，供离线分析和去重使用。"""
import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

_DEFAULT_DB_PATH = Path("data/binance_square.db")

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS posts (
    id              TEXT PRIMARY KEY,
    author_id       TEXT,
    author_nickname TEXT,
    author_is_kol   INTEGER DEFAULT 0,
    content         TEXT,
    created_at      TEXT,
    created_at_ms   INTEGER,
    likes           INTEGER DEFAULT 0,
    comments        INTEGER DEFAULT 0,
    shares          INTEGER DEFAULT 0,
    views           INTEGER DEFAULT 0,
    symbols         TEXT,
    sentiment       TEXT,
    has_trade_widget INTEGER DEFAULT 0,
    fetched_at_ms   INTEGER
);
CREATE INDEX IF NOT EXISTS idx_posts_created ON posts(created_at_ms);
CREATE INDEX IF NOT EXISTS idx_posts_author  ON posts(author_id);
"""


class PostStorage:
    """线程安全的 SQLite 帖子存储。

    Parameters
    ----------
    db_path:
        SQLite 文件路径（不存在时自动创建）。
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(_CREATE_TABLE_SQL)
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_posts(self, posts: List[Dict]) -> int:
        """批量保存帖子（已存在则忽略）。返回实际插入的条数。"""
        import json
        import time

        now_ms = int(time.time() * 1000)
        inserted = 0

        with self._lock:
            conn = self._connect()
            try:
                for p in posts:
                    post_id = p.get("id", "")
                    if not post_id:
                        continue
                    symbols_str = json.dumps(p.get("symbols", []))
                    try:
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO posts
                                (id, author_id, author_nickname, author_is_kol,
                                 content, created_at, created_at_ms,
                                 likes, comments, shares, views,
                                 symbols, sentiment, has_trade_widget, fetched_at_ms)
                            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                            """,
                            (
                                post_id,
                                p.get("author_id", ""),
                                p.get("author_nickname", ""),
                                int(p.get("author_is_kol", False)),
                                p.get("content", ""),
                                p.get("created_at"),
                                p.get("created_at_ms"),
                                p.get("likes", 0),
                                p.get("comments", 0),
                                p.get("shares", 0),
                                p.get("views", 0),
                                symbols_str,
                                p.get("sentiment", "neutral"),
                                int(p.get("has_trade_widget", False)),
                                now_ms,
                            ),
                        )
                        if conn.execute("SELECT changes()").fetchone()[0]:
                            inserted += 1
                    except sqlite3.Error as exc:
                        logger.warning(f"[PostStorage] insert error for id={post_id}: {exc}")
                conn.commit()
            finally:
                conn.close()

        logger.debug(f"[PostStorage] saved {inserted}/{len(posts)} new posts")
        return inserted

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_recent_posts(
        self,
        symbol: Optional[str] = None,
        limit: int = 200,
        since_ms: Optional[int] = None,
    ) -> List[Dict]:
        """查询最近帖子。可按 symbol 过滤（文本包含匹配）。"""
        import json

        conditions = []
        params: list = []
        if since_ms is not None:
            conditions.append("created_at_ms >= ?")
            params.append(since_ms)
        if symbol:
            conditions.append("(symbols LIKE ? OR content LIKE ?)")
            like = f"%{symbol}%"
            params.extend([like, like])

        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = f"""
            SELECT * FROM posts
            {where_clause}
            ORDER BY created_at_ms DESC
            LIMIT ?
        """
        params.append(limit)

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(sql, params).fetchall()
                result = []
                for row in rows:
                    d = dict(row)
                    try:
                        d["symbols"] = json.loads(d.get("symbols") or "[]")
                    except (ValueError, TypeError):
                        d["symbols"] = []
                    d["author_is_kol"] = bool(d.get("author_is_kol"))
                    d["has_trade_widget"] = bool(d.get("has_trade_widget"))
                    result.append(d)
                return result
            finally:
                conn.close()

    def post_count(self) -> int:
        """返回数据库中的帖子总数。"""
        with self._lock:
            conn = self._connect()
            try:
                return conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
            finally:
                conn.close()
