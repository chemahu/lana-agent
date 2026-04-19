"""SQLite 持久化存储：帖子去重 + 增量写入。"""
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple


_DEFAULT_DB_PATH = Path("data/square_posts.db")

_DDL = """
CREATE TABLE IF NOT EXISTS posts (
    id            TEXT PRIMARY KEY,
    symbol        TEXT,
    author_id     TEXT,
    content       TEXT,
    sentiment     TEXT,
    created_at_ms INTEGER,
    likes         INTEGER DEFAULT 0,
    comments      INTEGER DEFAULT 0,
    shares        INTEGER DEFAULT 0,
    views         INTEGER DEFAULT 0,
    kol_mentioned INTEGER DEFAULT 0,
    has_trade_widget INTEGER DEFAULT 0,
    inserted_at   INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_posts_symbol      ON posts(symbol);
CREATE INDEX IF NOT EXISTS idx_posts_created_ms  ON posts(created_at_ms);
CREATE INDEX IF NOT EXISTS idx_posts_inserted_at ON posts(inserted_at);
"""


class PostStorage:
    """线程安全的 SQLite 帖子存储，支持去重写入和批量查询。"""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    # ------------------------------------------------------------------
    # DB initialisation
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_DDL)

    # ------------------------------------------------------------------
    # Context manager for thread-safe writes
    # ------------------------------------------------------------------

    @contextmanager
    def _write_conn(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = self._connect()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def save_post(self, post: Dict, symbol: str = "") -> bool:
        """插入单条帖子；若已存在（同 id）则忽略。返回是否为新插入。"""
        post_id = str(post.get("id", ""))
        if not post_id:
            return False

        now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        row: Tuple = (
            post_id,
            symbol or "",
            str(post.get("author_id", "")),
            str(post.get("content", "")),
            str(post.get("sentiment", "neutral")),
            post.get("created_at_ms"),
            int(post.get("likes", 0)),
            int(post.get("comments", 0)),
            int(post.get("shares", 0)),
            int(post.get("views", 0)),
            int(bool(post.get("author_is_kol", False))),
            int(bool(post.get("has_trade_widget", False))),
            now_ms,
        )
        with self._write_conn() as conn:
            cursor = conn.execute(
                """INSERT OR IGNORE INTO posts
                   (id, symbol, author_id, content, sentiment,
                    created_at_ms, likes, comments, shares, views,
                    kol_mentioned, has_trade_widget, inserted_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                row,
            )
            return cursor.rowcount > 0

    def save_posts(self, posts: List[Dict], symbol: str = "") -> int:
        """批量插入帖子，返回实际新增条数。"""
        new_count = 0
        for post in posts:
            if self.save_post(post, symbol=symbol):
                new_count += 1
        return new_count

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_posts(
        self,
        symbol: str = "",
        since_ms: Optional[int] = None,
        limit: int = 200,
    ) -> List[Dict]:
        """查询帖子列表，可按 symbol 和时间过滤。"""
        query = "SELECT * FROM posts"
        conditions = []
        params: List = []
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if since_ms is not None:
            conditions.append("created_at_ms >= ?")
            params.append(since_ms)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at_ms DESC LIMIT ?"
        params.append(limit)

        conn = self._connect()
        try:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def exists(self, post_id: str) -> bool:
        """判断帖子是否已在库中。"""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT 1 FROM posts WHERE id = ? LIMIT 1", (post_id,)
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def count(self, symbol: str = "") -> int:
        """返回存储总条数（可按 symbol 过滤）。"""
        conn = self._connect()
        try:
            if symbol:
                row = conn.execute(
                    "SELECT COUNT(*) FROM posts WHERE symbol = ?", (symbol,)
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM posts").fetchone()
            return row[0] if row else 0
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def prune_old(self, older_than_ms: int) -> int:
        """删除 created_at_ms < older_than_ms 的旧记录，返回删除条数。"""
        with self._write_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM posts WHERE created_at_ms < ? AND created_at_ms IS NOT NULL",
                (older_than_ms,),
            )
            return cursor.rowcount
