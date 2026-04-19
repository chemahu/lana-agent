"""SQLite 持久化存储：帖子去重 + 增量写入。

Schema（单表 ``posts``）：
  id TEXT PRIMARY KEY       — 帖子唯一 ID（来自 parser.parse_post["id"]）
  symbol TEXT               — 关联代币（搜索时的 keyword）
  author_id TEXT
  content TEXT
  sentiment TEXT
  created_at TEXT           — ISO-8601 UTC 字符串
  created_at_ms INTEGER
  likes INTEGER
  comments INTEGER
  shares INTEGER
  views INTEGER
  author_is_kol INTEGER     — 0/1
  has_trade_widget INTEGER  — 0/1
  fetched_at TEXT           — 本次抓取时间（ISO-8601 UTC）
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

_DEFAULT_DB_PATH = Path("data/square_posts.db")


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class PostStorage:
    """线程不安全的轻量 SQLite 存储（单进程使用）。"""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS posts (
                id               TEXT PRIMARY KEY,
                symbol           TEXT NOT NULL,
                author_id        TEXT,
                content          TEXT,
                sentiment        TEXT,
                created_at       TEXT,
                created_at_ms    INTEGER,
                likes            INTEGER DEFAULT 0,
                comments         INTEGER DEFAULT 0,
                shares           INTEGER DEFAULT 0,
                views            INTEGER DEFAULT 0,
                author_is_kol    INTEGER DEFAULT 0,
                has_trade_widget INTEGER DEFAULT 0,
                fetched_at       TEXT
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_posts_symbol ON posts (symbol)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_posts_created_ms "
            "ON posts (created_at_ms)"
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def save_posts(self, posts: List[Dict], symbol: str) -> int:
        """批量写入帖子，已存在的跳过（INSERT OR IGNORE）。

        Parameters
        ----------
        posts:
            ``parser.parse_post`` 返回的标准化帖子列表。
        symbol:
            搜索时使用的 keyword（如 "BTC"）。

        Returns
        -------
        int
            实际新增的行数。
        """
        fetched_at = _now_iso()
        rows = [
            (
                p.get("id") or "",
                symbol,
                p.get("author_id") or "",
                p.get("content") or "",
                p.get("sentiment") or "neutral",
                p.get("created_at"),
                p.get("created_at_ms"),
                p.get("likes", 0),
                p.get("comments", 0),
                p.get("shares", 0),
                p.get("views", 0),
                1 if p.get("author_is_kol") else 0,
                1 if p.get("has_trade_widget") else 0,
                fetched_at,
            )
            for p in posts
            if p.get("id")  # 跳过缺少 id 的帖子
        ]
        if not rows:
            return 0

        cur = self._conn.executemany(
            """
            INSERT OR IGNORE INTO posts
              (id, symbol, author_id, content, sentiment,
               created_at, created_at_ms,
               likes, comments, shares, views,
               author_is_kol, has_trade_widget, fetched_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            rows,
        )
        self._conn.commit()
        return cur.rowcount

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def get_posts(
        self,
        symbol: str,
        since_ms: Optional[int] = None,
        limit: int = 500,
    ) -> List[Dict]:
        """查询某 symbol 的帖子。

        Parameters
        ----------
        symbol:
            代币符号关键字。
        since_ms:
            仅返回 ``created_at_ms >= since_ms`` 的帖子；None 表示不过滤。
        limit:
            最多返回条数。

        Returns
        -------
        list[dict]
        """
        if since_ms is not None:
            rows = self._conn.execute(
                "SELECT * FROM posts WHERE symbol=? AND created_at_ms>=? "
                "ORDER BY created_at_ms DESC LIMIT ?",
                (symbol, since_ms, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM posts WHERE symbol=? "
                "ORDER BY created_at_ms DESC LIMIT ?",
                (symbol, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def known_ids(self, symbol: str) -> set:
        """返回已存储的帖子 ID 集合（用于增量去重）。"""
        rows = self._conn.execute(
            "SELECT id FROM posts WHERE symbol=?", (symbol,)
        ).fetchall()
        return {r["id"] for r in rows}

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def prune_old(self, keep_days: int = 7) -> int:
        """删除超过 keep_days 天的旧记录，返回删除行数。"""
        cutoff_ms = (
            int(datetime.now(tz=timezone.utc).timestamp()) - keep_days * 86400
        ) * 1000
        cur = self._conn.execute(
            "DELETE FROM posts WHERE created_at_ms IS NOT NULL AND created_at_ms<?",
            (cutoff_ms,),
        )
        self._conn.commit()
        return cur.rowcount

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "PostStorage":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
