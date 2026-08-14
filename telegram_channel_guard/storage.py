import asyncio
import sqlite3
from pathlib import Path
from typing import Optional


class Storage:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self._lock = asyncio.Lock()

    async def init(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with self._lock:
            await asyncio.to_thread(self._init_sync)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_sync(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS blocked_channels (
                    group_id   INTEGER NOT NULL,
                    channel_id INTEGER NOT NULL,
                    title      TEXT NOT NULL,
                    username   TEXT,
                    added_by   INTEGER,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (group_id, channel_id)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_blocked_group
                ON blocked_channels(group_id)
                """
            )
            conn.commit()

    async def block_channel(
        self,
        group_id: int,
        channel_id: int,
        title: str,
        username: Optional[str],
        added_by: Optional[int],
    ) -> bool:
        async with self._lock:
            return await asyncio.to_thread(
                self._block_channel_sync,
                group_id,
                channel_id,
                title,
                username,
                added_by,
            )

    def _block_channel_sync(
        self,
        group_id: int,
        channel_id: int,
        title: str,
        username: Optional[str],
        added_by: Optional[int],
    ) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO blocked_channels
                    (group_id, channel_id, title, username, added_by)
                VALUES (?, ?, ?, ?, ?)
                """,
                (group_id, channel_id, title, username, added_by),
            )
            conn.commit()
            return cur.rowcount > 0

    async def unblock_channel(self, group_id: int, channel_id: int) -> bool:
        async with self._lock:
            return await asyncio.to_thread(
                self._unblock_channel_sync, group_id, channel_id
            )

    def _unblock_channel_sync(self, group_id: int, channel_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                """
                DELETE FROM blocked_channels
                WHERE group_id = ? AND channel_id = ?
                """,
                (group_id, channel_id),
            )
            conn.commit()
            return cur.rowcount > 0

    async def is_blocked(self, group_id: int, channel_id: int) -> bool:
        async with self._lock:
            return await asyncio.to_thread(
                self._is_blocked_sync, group_id, channel_id
            )

    def _is_blocked_sync(self, group_id: int, channel_id: int) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM blocked_channels
                WHERE group_id = ? AND channel_id = ?
                LIMIT 1
                """,
                (group_id, channel_id),
            ).fetchone()
            return row is not None

    async def list_blocked(self, group_id: int):
        async with self._lock:
            return await asyncio.to_thread(self._list_blocked_sync, group_id)

    def _list_blocked_sync(self, group_id: int):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT channel_id, title, username, added_by, created_at
                FROM blocked_channels
                WHERE group_id = ?
                ORDER BY created_at ASC
                """,
                (group_id,),
            ).fetchall()
            return [dict(row) for row in rows]
