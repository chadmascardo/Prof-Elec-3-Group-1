"""Event sinks: where detection events get recorded.

Open/Closed: new destinations are added as new EventSink implementations
(MysqlEventSink was added exactly this way, without touching any consumer);
CompositeEventSink fans out to several sinks at once.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from interfaces import Event, EventSink

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT NOT NULL,
    camera    TEXT NOT NULL,
    kind      TEXT NOT NULL,
    names     TEXT,
    course    TEXT,
    snapshot  TEXT,
    detail    TEXT
);
"""


class SqliteEventSink(EventSink):
    def __init__(self, db_path: Path):
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(_SCHEMA)
            self._conn.commit()

    def write(self, event: Event) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO events (ts, camera, kind, names, course, "
                "snapshot, detail) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (event.ts.isoformat(sep=" ", timespec="seconds"),
                 event.camera, event.kind, ", ".join(event.names),
                 event.course, event.snapshot, event.detail))
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


_MYSQL_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id        INT AUTO_INCREMENT PRIMARY KEY,
    ts        DATETIME NOT NULL,
    camera    VARCHAR(64) NOT NULL,
    kind      VARCHAR(32) NOT NULL,
    names     TEXT,
    course    VARCHAR(64),
    snapshot  TEXT,
    detail    TEXT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

_MYSQL_INSERT = ("INSERT INTO events (ts, camera, kind, names, course, "
                 "snapshot, detail) VALUES (%s, %s, %s, %s, %s, %s, %s)")


class MysqlEventSink(EventSink):
    """Stores events in a MySQL server (created lazily: database and table
    are auto-created on first run).

    mysql-connector-python is imported inside __init__ so the rest of the
    system keeps working without the package when --storage sqlite is used.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 3306,
                 user: str = "root", password: str = "",
                 database: str = "face_surveillance"):
        try:
            import mysql.connector
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "mysql-connector-python is not installed - "
                "run: pip install mysql-connector-python") from exc
        self._lock = threading.Lock()

        bootstrap = mysql.connector.connect(
            host=host, port=port, user=user, password=password)
        try:
            cur = bootstrap.cursor()
            cur.execute(f"CREATE DATABASE IF NOT EXISTS `{database}` "
                        "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            bootstrap.commit()
            cur.close()
        finally:
            bootstrap.close()

        self._conn = mysql.connector.connect(
            host=host, port=port, user=user, password=password,
            database=database)
        cur = self._conn.cursor()
        cur.execute(_MYSQL_SCHEMA)
        self._conn.commit()
        cur.close()

    def write(self, event: Event) -> None:
        with self._lock:
            self._conn.ping(reconnect=True, attempts=3, delay=1)
            cur = self._conn.cursor()
            cur.execute(_MYSQL_INSERT,
                        (event.ts, event.camera, event.kind,
                         ", ".join(event.names), event.course,
                         event.snapshot, event.detail))
            self._conn.commit()
            cur.close()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class ConsoleEventSink(EventSink):
    def write(self, event: Event) -> None:
        parts = [f"[{event.ts:%Y-%m-%d %H:%M:%S}]",
                 f"{event.camera:<10}", f"{event.kind:<18}"]
        if event.names:
            parts.append("names=" + ", ".join(event.names))
        parts.append(f"class={event.course or '-'}")
        if event.snapshot:
            parts.append(f"snapshot={event.snapshot}")
        if event.detail:
            parts.append(event.detail)
        print("  ".join(parts), flush=True)


class CompositeEventSink(EventSink):
    def __init__(self, *sinks: EventSink):
        self._sinks = sinks

    def write(self, event: Event) -> None:
        for sink in self._sinks:
            sink.write(event)
