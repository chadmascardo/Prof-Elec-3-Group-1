"""User accounts for the dashboard (database side only).

Access policy: ONLY users with role 'admin' may see anything.
The web dashboard must call verify_login() and reject any result that is
None or whose role != "admin". A 'viewer' role exists in the schema for
the future but grants nothing today.

Passwords are stored as PBKDF2-SHA256 (Python stdlib, no extra packages):
    pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>

Dashboard usage example (teammate):
    from users import MysqlUserStore
    store = MysqlUserStore(host=..., user=..., password=..., database=...)
    user = store.verify_login(form_username, form_password)
    if user is None or user.role != "admin":
        abort(403)   # not the admin -> sees nothing
"""
from __future__ import annotations

import hashlib
import hmac
import os
import threading
from dataclasses import dataclass
from typing import List, Optional

from db import connect_with_database

_ITERATIONS = 600_000


def hash_password(password: str, iterations: int = _ITERATIONS) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                                 salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt_hex, hash_hex = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                                     bytes.fromhex(salt_hex), int(iters))
        return hmac.compare_digest(digest.hex(), hash_hex)
    except (ValueError, AttributeError):
        return False


@dataclass(frozen=True)
class User:
    username: str
    role: str


_USERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    username      VARCHAR(32) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role          ENUM('admin', 'viewer') NOT NULL DEFAULT 'admin',
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""


class MysqlUserStore:
    """Account storage and login checks (thread-safe)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 3306,
                 user: str = "root", password: str = "",
                 database: str = "face_surveillance"):
        self._lock = threading.Lock()
        self._conn = connect_with_database(host, port, user, password,
                                           database)
        cur = self._conn.cursor()
        cur.execute(_USERS_SCHEMA)
        self._conn.commit()
        cur.close()

    def _cursor(self):
        self._conn.ping(reconnect=True, attempts=3, delay=1)
        return self._conn.cursor()

    def add_user(self, username: str, password: str,
                 role: str = "admin") -> None:
        with self._lock:
            cur = self._cursor()
            cur.execute("INSERT INTO users (username, password_hash, role) "
                        "VALUES (%s, %s, %s)",
                        (username, hash_password(password), role))
            self._conn.commit()
            cur.close()

    def verify_login(self, username: str,
                     password: str) -> Optional[User]:
        """Returns the User on success, None on bad credentials."""
        with self._lock:
            cur = self._cursor()
            cur.execute("SELECT password_hash, role FROM users "
                        "WHERE username = %s", (username,))
            row = cur.fetchone()
            cur.close()
        if row is not None and verify_password(password, row[0]):
            return User(username=username, role=row[1])
        return None

    def set_password(self, username: str, password: str) -> bool:
        with self._lock:
            cur = self._cursor()
            cur.execute("UPDATE users SET password_hash = %s "
                        "WHERE username = %s",
                        (hash_password(password), username))
            changed = cur.rowcount > 0
            self._conn.commit()
            cur.close()
        return changed

    def delete_user(self, username: str) -> bool:
        with self._lock:
            cur = self._cursor()
            cur.execute("DELETE FROM users WHERE username = %s", (username,))
            changed = cur.rowcount > 0
            self._conn.commit()
            cur.close()
        return changed

    def list_users(self) -> List[User]:
        with self._lock:
            cur = self._cursor()
            cur.execute("SELECT username, role FROM users ORDER BY username")
            rows = cur.fetchall()
            cur.close()
        return [User(username=r[0], role=r[1]) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
