"""Shared MySQL bootstrap: connect and auto-create the database.

Single responsibility: the one place that knows how to reach MySQL
(used by events.MysqlEventSink and users.MysqlUserStore).
mysql-connector-python is imported lazily so the rest of the system stays
importable without the package (e.g. when running --storage sqlite).
"""
from __future__ import annotations


def connect_with_database(host: str, port: int, user: str, password: str,
                          database: str):
    try:
        import mysql.connector
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "mysql-connector-python is not installed - "
            "run: pip install mysql-connector-python") from exc

    bootstrap = mysql.connector.connect(host=host, port=port, user=user,
                                        password=password)
    try:
        cur = bootstrap.cursor()
        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{database}` "
                    "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        bootstrap.commit()
        cur.close()
    finally:
        bootstrap.close()

    return mysql.connector.connect(host=host, port=port, user=user,
                                   password=password, database=database)
