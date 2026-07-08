"""Manage dashboard accounts. Only 'admin' users can see anything.

Usage:
    python manage_users.py add <username>            # role admin (default)
    python manage_users.py add <username> --role viewer
    python manage_users.py passwd <username>
    python manage_users.py remove <username>
    python manage_users.py list

Connection settings come from .env (MYSQL_*), same as main.py.
"""
from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv
    load_dotenv(BASE / ".env")
except ImportError:
    pass

from users import MysqlUserStore


def store_from_env() -> MysqlUserStore:
    env = os.environ.get
    return MysqlUserStore(
        host=env("MYSQL_HOST", "127.0.0.1"),
        port=int(env("MYSQL_PORT", "3306")),
        user=env("MYSQL_USER", "root"),
        password=env("MYSQL_PASSWORD", ""),
        database=env("MYSQL_DATABASE", "face_surveillance"))


def prompt_password() -> str:
    pw = getpass.getpass("password: ")
    if pw != getpass.getpass("confirm : "):
        sys.exit("passwords do not match")
    if len(pw) < 8:
        sys.exit("use at least 8 characters")
    return pw


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    p_add = sub.add_parser("add", help="create an account")
    p_add.add_argument("username")
    p_add.add_argument("--role", choices=("admin", "viewer"), default="admin")
    p_pwd = sub.add_parser("passwd", help="change a password")
    p_pwd.add_argument("username")
    p_rm = sub.add_parser("remove", help="delete an account")
    p_rm.add_argument("username")
    sub.add_parser("list", help="show all accounts")
    args = p.parse_args()

    try:
        store = store_from_env()
    except Exception as exc:
        print(f"could not connect to MySQL: {exc}\n"
              f"check the server and the credentials in .env",
              file=sys.stderr)
        return 1

    try:
        if args.cmd == "add":
            store.add_user(args.username, prompt_password(), args.role)
            print(f"created {args.role} account '{args.username}'")
        elif args.cmd == "passwd":
            if store.set_password(args.username, prompt_password()):
                print(f"password updated for '{args.username}'")
            else:
                sys.exit(f"no such user: {args.username}")
        elif args.cmd == "remove":
            if store.delete_user(args.username):
                print(f"removed '{args.username}'")
            else:
                sys.exit(f"no such user: {args.username}")
        elif args.cmd == "list":
            users = store.list_users()
            if not users:
                print("no accounts yet - create one with: "
                      "python manage_users.py add admin")
            for u in users:
                print(f"{u.username:<20} {u.role}")
    except Exception as exc:
        if "Duplicate entry" in str(exc):
            sys.exit(f"user '{args.username}' already exists")
        raise
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
