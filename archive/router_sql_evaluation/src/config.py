"""
Database configuration module.

Sensitive information (host, password) should be loaded from environment variables.
Non-sensitive information (database, table_name, username, port) can be set here.
"""
import os
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)


def get_db_config() -> Dict[str, str]:
    db_host = os.getenv("DB_HOST")
    db_password = os.getenv("DB_PASSWORD")

    if not db_host:
        raise ValueError("DB_HOST environment variable is required")
    if not db_password:
        raise ValueError("DB_PASSWORD environment variable is required")

    # Non-sensitive fields with defaults
    config = {
        "username": os.getenv("DB_USERNAME", "postgres"),
        "password": db_password,
        "host": db_host,
        "port": int(os.getenv("DB_PORT", "5432")),
        "database": os.getenv("DB_DATABASE", "customer_transaction_db"),
        "table_name": os.getenv("DB_TABLE_NAME", "transactions"),
    }

    return config


def get_DB_CONFIG():
    return get_db_config()
