from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from langchain_community.utilities import SQLDatabase


def get_sql_db(
    username: str,
    password: str,
    host: str,
    port: int,
    database: str,
    drivername: str = "postgresql+psycopg2",
    echo: bool = False):
    url_object = URL.create(
        drivername=drivername,
        username=username,
        password=password,
        host=host,
        port=port,
        database=database,
    )

    engine = create_engine(url_object, echo=echo)

    db = SQLDatabase(engine=engine)
    return db



