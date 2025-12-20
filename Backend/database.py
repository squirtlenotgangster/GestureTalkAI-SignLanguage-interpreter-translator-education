from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# This creates the file 'sql_app.db' in the same directory
SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"

# connect_args is needed only for SQLite
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()