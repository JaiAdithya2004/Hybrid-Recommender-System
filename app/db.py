# app/db.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

# SQLite file will be created in project root
DB_FILE = os.getenv("SQLITE_FILE", "recommendations.db")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_FILE}"

# For sqlite need connect_args
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
