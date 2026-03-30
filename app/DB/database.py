import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

load_dotenv()
app_env = os.getenv("APP_ENV", "development").strip().lower()
database_url = os.getenv("DATABASE_URL")
local_database_url = os.getenv("DB_URL")

# In local development, prefer the explicit local DB URL to avoid
# hard failures when shared cloud credentials are unavailable.
if app_env in {"development", "dev", "local"} and local_database_url:
    database_url = local_database_url
elif not database_url and local_database_url:
    database_url = local_database_url

if database_url is None:
    raise ValueError("DATABASE_URL/DB_URL environment variable not set")

engine = create_engine(database_url, echo=True, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
