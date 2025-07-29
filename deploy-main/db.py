import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Skip database connection if it's causing issues
SKIP_DATABASE = os.getenv("SKIP_DATABASE", "false").lower() == "true"

if SKIP_DATABASE:
    DATABASE_URL = 'sqlite:///trading_bot.db'
    print("⚠️ Database connection skipped, using SQLite fallback")
elif not DATABASE_URL:
    # Set default SQLite database URL
    DATABASE_URL = 'sqlite:///trading_bot.db'
    print("⚠️ DATABASE_URL not set, using default SQLite database: trading_bot.db")

logging.basicConfig(level=logging.INFO)

# Skip database if requested
if SKIP_DATABASE:
    logging.info("[DB] Database connection skipped (SKIP_DATABASE=true)")
    # Create dummy objects to avoid import errors
    from sqlalchemy.orm import declarative_base
    Base = declarative_base()
    class DummySession:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def commit(self):
            pass
        async def rollback(self):
            pass
        async def refresh(self, obj):
            # Mock refresh - just set an ID if it doesn't have one
            if hasattr(obj, 'id') and obj.id is None:
                obj.id = 1
        def add(self, obj):
            # Mock add - just set an ID
            if hasattr(obj, 'id') and obj.id is None:
                obj.id = 1
        def query(self, *args):
            return self
        def filter(self, *args):
            return self
        def first(self):
            return None
        def all(self):
            return []
        async def execute(self, query):
            # Mock execute for SQLAlchemy queries
            class DummyResult:
                def scalars(self):
                    return self
                def first(self):
                    return None
                def all(self):
                    return []
            return DummyResult()
        async def delete(self, obj):
            pass
    AsyncSessionLocal = lambda: DummySession()
    engine = None
    async def get_db():
        yield DummySession()
else:
    logging.info(f"[DB] Connecting to: {DATABASE_URL}")

    connect_args = {}
    if DATABASE_URL.startswith("postgresql+asyncpg"):
        connect_args = {"ssl": True}
    elif DATABASE_URL.startswith("postgresql+psycopg2"):
        connect_args = {"sslmode": "require"}

    try:
        # Add timeout to prevent hanging
        engine_args = {
            "echo": False,
            "pool_size": 5,  # Reduced pool size
            "max_overflow": 10,  # Reduced overflow
            "pool_pre_ping": True,  # Test connections before using
            "pool_timeout": 10,  # 10 second timeout
            "connect_args": connect_args
        }
        
        # For PostgreSQL, add connection timeout
        if DATABASE_URL.startswith("postgresql"):
            engine_args["pool_recycle"] = 3600  # Recycle connections after 1 hour
            
        # For SQLite, use synchronous mode
        if DATABASE_URL.startswith("sqlite"):
            from sqlalchemy import create_engine
            # Create synchronous engine for SQLite
            sync_engine = create_engine(DATABASE_URL.replace("sqlite:", "sqlite+pysqlite:"), echo=False)
            # Wrap it in an async engine (will use thread pool)
            from sqlalchemy.ext.asyncio import async_engine_from_config
            engine = create_async_engine("sqlite+aiosqlite:///trading_bot.db") if False else None
            # Use a simpler approach - create a dummy async engine
            class DummyAsyncEngine:
                def __init__(self, sync_engine):
                    self.sync_engine = sync_engine
                async def dispose(self):
                    pass
            engine = DummyAsyncEngine(sync_engine)
            logging.info("[DB] SQLite database engine created")
        else:
            engine = create_async_engine(DATABASE_URL, **engine_args)
            logging.info("[DB] Database engine created successfully")
    except Exception as e:
        logging.error(f"[DB] Failed to create database engine: {e}")
        # Create a dummy engine for SQLite fallback
        logging.info("[DB] Falling back to in-memory database")
        from sqlalchemy import create_engine
        sync_engine = create_engine('sqlite:///:memory:', echo=False)
        class DummyAsyncEngine:
            def __init__(self, sync_engine):
                self.sync_engine = sync_engine
            async def dispose(self):
                pass
        engine = DummyAsyncEngine(sync_engine)
    AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    Base = declarative_base()

    async def get_db():
        async with AsyncSessionLocal() as session:
            yield session