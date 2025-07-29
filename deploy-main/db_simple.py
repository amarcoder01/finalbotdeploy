"""
Simple database module that avoids async complexity
"""
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Skip database if requested
SKIP_DATABASE = os.getenv("SKIP_DATABASE", "false").lower() == "true"

logging.basicConfig(level=logging.INFO)

if SKIP_DATABASE:
    logging.info("[DB] Database operations disabled - SKIP_DATABASE=true")
    
    # Create dummy classes to avoid import errors
    class DummyBase:
        pass
    
    class DummySession:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def commit(self):
            pass
        async def rollback(self):
            pass
        def add(self, obj):
            pass
        def query(self, *args):
            return self
        def filter(self, *args):
            return self
        def first(self):
            return None
        def all(self):
            return []
    
    class DummySessionLocal:
        def __call__(self):
            return DummySession()
    
    # Export dummy objects
    Base = DummyBase
    AsyncSessionLocal = DummySessionLocal()
    engine = None
    
    async def get_db():
        yield DummySession()
    
    logging.info("[DB] Using dummy database objects")
else:
    # Import the real database module
    from db import Base, AsyncSessionLocal, engine, get_db
    logging.info("[DB] Using real database connection")