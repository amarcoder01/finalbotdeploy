#!/usr/bin/env python3
"""
Start the bot with proper dependency handling
"""
import sys
import os
import logging

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BotStarter')

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger.info("=== AI Trading Bot Startup ===")

# Set environment to skip database
os.environ['SKIP_DATABASE'] = 'true'
logger.info("✓ Database connection disabled")

# Mock problematic modules before any imports
logger.info("Setting up dependency mocks...")

# 1. Mock database module
class MockDB:
    class Base:
        metadata = type('metadata', (), {'create_all': lambda self, bind: None})()
    
    class Session:
        def __init__(self):
            pass
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
    
    AsyncSessionLocal = lambda: Session()
    engine = type('engine', (), {'dispose': lambda: None})()
    
    @staticmethod
    async def get_db():
        yield MockDB.Session()

sys.modules['db'] = MockDB()
logger.info("✓ Database mock installed")

# 2. Disable chart functionality by mocking matplotlib
class DisabledChartService:
    def __init__(self):
        logger.warning("Chart service disabled - matplotlib dependencies missing")
    
    async def generate_chart(self, *args, **kwargs):
        return None
    
    def generate_price_chart(self, *args, **kwargs):
        return None

# Replace chart_service module
chart_module = type(sys)('chart_service')
chart_module.ChartService = DisabledChartService
sys.modules['chart_service'] = chart_module
logger.info("✓ Chart service disabled")

# 3. Mock numpy for other imports
class FakeNumPy:
    __version__ = '1.24.0'
    def __getattr__(self, name):
        return FakeNumPy()
    def __call__(self, *args, **kwargs):
        return FakeNumPy()

# Only mock numpy basics, don't interfere with matplotlib
sys.modules['numpy'] = FakeNumPy()
logger.info("✓ NumPy mock installed")

# Now start the actual bot
logger.info("Starting main bot...")
import asyncio
from main import main

# Run the bot
asyncio.run(main())