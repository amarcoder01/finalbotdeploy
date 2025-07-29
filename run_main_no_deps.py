#!/usr/bin/env python3
"""
Run main.py with all problematic dependencies disabled
"""
import sys
import os

# Set environment to skip problematic features
os.environ['SKIP_DATABASE'] = 'true'
os.environ['DISABLE_PANDAS'] = 'true'
os.environ['DISABLE_NUMPY'] = 'true'
os.environ['DISABLE_MATPLOTLIB'] = 'true'

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=== Starting main.py with dependencies disabled ===")

# Mock all problematic imports before anything else
class DummyModule:
    def __getattr__(self, name):
        return DummyModule()
    def __call__(self, *args, **kwargs):
        return DummyModule()

# Mock database with proper Base class
from sqlalchemy.orm import declarative_base

class MockDB:
    Base = declarative_base()  # Use real SQLAlchemy Base
    AsyncSessionLocal = lambda: DummyModule()
    engine = DummyModule()
    
    @staticmethod
    async def get_db():
        yield DummyModule()

mock_db = MockDB()
sys.modules['db'] = mock_db

# Mock pandas
sys.modules['pandas'] = DummyModule()
sys.modules['pandas.core'] = DummyModule()
sys.modules['pandas.core.api'] = DummyModule()

# Mock numpy
sys.modules['numpy'] = DummyModule()
sys.modules['numpy.core'] = DummyModule()
sys.modules['numpy.linalg'] = DummyModule()

# Mock matplotlib
sys.modules['matplotlib'] = DummyModule()
sys.modules['matplotlib.pyplot'] = DummyModule()

# Mock chart service
class DisabledChartService:
    async def generate_chart(self, *args, **kwargs):
        return None

chart_module = type(sys)('chart_service')
chart_module.ChartService = DisabledChartService
sys.modules['chart_service'] = chart_module

# Mock trading intelligence to avoid pandas imports
class SimpleTradingIntelligence:
    def __init__(self):
        from openai_service import OpenAIService
        self.openai_service = OpenAIService()
    
    async def analyze_stock(self, symbol, user_id=None):
        prompt = f"Provide a brief analysis of {symbol} stock"
        response = await self.openai_service.get_trading_insights(prompt)
        return {'analysis': response}

intel_module = type(sys)('trading_intelligence')
intel_module.TradingIntelligence = SimpleTradingIntelligence
sys.modules['trading_intelligence'] = intel_module

print("✓ All mocks installed")

# Now run main.py
exec(open('main.py').read())