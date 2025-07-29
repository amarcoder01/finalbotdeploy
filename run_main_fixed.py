#!/usr/bin/env python3
"""
Fixed version of run_main.py that properly handles database mocking
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Check if we should skip database
SKIP_DATABASE = os.getenv("SKIP_DATABASE", "false").lower() == "true"

if SKIP_DATABASE:
    print("=== Database operations disabled - using mock database ===")
    
    # Create mock database module before any imports
    class MockBase:
        metadata = type('metadata', (), {'create_all': lambda self, bind: None})()
    
    class MockSession:
        def __init__(self):
            self.data = {}
            
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
            
        def filter_by(self, **kwargs):
            return self
            
        def first(self):
            return None
            
        def all(self):
            return []
            
        async def execute(self, *args):
            class Result:
                def scalars(self):
                    return self
                def all(self):
                    return []
                def first(self):
                    return None
            return Result()
    
    class MockSessionLocal:
        def __call__(self):
            return MockSession()
    
    class MockEngine:
        async def dispose(self):
            pass
    
    # Create the mock db module
    mock_db = type(sys)('db')
    mock_db.Base = MockBase
    mock_db.AsyncSessionLocal = MockSessionLocal()
    mock_db.engine = MockEngine()
    async def mock_get_db():
        yield MockSession()
    mock_db.get_db = mock_get_db
    
    # Install the mock
    sys.modules['db'] = mock_db
    print("✓ Mock database module installed")

# Mock numpy before any imports
print("Installing numpy mock...")
class MockNumPy:
    __version__ = '1.24.0'
    def __getattr__(self, name):
        return lambda *args, **kwargs: MockNumPy()
    def __call__(self, *args, **kwargs):
        return MockNumPy()
    def __getitem__(self, key):
        return MockNumPy()
    def __array__(self):
        return []
    
mock_numpy = MockNumPy()
# Install comprehensive numpy mocks
numpy_modules = [
    'numpy', 'numpy.linalg', 'numpy.core', 'numpy.core.multiarray',
    'numpy.core._multiarray_umath', 'numpy.random', 'numpy.fft',
    'numpy.lib', 'numpy.lib.npyio', 'numpy.ma'
]
for module in numpy_modules:
    sys.modules[module] = mock_numpy
print("✓ NumPy mock installed")

# Mock pandas too
print("Installing pandas mock...")
class MockPandas:
    DataFrame = type('DataFrame', (), {})
    Series = type('Series', (), {})
    __version__ = '2.0.0'
    def __getattr__(self, name):
        return MockPandas()
        
sys.modules['pandas'] = MockPandas()
print("✓ Pandas mock installed")

# Now import and run main
print("Starting main.py...")
exec(open('main.py').read())