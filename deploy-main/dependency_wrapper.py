"""
Dependency wrapper to handle missing packages gracefully
"""
import sys
import logging

logger = logging.getLogger('DependencyWrapper')

# Mock numpy if not available
try:
    import numpy
except ImportError:
    logger.warning("NumPy not available, creating mock...")
    class MockNumPy:
        def __getattr__(self, name):
            if name == '__version__':
                return '1.24.0-mock'
            return lambda *args, **kwargs: None
        
        def array(self, *args, **kwargs):
            return []
        
        def zeros(self, *args, **kwargs):
            return []
            
    sys.modules['numpy'] = MockNumPy()
    import numpy

# Mock pandas if not available
try:
    import pandas
except ImportError:
    logger.warning("Pandas not available, creating mock...")
    class MockDataFrame:
        def __init__(self, *args, **kwargs):
            self.data = []
        
        def __getattr__(self, name):
            return lambda *args, **kwargs: self
        
        def __len__(self):
            return 0
        
        def __getitem__(self, key):
            return []
    
    class MockPandas:
        DataFrame = MockDataFrame
        
        def __getattr__(self, name):
            return lambda *args, **kwargs: MockDataFrame()
    
    sys.modules['pandas'] = MockPandas()
    import pandas

# Mock yfinance if not available
try:
    import yfinance
except ImportError:
    logger.warning("yfinance not available, creating mock...")
    class MockTicker:
        def __init__(self, symbol):
            self.symbol = symbol
        
        @property
        def info(self):
            return {'regularMarketPrice': 100.0, 'symbol': self.symbol}
        
        def history(self, *args, **kwargs):
            return MockDataFrame()
    
    class MockYFinance:
        def Ticker(self, symbol):
            return MockTicker(symbol)
        
        def download(self, *args, **kwargs):
            return MockDataFrame()
    
    sys.modules['yfinance'] = MockYFinance()
    import yfinance

logger.info("Dependency wrapper initialized")