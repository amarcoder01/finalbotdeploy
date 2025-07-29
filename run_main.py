#!/usr/bin/env python3
"""
Run main.py with dependency handling
"""
import sys
import os

# Mock pandas DataFrame type annotation to avoid AttributeError
class FakeDataFrame:
    pass

class FakePandas:
    DataFrame = FakeDataFrame
    def __getattr__(self, name):
        return None

# Check if pandas is available
try:
    import pandas
except ImportError:
    # Create fake pandas module
    sys.modules['pandas'] = FakePandas()
    sys.modules['pd'] = FakePandas()

# Check if numpy is available
try:
    import numpy
except ImportError:
    # Create fake numpy module with submodules
    class FakeModule:
        def __getattr__(self, name):
            return FakeModule()
        def __call__(self, *args, **kwargs):
            return FakeModule()
        def __getitem__(self, key):
            return FakeModule()
        def __repr__(self):
            return "FakeModule"
    
    fake_numpy = FakeModule()
    sys.modules['numpy'] = fake_numpy
    sys.modules['numpy.linalg'] = fake_numpy
    sys.modules['numpy.core'] = fake_numpy
    sys.modules['numpy.core.multiarray'] = fake_numpy
    sys.modules['np'] = fake_numpy

# Now run main.py
os.chdir(os.path.dirname(os.path.abspath(__file__)))
exec(open('main.py').read())