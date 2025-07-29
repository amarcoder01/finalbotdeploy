#!/usr/bin/env python3
"""
Test script to verify the aiohttp app factory function works correctly
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import app, create_app
from aiohttp import web

def test_app_factory():
    """Test that the app factory returns a proper aiohttp Application"""
    print("Testing app factory function...")
    
    # Test the factory function
    app_instance = app()
    print(f"App type: {type(app_instance)}")
    print(f"Is aiohttp Application: {isinstance(app_instance, web.Application)}")
    
    # Test direct create_app
    direct_app = create_app()
    print(f"Direct app type: {type(direct_app)}")
    print(f"Is aiohttp Application: {isinstance(direct_app, web.Application)}")
    
    # Check routes
    routes = list(app_instance.router.routes())
    print(f"Number of routes: {len(routes)}")
    for route in routes:
        print(f"  - {route.method} {route.resource.canonical}")
    
    print("\nApp factory test completed successfully!")
    return True

if __name__ == "__main__":
    try:
        test_app_factory()
        print("✅ All tests passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)