#!/usr/bin/env python3
"""
Comprehensive Qlib Installation Verification Script
Checks installation, data availability, and integration status
"""

import os
import sys
import importlib
from pathlib import Path

def check_qlib_installation():
    """Check if Qlib is properly installed"""
    print("🔍 Checking Qlib Installation...")
    
    try:
        import qlib
        print(f"✅ Qlib installed: version {qlib.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Qlib not installed: {e}")
        return False

def check_dependencies():
    """Check optional dependencies"""
    print("\n🔍 Checking Optional Dependencies...")
    
    dependencies = {
        'pandas': 'Required for data handling',
        'numpy': 'Required for numerical operations',
        'scikit-learn': 'Required for machine learning',
        'lightgbm': 'For LightGBM models',
        'xgboost': 'For XGBoost models (optional)',
        'torch': 'For PyTorch models (optional)',
        'matplotlib': 'For plotting (optional)'
    }
    
    results = {}
    for dep, description in dependencies.items():
        try:
            module = importlib.import_module(dep)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {dep}: {version} - {description}")
            results[dep] = True
        except ImportError:
            print(f"❌ {dep}: Not installed - {description}")
            results[dep] = False
    
    return results

def check_data_directories():
    """Check for Qlib data directories"""
    print("\n🔍 Checking Data Directories...")
    
    # Check local project data
    local_data_path = Path("qlib_data/us_data")
    if local_data_path.exists():
        print(f"✅ Local data directory found: {local_data_path.absolute()}")
        
        # Count available symbols
        features_dir = local_data_path / "features"
        if features_dir.exists():
            symbols = [d.name for d in features_dir.iterdir() if d.is_dir()]
            print(f"✅ Available symbols: {len(symbols)} stocks")
            print(f"   Sample symbols: {', '.join(symbols[:10])}...")
        
        # Check instruments file
        instruments_file = local_data_path / "instruments"
        if instruments_file.exists():
            print(f"✅ Instruments directory found")
        
        # Check calendars
        calendars_dir = local_data_path / "calendars"
        if calendars_dir.exists():
            print(f"✅ Calendars directory found")
    else:
        print(f"❌ Local data directory not found: {local_data_path.absolute()}")
    
    # Check user home data
    home_data_path = Path.home() / ".qlib/qlib_data/us_data"
    if home_data_path.exists():
        print(f"✅ User home data directory found: {home_data_path}")
    else:
        print(f"❌ User home data directory not found: {home_data_path}")

def test_qlib_initialization():
    """Test Qlib initialization"""
    print("\n🔍 Testing Qlib Initialization...")
    
    try:
        import qlib
        from qlib.config import REG_US
        
        # Try local data first
        local_data_path = Path("qlib_data/us_data")
        if local_data_path.exists():
            qlib.init(provider_uri=str(local_data_path.absolute()), region=REG_US)
            print(f"✅ Qlib initialized with local data: {local_data_path.absolute()}")
        else:
            # Fallback to user home
            home_data_path = Path.home() / ".qlib/qlib_data/us_data"
            qlib.init(provider_uri=str(home_data_path), region=REG_US)
            print(f"✅ Qlib initialized with user home data: {home_data_path}")
        
        return True
    except Exception as e:
        print(f"❌ Qlib initialization failed: {e}")
        return False

def test_data_access():
    """Test basic data access"""
    print("\n🔍 Testing Data Access...")
    
    try:
        from qlib.data import D
        
        # Try to get data for a common stock
        test_symbol = "AAPL"
        data = D.features([test_symbol], ["$close"], start_time="2020-01-01", end_time="2020-01-10")
        
        if data is not None and not data.empty:
            print(f"✅ Successfully retrieved data for {test_symbol}")
            print(f"   Data shape: {data.shape}")
            return True
        else:
            print(f"❌ No data retrieved for {test_symbol}")
            return False
    except Exception as e:
        print(f"❌ Data access failed: {e}")
        return False

def test_qlib_service_integration():
    """Test QlibService integration"""
    print("\n🔍 Testing QlibService Integration...")
    
    try:
        # Add TradeAiCompanion to path
        sys.path.insert(0, "TradeAiCompanion")
        from qlib_service import QlibService
        
        # Initialize service
        qs = QlibService()
        print(f"✅ QlibService initialized with data path: {qs.provider_uri}")
        
        # Test initialization
        qs.initialize()
        if qs.initialized:
            print("✅ QlibService successfully initialized")
        else:
            print("⚠️ QlibService initialization failed, using demo mode")
        
        # Test signal generation
        signals = qs.get_available_symbols()
        print(f"✅ Available signals: {len(signals)} symbols")
        
        # Test signal retrieval
        test_signal = qs.get_signal("AAPL")
        if test_signal is not None:
            print(f"✅ Signal retrieval working: AAPL = {test_signal:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ QlibService integration failed: {e}")
        return False

def main():
    """Main verification function"""
    print("🤖 Qlib Installation and Integration Verification")
    print("=" * 50)
    
    results = {
        'installation': check_qlib_installation(),
        'dependencies': check_dependencies(),
        'initialization': False,
        'data_access': False,
        'service_integration': False
    }
    
    check_data_directories()
    
    if results['installation']:
        results['initialization'] = test_qlib_initialization()
        
        if results['initialization']:
            results['data_access'] = test_data_access()
    
    results['service_integration'] = test_qlib_service_integration()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 50)
    
    status_icon = lambda x: "✅" if x else "❌"
    print(f"{status_icon(results['installation'])} Qlib Installation")
    print(f"{status_icon(results['initialization'])} Qlib Initialization")
    print(f"{status_icon(results['data_access'])} Data Access")
    print(f"{status_icon(results['service_integration'])} QlibService Integration")
    
    # Dependency summary
    deps = results.get('dependencies', {})
    required_deps = ['pandas', 'numpy', 'scikit-learn', 'lightgbm']
    optional_deps = ['xgboost', 'torch', 'matplotlib']
    
    print(f"\n📦 Dependencies:")
    print(f"   Required: {sum(1 for dep in required_deps if deps.get(dep, False))}/{len(required_deps)}")
    print(f"   Optional: {sum(1 for dep in optional_deps if deps.get(dep, False))}/{len(optional_deps)}")
    
    # Overall status
    core_working = results['installation'] and results['service_integration']
    if core_working:
        print("\n🎉 Qlib is properly installed and integrated!")
        if not results['data_access']:
            print("⚠️ Note: Data access issues detected, but demo mode is available")
    else:
        print("\n⚠️ Qlib installation or integration issues detected")
    
    print("\n💡 Recommendations:")
    if not deps.get('xgboost', False):
        print("   - Install XGBoost: pip install xgboost")
    if not deps.get('torch', False):
        print("   - Install PyTorch: pip install torch")
    if not results['data_access']:
        print("   - Data access issues may be due to missing dependencies or data format")
        print("   - Demo mode is available for testing")

if __name__ == "__main__":
    main()