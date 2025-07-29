#!/usr/bin/env python3
"""
Quick Setup Script for Telegram AI Trading Bot
Automatically configures the environment and validates setup
"""

import os
import sys
import subprocess
import platform
from dotenv import load_dotenv

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor} is not compatible. Need Python 3.8+")
        return False

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        # Install basic dependencies directly
        packages = [
            "python-telegram-bot", "openai", "yfinance", "matplotlib", 
            "pandas", "numpy", "python-dotenv", "asyncio", "aiohttp"
        ]
        subprocess.run([sys.executable, "-m", "pip", "install"] + packages, 
                      check=True, capture_output=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_env_file():
    """Create .env file with basic template if it doesn't exist"""
    if not os.path.exists('.env'):
        print("📝 Creating .env file with basic template...")
        env_template = """# Telegram AI Trading Bot Configuration
# Replace the placeholder values with your actual API keys

TELEGRAM_API_TOKEN=your_telegram_token_here
OPENAI_API_KEY=your_openai_api_key_here
DATABASE_URL=sqlite:///trading_bot.db

# Optional configurations
LOG_LEVEL=INFO
DEBUG=False
"""
        with open('.env', 'w') as dst:
            dst.write(env_template)
        print("✅ .env file created. Please edit it with your API keys.")
        return True
    else:
        print("✅ .env file already exists")
        return True

def validate_env_vars():
    """Check if required environment variables are set"""
    required_vars = ['TELEGRAM_API_TOKEN', 'OPENAI_API_KEY']
    
    # Load .env file
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    if value and value != f'your_{key.lower()}_here':
                        os.environ[key] = value
    
    missing_vars = []
    for var in required_vars:
        if not os.environ.get(var) or os.environ.get(var).startswith('your_'):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️  Missing API keys: {', '.join(missing_vars)}")
        print("Please edit .env file with your actual API keys")
        return False
    else:
        print("✅ All required API keys are configured")
        return True

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing module imports...")
    try:
        import telegram
        import openai
        import yfinance
        import matplotlib
        import pandas
        import numpy
        print("✅ All modules import successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main setup function"""
    print("🤖 Telegram AI Trading Bot - Quick Setup")
    print("=" * 50)
    
    print(f"🖥️  Platform: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {sys.version}")
    print()
    
    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Step 2: Install dependencies
    if not install_dependencies():
        print("Please install dependencies manually using pip")
        sys.exit(1)
    
    # Step 3: Create .env file
    if not create_env_file():
        sys.exit(1)
    
    # Step 4: Test imports
    if not test_imports():
        print("Please check your Python environment and dependencies")
        sys.exit(1)
    
    # Step 5: Validate environment variables
    env_ok = validate_env_vars()
    
    print()
    print("🎉 Setup Complete!")
    print("=" * 50)
    
    if env_ok:
        print("✅ Your bot is ready to run!")
        print("Execute: python main.py")
    else:
        print("⚠️  Please configure your API keys in .env file, then run:")
        print("python main.py")
    
    print()
    print("📚 Need help? Check README.md for more information")

if __name__ == "__main__":
    main()