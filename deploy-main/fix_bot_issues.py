#!/usr/bin/env python3
"""
Bot Issue Diagnosis and Fix Script
This script identifies and fixes common issues preventing the bot from responding
"""

import os
import sys
import subprocess
import requests
from datetime import datetime

def check_openai_quota():
    """Check OpenAI API quota status"""
    print("\n🔍 Checking OpenAI API Status...")
    try:
        from openai import OpenAI
        from dotenv import load_dotenv
        
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            print("❌ OPENAI_API_KEY not found in environment")
            return False
            
        client = OpenAI(api_key=api_key)
        
        # Test with a minimal request
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5
        )
        
        print("✅ OpenAI API is working")
        return True
        
    except Exception as e:
        error_str = str(e)
        if "insufficient_quota" in error_str or "429" in error_str:
            print("❌ OpenAI API Quota Exceeded!")
            print("   - Check your OpenAI billing and usage at https://platform.openai.com/usage")
            print("   - Consider upgrading your plan or waiting for quota reset")
        elif "401" in error_str:
            print("❌ Invalid OpenAI API Key")
        else:
            print(f"❌ OpenAI API Error: {error_str}")
        return False

def check_telegram_bot():
    """Check Telegram bot status"""
    print("\n🔍 Checking Telegram Bot Status...")
    try:
        from dotenv import load_dotenv
        
        load_dotenv()
        bot_token = os.getenv('TELEGRAM_API_TOKEN')
        
        if not bot_token:
            print("❌ TELEGRAM_API_TOKEN not found in environment")
            return False
            
        # Test bot API
        response = requests.get(f"https://api.telegram.org/bot{bot_token}/getMe")
        
        if response.status_code == 200:
            bot_info = response.json()
            if bot_info['ok']:
                print(f"✅ Telegram Bot is active: @{bot_info['result']['username']}")
                return True
        
        print(f"❌ Telegram Bot API Error: {response.status_code}")
        return False
        
    except Exception as e:
        print(f"❌ Telegram Bot Check Error: {e}")
        return False

def check_dependencies():
    """Check for missing dependencies"""
    print("\n🔍 Checking Dependencies...")
    
    missing_deps = []
    
    # Check for pypfopt
    try:
        import pypfopt
        print("✅ pypfopt is installed")
    except ImportError:
        missing_deps.append('pypfopt')
        print("❌ pypfopt is missing (needed for portfolio optimization)")
    
    # Check for other critical dependencies
    critical_deps = ['telegram', 'openai', 'pandas', 'numpy', 'yfinance']
    
    for dep in critical_deps:
        try:
            __import__(dep)
            print(f"✅ {dep} is installed")
        except ImportError:
            missing_deps.append(dep)
            print(f"❌ {dep} is missing")
    
    return missing_deps

def install_missing_dependencies(missing_deps):
    """Install missing dependencies"""
    if not missing_deps:
        return True
        
    print(f"\n📦 Installing missing dependencies: {', '.join(missing_deps)}")
    
    try:
        for dep in missing_deps:
            print(f"Installing {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"✅ {dep} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_bot_processes():
    """Check for multiple bot processes"""
    print("\n🔍 Checking for Multiple Bot Processes...")
    
    try:
        # On Windows, use tasklist
        if os.name == 'nt':
            result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                                  capture_output=True, text=True)
            python_processes = result.stdout.count('python.exe')
            
            if python_processes > 1:
                print(f"⚠️  Found {python_processes} Python processes running")
                print("   Multiple bot instances might be causing conflicts")
                return False
            else:
                print("✅ No conflicting processes detected")
                return True
        else:
            # On Unix-like systems
            result = subprocess.run(['pgrep', '-f', 'main.py'], 
                                  capture_output=True, text=True)
            if result.stdout.strip():
                processes = result.stdout.strip().split('\n')
                if len(processes) > 1:
                    print(f"⚠️  Found {len(processes)} bot processes running")
                    return False
            
            print("✅ No conflicting processes detected")
            return True
            
    except Exception as e:
        print(f"⚠️  Could not check processes: {e}")
        return True

def generate_fix_recommendations():
    """Generate recommendations to fix the bot"""
    print("\n🔧 RECOMMENDED FIXES:")
    print("="*50)
    
    # Check all issues
    openai_ok = check_openai_quota()
    telegram_ok = check_telegram_bot()
    missing_deps = check_dependencies()
    processes_ok = check_bot_processes()
    
    print("\n📋 SUMMARY & FIXES:")
    print("="*30)
    
    if not openai_ok:
        print("\n🚨 CRITICAL: OpenAI API Issue")
        print("   1. Check your OpenAI billing at https://platform.openai.com/usage")
        print("   2. Upgrade your plan or wait for quota reset")
        print("   3. Consider using a different API key")
    
    if not telegram_ok:
        print("\n🚨 CRITICAL: Telegram Bot Issue")
        print("   1. Verify TELEGRAM_API_TOKEN environment variable")
        print("   2. Check if bot token is valid")
        print("   3. Ensure bot is not revoked")
    
    if missing_deps:
        print("\n⚠️  DEPENDENCIES: Missing packages")
        print(f"   Run: pip install {' '.join(missing_deps)}")
        
        # Auto-install if user wants
        response = input("\n❓ Install missing dependencies now? (y/n): ")
        if response.lower() == 'y':
            install_missing_dependencies(missing_deps)
    
    if not processes_ok:
        print("\n⚠️  PROCESSES: Multiple bot instances detected")
        print("   1. Stop all running bot instances")
        print("   2. Restart only one instance")
        print("   3. Use Ctrl+C to stop current bot")
    
    # Overall status
    print("\n🎯 OVERALL STATUS:")
    if openai_ok and telegram_ok and not missing_deps and processes_ok:
        print("✅ All systems appear to be working!")
        print("   If bot still not responding, try restarting it.")
    else:
        print("❌ Issues detected that need fixing")
        print("   Fix the above issues and restart the bot")
    
    print("\n" + "="*50)
    print(f"Diagnosis completed at {datetime.now()}")

if __name__ == "__main__":
    print("🤖 TradeAI Bot Issue Diagnosis")
    print("="*40)
    
    generate_fix_recommendations()