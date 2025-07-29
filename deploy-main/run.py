#!/usr/bin/env python3
"""
Alternative entry point for the Telegram AI Trading Bot
This file provides compatibility across different environments
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import main

if __name__ == "__main__":
    main()