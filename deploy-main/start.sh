#!/bin/bash
# Startup script for the original bot with proper environment setup
export SKIP_DATABASE=true
export PYTHONPATH=/home/runner/workspace/Bot_Deployment:$PYTHONPATH
cd /home/runner/workspace/Bot_Deployment
echo "Starting Trading Bot from original codebase..."
python main.py