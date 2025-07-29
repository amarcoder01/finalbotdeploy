#!/bin/bash
# Start main.py with database connection skipped
export SKIP_DATABASE=true
echo "Starting main.py with database skip..."
cd /home/runner/workspace/Bot_Deployment
python run_main.py