#!/bin/bash
# Minimal startup script that avoids heavy dependencies
echo "Starting minimal bot deployment..."

# Set environment variable to skip heavy imports
export SKIP_HEAVY_DEPS=true
export MINIMAL_MODE=true

# Run the deployment bot
cd /home/runner/workspace/Bot_Deployment
python deploy_bot.py