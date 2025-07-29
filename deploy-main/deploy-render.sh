#!/bin/bash

# Render Deployment Guide for TradeAI Companion
# Since Render doesn't have an official CLI for deployments,
# this script helps with the deployment process

echo "=== Render Deployment Setup ==="

# Ensure all files are committed
echo "Checking git status..."
git status

echo "\nMake sure all changes are committed and pushed to GitHub."
echo "\nTo deploy on Render:"
echo "1. Go to https://dashboard.render.com"
echo "2. Click 'New +' -> 'Web Service'"
echo "3. Connect your GitHub repository: amarcoder01/deploy"
echo "4. Use these settings:"
echo "   - Name: tradeai-companion"
echo "   - Runtime: Python 3"
echo "   - Build Command: pip install -r requirements.txt"
echo "   - Start Command: python -m aiohttp.web -H 0.0.0.0 -P \$PORT main:app"
echo "\nOr use the render.yaml file for automatic configuration."
echo "\nEnvironment variables to set:"
echo "- TELEGRAM_API_TOKEN"
echo "- OPENAI_API_KEY"
echo "- ALPACA_API_KEY"
echo "- ALPACA_API_SECRET"
echo "- CHART_IMG_API_KEY"

echo "\n=== Auto-deployment is enabled via render.yaml ==="
echo "Any push to main branch will trigger automatic deployment."