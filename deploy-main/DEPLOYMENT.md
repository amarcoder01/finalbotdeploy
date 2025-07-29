# TradeAI Companion - Render Deployment Guide

## Overview
This guide explains how to deploy the TradeAI Companion bot to Render.com using the automated `render.yaml` configuration.

## Prerequisites
- GitHub repository with your code
- Render.com account
- Required API keys (see Environment Variables section)

## Deployment Methods

### Method 1: Automatic Deployment via render.yaml (Recommended)

1. **Push to GitHub**: Ensure all your code is committed and pushed to the `main` branch
2. **Connect to Render**:
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository: `amarcoder01/deploy`
   - Render will automatically detect the `render.yaml` file

3. **Auto-configuration**: The `render.yaml` file will automatically configure:
   - Python 3 runtime
   - Build command: `pip install -r requirements.txt`
   - Start command: `python -m aiohttp.web -H 0.0.0.0 -P $PORT main:app`
   - Redis and PostgreSQL services

### Method 2: Manual Configuration

If you prefer manual setup:

1. **Create Web Service**:
   - Name: `tradeai-companion`
   - Runtime: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python -m aiohttp.web -H 0.0.0.0 -P $PORT main:app`

2. **Add Services**:
   - Redis: For caching and session management
   - PostgreSQL: For data persistence

## Environment Variables

Set these environment variables in your Render service:

```
TELEGRAM_API_TOKEN=your_telegram_bot_token
OPENAI_API_KEY=your_openai_api_key
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_API_SECRET=your_alpaca_secret_key
CHART_IMG_API_KEY=your_chart_api_key
```

## Deployment Scripts

### For Windows (PowerShell)
```powershell
.\deploy-render.ps1
```

### For Linux/Mac (Bash)
```bash
./deploy-render.sh
```

These scripts will:
- Check git status
- Provide deployment instructions
- Optionally push changes to trigger auto-deployment

## Auto-Deployment

Once configured, any push to the `main` branch will automatically trigger a new deployment on Render.

## Monitoring

- **Logs**: Available in Render dashboard
- **Metrics**: CPU, memory, and request metrics
- **Health Checks**: Automatic health monitoring

## Troubleshooting

### Common Issues

1. **Build Failures**:
   - Check `requirements.txt` for correct dependencies
   - Ensure Python version compatibility

2. **Start Command Issues**:
   - Verify the start command matches: `python -m aiohttp.web -H 0.0.0.0 -P $PORT main:app`
   - Check that `main.py` contains the `app` object

3. **Environment Variables**:
   - Ensure all required API keys are set
   - Check for typos in variable names

### Support

- [Render Documentation](https://render.com/docs)
- [Render Community](https://community.render.com)

## Cost Optimization

- Use Render's free tier for development
- Monitor usage to optimize resource allocation
- Consider scaling options for production use