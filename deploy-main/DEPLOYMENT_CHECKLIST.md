# Render Deployment Checklist for TradeAI Companion Bot

## Pre-Deployment Checklist

### ✅ Repository Preparation
- [ ] Code is committed to GitHub repository
- [ ] All required files are present:
  - [ ] `main.py` (main application entry point)
  - [ ] `requirements.txt` (Python dependencies)
  - [ ] `render.yaml` (Render service configuration)
  - [ ] `build.sh` (build script)
  - [ ] `runtime.txt` (Python version)
  - [ ] `.env.example` (environment variables template)
  - [ ] `Procfile` (alternative process definition)

### ✅ API Keys and Credentials
- [ ] Telegram Bot Token obtained from @BotFather
- [ ] OpenAI API Key obtained from OpenAI Platform
- [ ] Optional: Alpaca API credentials for trading features
- [ ] Optional: Chart IMG API key for chart generation
- [ ] Optional: Alpha Vantage API key for additional market data

### ✅ Render Account Setup
- [ ] Render account created at [render.com](https://render.com)
- [ ] GitHub repository connected to Render
- [ ] Payment method added (for paid plans if needed)

## Deployment Steps

### Step 1: Create Services on Render

#### Option A: Using Blueprint (Recommended)
1. [ ] Go to Render Dashboard
2. [ ] Click "New" → "Blueprint"
3. [ ] Connect your GitHub repository
4. [ ] Select the repository containing the bot code
5. [ ] Render will detect `render.yaml` automatically
6. [ ] Review the services to be created:
   - [ ] Web Service: `tradeai-companion`
   - [ ] Redis: `tradeai-redis`
   - [ ] PostgreSQL: `tradeai-db`
7. [ ] Click "Apply" to create all services

#### Option B: Manual Setup
1. [ ] Create PostgreSQL Database:
   - [ ] Name: `tradeai-db`
   - [ ] Database Name: `tradeai_companion`
   - [ ] User: `tradeai_user`
   - [ ] Plan: Starter ($7/month)

2. [ ] Create Redis Service:
   - [ ] Name: `tradeai-redis`
   - [ ] Plan: Starter ($7/month)

3. [ ] Create Web Service:
   - [ ] Name: `tradeai-companion`
   - [ ] Runtime: Python 3
   - [ ] Build Command: `./build.sh`
   - [ ] Start Command: `gunicorn main:app --bind 0.0.0.0:$PORT --workers 2 --worker-class aiohttp.GunicornWebWorker --timeout 120`
   - [ ] Plan: Starter ($7/month)

### Step 2: Configure Environment Variables

#### Required Variables
- [ ] `TELEGRAM_API_TOKEN` = your_telegram_bot_token
- [ ] `OPENAI_API_KEY` = your_openai_api_key
- [ ] `ENVIRONMENT` = production
- [ ] `PORT` = 10000
- [ ] `PYTHONPATH` = .
- [ ] `TZ` = UTC
- [ ] `APSCHEDULER_TIMEZONE` = UTC
- [ ] `MPLBACKEND` = Agg
- [ ] `PYTHONUNBUFFERED` = 1
- [ ] `PIP_NO_CACHE_DIR` = 1

#### Auto-Generated Variables (Blueprint only)
- [ ] `DATABASE_URL` (from PostgreSQL service)
- [ ] `REDIS_URL` (from Redis service)

#### Optional Variables
- [ ] `ALPACA_API_KEY` = your_alpaca_key
- [ ] `ALPACA_API_SECRET` = your_alpaca_secret
- [ ] `ALPACA_BASE_URL` = https://paper-api.alpaca.markets
- [ ] `CHART_IMG_API_KEY` = your_chart_img_key
- [ ] `ALPHA_VANTAGE_API_KEY` = your_alpha_vantage_key

### Step 3: Deploy and Verify

#### Deployment
1. [ ] Click "Create Web Service" or "Apply Blueprint"
2. [ ] Monitor build logs for any errors
3. [ ] Wait for deployment to complete (usually 5-10 minutes)

#### Verification
1. [ ] Check service status in Render dashboard
2. [ ] Verify endpoints are responding:
   - [ ] `https://your-app.onrender.com/` - Main page
   - [ ] `https://your-app.onrender.com/health` - Health check
   - [ ] `https://your-app.onrender.com/ready` - Readiness check
   - [ ] `https://your-app.onrender.com/metrics` - Application metrics

3. [ ] Test Telegram bot functionality:
   - [ ] Send `/start` command to bot
   - [ ] Verify bot responds correctly
   - [ ] Test basic commands like `/help`

## Post-Deployment Tasks

### ✅ Monitoring Setup
- [ ] Set up health check monitoring
- [ ] Configure log monitoring
- [ ] Set up alerts for service downtime
- [ ] Monitor resource usage

### ✅ Security Review
- [ ] Verify all API keys are properly secured
- [ ] Check that sensitive data is not logged
- [ ] Review access controls
- [ ] Ensure HTTPS is enabled

### ✅ Performance Optimization
- [ ] Monitor response times
- [ ] Check memory usage
- [ ] Optimize worker count if needed
- [ ] Consider upgrading plans for better performance

## Troubleshooting Common Issues

### Build Failures
- [ ] Check `requirements.txt` for version conflicts
- [ ] Verify `build.sh` script permissions
- [ ] Review build logs for specific errors
- [ ] Ensure all dependencies are compatible

### Runtime Errors
- [ ] Check environment variables are set correctly
- [ ] Verify API keys are valid and have proper permissions
- [ ] Review application logs for error details
- [ ] Check database connection status

### Performance Issues
- [ ] Monitor resource usage in Render dashboard
- [ ] Check for memory leaks
- [ ] Consider upgrading to higher tier plans
- [ ] Optimize database queries

## Cost Estimation

### Starter Plan (Testing/Small Scale)
- Web Service: $7/month
- PostgreSQL: $7/month
- Redis: $7/month
- **Total: ~$21/month**

### Standard Plan (Production)
- Web Service: $25/month
- PostgreSQL: $20/month
- Redis: $20/month
- **Total: ~$65/month**

## Support Resources

- [ ] Render Documentation: [render.com/docs](https://render.com/docs)
- [ ] Render Community: [community.render.com](https://community.render.com)
- [ ] Application logs available in Render dashboard
- [ ] Health endpoints for monitoring

## Final Verification

- [ ] All services are running and healthy
- [ ] Bot responds to Telegram commands
- [ ] All required features are working
- [ ] Monitoring is set up
- [ ] Documentation is updated
- [ ] Team has access to deployment

---

**Deployment Date:** ___________
**Deployed By:** ___________
**Version:** ___________
**Notes:** ___________