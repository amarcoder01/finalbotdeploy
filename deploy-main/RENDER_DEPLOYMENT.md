# Render Deployment Guide for TradeAI Companion Bot

This guide will help you deploy the TradeAI Companion Bot to Render.com for production use.

## Prerequisites

1. **Render Account**: Sign up at [render.com](https://render.com)
2. **GitHub Repository**: Your code should be in a GitHub repository
3. **Required API Keys**:
   - Telegram Bot Token (from @BotFather)
   - OpenAI API Key (from OpenAI Platform)
   - Optional: Alpaca API credentials, Chart IMG API key

## Deployment Steps

### 1. Prepare Your Repository

Ensure your repository contains these files:
- `requirements.txt` - Python dependencies
- `render.yaml` - Render service configuration
- `build.sh` - Build script for deployment
- `Procfile` - Alternative process definition
- `.env.example` - Environment variables template

### 2. Create Services on Render

#### Option A: Using render.yaml (Recommended)

1. Go to your Render Dashboard
2. Click "New" → "Blueprint"
3. Connect your GitHub repository
4. Render will automatically detect the `render.yaml` file
5. Review the services that will be created:
   - **Web Service**: Main application
   - **Redis**: For caching and session management
   - **PostgreSQL**: Database for user data and trading history

#### Option B: Manual Setup

1. **Create Web Service**:
   - Go to Render Dashboard
   - Click "New" → "Web Service"
   - Connect your GitHub repository
   - Configure:
     - **Name**: `tradeai-companion`
     - **Runtime**: `Python 3`
     - **Build Command**: `./build.sh`
     - **Start Command**: `gunicorn main:app --bind 0.0.0.0:$PORT --workers 2 --worker-class uvicorn.workers.UvicornWorker --timeout 120`
     - **Plan**: Start with "Starter" (can upgrade later)

2. **Create Redis Service**:
   - Click "New" → "Redis"
   - **Name**: `tradeai-redis`
   - **Plan**: "Starter"

3. **Create PostgreSQL Database**:
   - Click "New" → "PostgreSQL"
   - **Name**: `tradeai-db`
   - **Database Name**: `tradeai_companion`
   - **User**: `tradeai_user`
   - **Plan**: "Starter"

### 3. Configure Environment Variables

In your Web Service settings, add these environment variables:

#### Required Variables
```
TELEGRAM_API_TOKEN=your_telegram_bot_token
OPENAI_API_KEY=your_openai_api_key
ENVIRONMENT=production
PORT=8080
PYTHONPATH=.
TZ=UTC
APSCHEDULER_TIMEZONE=UTC
```

#### Optional Variables
```
ALPACA_API_KEY=your_alpaca_key
ALPACA_API_SECRET=your_alpaca_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
CHART_IMG_API_KEY=your_chart_img_key
```

#### Auto-Generated Variables (if using Blueprint)
```
REDIS_URL=redis://...
DATABASE_URL=postgresql://...
```

### 4. Deploy

1. Click "Create Web Service" or "Apply Blueprint"
2. Render will:
   - Clone your repository
   - Run the build script
   - Install dependencies
   - Start your application

### 5. Verify Deployment

Once deployed, check these endpoints:
- `https://your-app.onrender.com/` - Main page
- `https://your-app.onrender.com/health` - Health check
- `https://your-app.onrender.com/ready` - Readiness check
- `https://your-app.onrender.com/metrics` - Application metrics

## Configuration Details

### Build Process

The `build.sh` script will:
1. Update pip
2. Install system dependencies (tesseract, poppler)
3. Install Python packages from `requirements.txt`
4. Create necessary directories
5. Initialize database
6. Setup qlib data

### Runtime Configuration

- **Web Server**: Gunicorn with Uvicorn workers
- **Workers**: 2 (can be adjusted based on plan)
- **Timeout**: 120 seconds
- **Health Check**: `/health` endpoint
- **Port**: 8080 (Render standard)

### Database Setup

The application will automatically:
1. Connect to PostgreSQL using `DATABASE_URL`
2. Initialize database tables
3. Set up user management and trading data storage

### Redis Configuration

Redis is used for:
- Performance caching
- Session management
- Rate limiting
- Temporary data storage

## Monitoring and Maintenance

### Health Monitoring

Render automatically monitors:
- Health check endpoint (`/health`)
- Application responsiveness
- Resource usage

### Logs

Access logs through:
- Render Dashboard → Your Service → Logs
- Real-time log streaming
- Historical log search

### Scaling

To handle more users:
1. Upgrade to "Standard" or "Pro" plan
2. Increase worker count in start command
3. Upgrade database and Redis plans

## Troubleshooting

### Common Issues

1. **Build Failures**:
   - Check `requirements.txt` for version conflicts
   - Verify system dependencies in `build.sh`
   - Check build logs for specific errors

2. **Environment Variables**:
   - Ensure all required variables are set
   - Check for typos in variable names
   - Verify API keys are valid

3. **Database Connection**:
   - Verify `DATABASE_URL` is correctly set
   - Check database service status
   - Review connection logs

4. **Telegram Bot Issues**:
   - Verify bot token is correct
   - Check bot permissions
   - Ensure webhook is properly configured

### Performance Optimization

1. **Caching**: Redis is configured for optimal performance
2. **Connection Pooling**: Database connections are pooled
3. **Async Processing**: All I/O operations are asynchronous
4. **Resource Monitoring**: Metrics endpoint provides insights

## Security Considerations

- All API keys are stored as environment variables
- Database connections use SSL
- Rate limiting is enabled
- Input validation is enforced
- Secure logging is implemented

## Support

For deployment issues:
1. Check Render documentation: [render.com/docs](https://render.com/docs)
2. Review application logs
3. Verify environment configuration
4. Test locally before deploying

## Cost Estimation

**Starter Plan** (Recommended for testing):
- Web Service: $7/month
- PostgreSQL: $7/month
- Redis: $7/month
- **Total**: ~$21/month

**Standard Plan** (Production):
- Web Service: $25/month
- PostgreSQL: $20/month
- Redis: $20/month
- **Total**: ~$65/month

---

**Note**: This deployment configuration is optimized for production use with proper monitoring, security, and scalability features.