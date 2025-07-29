# 🚀 Render Deployment Setup Instructions

## ✅ Port Binding Issue - RESOLVED

The "No open ports detected" error has been successfully fixed! Your bot is now properly binding to port 10000.

## 🔑 Required: Set Environment Variables

Your deployment is failing because the required API keys are missing. You need to manually set these environment variables in the Render dashboard:

### Step 1: Access Render Dashboard
1. Go to [Render Dashboard](https://dashboard.render.com)
2. Find your `tradeai-companion` service
3. Click on the service name
4. Go to the **Environment** tab

### Step 2: Add Required Environment Variables

Add these environment variables with your actual API keys:

| Variable Name | Description | Required |
|---------------|-------------|----------|
| `TELEGRAM_API_TOKEN` | Your Telegram Bot Token from @BotFather | ✅ Yes |
| `OPENAI_API_KEY` | Your OpenAI API Key | ✅ Yes |
| `ALPACA_API_KEY` | Your Alpaca Trading API Key | ⚠️ Optional* |
| `ALPACA_API_SECRET` | Your Alpaca Trading API Secret | ⚠️ Optional* |
| `CHART_IMG_API_KEY` | Chart image generation API key | ⚠️ Optional* |

*Optional variables: The bot will work without these but some features may be limited.

### Step 3: How to Add Environment Variables

1. In the Environment tab, click **Add Environment Variable**
2. Enter the **Key** (e.g., `TELEGRAM_API_TOKEN`)
3. Enter the **Value** (your actual API key)
4. Click **Save Changes**
5. Repeat for each required variable

### Step 4: Redeploy

After adding all environment variables:
1. Click **Manual Deploy** or
2. Push any small change to trigger auto-deployment

## 🔍 How to Get API Keys

### Telegram Bot Token
1. Message @BotFather on Telegram
2. Use `/newbot` command
3. Follow instructions to create your bot
4. Copy the token provided

### OpenAI API Key
1. Go to [OpenAI API Keys](https://platform.openai.com/api-keys)
2. Create a new API key
3. Copy the key (starts with `sk-`)

### Alpaca API Keys (Optional)
1. Sign up at [Alpaca](https://alpaca.markets)
2. Go to Paper Trading section
3. Generate API keys for paper trading

## 🎯 Current Status

✅ **Port Binding**: Fixed - Bot successfully binds to port 10000  
✅ **Web Server**: Working - Health endpoints available  
✅ **Application Structure**: Correct - No conflicts between servers  
❌ **Environment Variables**: Missing - Need to be set in Render dashboard  

## 🔧 Troubleshooting

If the bot still doesn't work after setting environment variables:

1. **Check Logs**: View deployment logs in Render dashboard
2. **Verify Variables**: Ensure no extra spaces in API keys
3. **Test Health Endpoint**: Visit `https://your-app.onrender.com/health`
4. **Check Bot**: Message your bot on Telegram

## 📞 Support

If you need help:
1. Check the deployment logs in Render dashboard
2. Verify all environment variables are correctly set
3. Test the health endpoint to confirm the web server is running

---

**Next Step**: Set the environment variables in Render dashboard and redeploy! 🚀