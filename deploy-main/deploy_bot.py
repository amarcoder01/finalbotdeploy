#!/usr/bin/env python3
"""
Direct deployment script that runs with existing packages only
No additional installations required
"""
import sys
import os
import asyncio
import logging

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DeployBot')

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Only import what's already installed
try:
    from telegram.ext import Application, CommandHandler, MessageHandler, filters
    from aiohttp import web
    import os
    from datetime import datetime
    
    # Import minimal bot components
    from config import Config
    
    logger.info("Core packages loaded successfully")
except ImportError as e:
    logger.error(f"Missing required package: {e}")
    sys.exit(1)

class DeploymentBot:
    """Minimal bot for deployment"""
    
    def __init__(self):
        self.config = Config()
        self.app = None
        
    async def start_command(self, update, context):
        """Handle /start command"""
        await update.message.reply_text(
            "🤖 TradeAI Companion Bot\n\n"
            "I'm your AI-powered trading assistant. Use /help to see available commands."
        )
    
    async def help_command(self, update, context):
        """Handle /help command"""
        help_text = """
📋 **Available Commands:**

/start - Start the bot
/help - Show this help message
/price SYMBOL - Get current price (e.g., /price AAPL)
/analyze SYMBOL - Basic market analysis

Bot is running in minimal mode for deployment.
        """
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def price_command(self, update, context):
        """Handle /price command"""
        if not context.args:
            await update.message.reply_text("Please provide a symbol. Example: /price AAPL")
            return
        
        symbol = context.args[0].upper()
        
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info
            current_price = info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))
            
            response = f"💰 **{symbol} Price**\n"
            response += f"Current: ${current_price}\n"
            response += f"Market: {info.get('market', 'US')}"
            
            await update.message.reply_text(response, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Price error: {e}")
            await update.message.reply_text(f"Error fetching price for {symbol}")
    
    async def analyze_command(self, update, context):
        """Handle /analyze command"""
        if not context.args:
            await update.message.reply_text("Please provide a symbol. Example: /analyze AAPL")
            return
        
        symbol = context.args[0].upper()
        
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            response = f"📊 **{symbol} Analysis**\n\n"
            response += f"**Price:** ${info.get('currentPrice', 'N/A')}\n"
            response += f"**52 Week High:** ${info.get('fiftyTwoWeekHigh', 'N/A')}\n"
            response += f"**52 Week Low:** ${info.get('fiftyTwoWeekLow', 'N/A')}\n"
            response += f"**Market Cap:** ${info.get('marketCap', 'N/A'):,}\n"
            response += f"**P/E Ratio:** {info.get('trailingPE', 'N/A')}"
            
            await update.message.reply_text(response, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            await update.message.reply_text(f"Error analyzing {symbol}")
    
    async def handle_message(self, update, context):
        """Handle regular messages"""
        message = update.message.text
        
        # Simple response
        response = "I understand you're asking about trading. Try commands like:\n"
        response += "/price AAPL - Get Apple stock price\n"
        response += "/analyze MSFT - Analyze Microsoft stock"
        
        await update.message.reply_text(response)
    
    async def health_check(self, request):
        """Health check endpoint"""
        return web.Response(text="OK", status=200)
    
    async def start_health_server(self):
        """Start health check server"""
        app = web.Application()
        app.router.add_get('/health', self.health_check)
        app.router.add_get('/', lambda r: web.Response(text="Bot is running"))
        
        runner = web.AppRunner(app)
        await runner.setup()
        
        port = int(os.environ.get('PORT', 5000))
        site = web.TCPSite(runner, '0.0.0.0', port)
        await site.start()
        
        logger.info(f"Health server started on port {port}")
        return runner
    
    async def run(self):
        """Run the bot"""
        # Skip database for minimal deployment
        logger.info("Running in minimal mode without database")
        
        # Start health server
        health_runner = await self.start_health_server()
        
        # Create telegram application
        telegram_token = self.config.TELEGRAM_BOT_TOKEN
        if not telegram_token:
            logger.error("No TELEGRAM_API_TOKEN found in environment!")
            raise ValueError("TELEGRAM_API_TOKEN is required")
        
        self.app = Application.builder().token(telegram_token).build()
        
        # Add handlers
        self.app.add_handler(CommandHandler("start", self.start_command))
        self.app.add_handler(CommandHandler("help", self.help_command))
        self.app.add_handler(CommandHandler("price", self.price_command))
        self.app.add_handler(CommandHandler("analyze", self.analyze_command))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        # Start bot
        logger.info("Starting minimal deployment bot...")
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling()
        
        logger.info("Bot is running! Press Ctrl+C to stop.")
        
        # Keep running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping bot...")
        finally:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
            await health_runner.cleanup()

async def main():
    """Main entry point"""
    bot = DeploymentBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())