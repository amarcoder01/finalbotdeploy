#!/usr/bin/env python3
"""
Clean bot startup that avoids problematic dependencies
"""
import sys
import os
import asyncio
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TradingBot')

# Add current directory to path  
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set environment variables
os.environ['SKIP_DATABASE'] = 'true'
os.environ['DISABLE_ADVANCED_FEATURES'] = 'true'

logger.info("=== Starting Trading Bot (Clean Mode) ===")

# Import only the core components we need
from config import Config
from security_config import secure_logger
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler

# Import the OpenAI service
from openai_service import OpenAIService

# Create a simplified telegram handler
class SimpleTelegramHandler:
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_API_TOKEN')
        if not self.bot_token:
            raise ValueError("TELEGRAM_API_TOKEN not set")
        
        self.openai_service = OpenAIService()
        self.application = None
        logger.info("SimpleTelegramHandler initialized")
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        welcome_msg = (
            "🤖 Welcome to TradeAI Companion Bot!\n\n"
            "I'm your AI-powered trading assistant. Here's what I can do:\n\n"
            "📊 Market Analysis\n"
            "💬 Natural Language Chat\n"
            "📈 Price Information\n\n"
            "Type /help to see all commands or just chat with me!"
        )
        await update.message.reply_text(welcome_msg)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = (
            "📚 Available Commands:\n\n"
            "/start - Welcome message\n"
            "/help - Show this help\n"
            "/price <symbol> - Get price info\n\n"
            "Or just chat with me naturally!"
        )
        await update.message.reply_text(help_text)
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages"""
        user_message = update.message.text
        logger.info(f"Received message: {user_message[:50]}...")
        
        try:
            # Use OpenAI to generate response
            response = await self.openai_service.get_trading_insights(user_message)
            await update.message.reply_text(response)
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await update.message.reply_text("Sorry, I encountered an error. Please try again.")
    
    async def price_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /price command"""
        if not context.args:
            await update.message.reply_text("Please provide a symbol. Example: /price AAPL")
            return
        
        symbol = context.args[0].upper()
        await update.message.reply_text(f"Price information for {symbol} is currently unavailable in clean mode.")
    
    def setup_handlers(self):
        """Setup command handlers"""
        app = self.application
        
        # Command handlers
        app.add_handler(CommandHandler("start", self.start_command))
        app.add_handler(CommandHandler("help", self.help_command))
        app.add_handler(CommandHandler("price", self.price_command))
        
        # Message handler
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        logger.info("Handlers setup complete")
    
    async def run(self):
        """Run the bot"""
        logger.info("Creating Telegram application...")
        
        # Create application
        self.application = Application.builder().token(self.bot_token).build()
        
        # Setup handlers
        self.setup_handlers()
        
        # Start the bot
        logger.info("Starting bot polling...")
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()
        
        logger.info("✅ Bot is running! Send /start to your bot to begin.")
        
        # Keep running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()

async def main():
    """Main function"""
    try:
        handler = SimpleTelegramHandler()
        await handler.run()
    except Exception as e:
        logger.error(f"Bot failed: {e}")
        raise

if __name__ == "__main__":
    logger.info("Starting asyncio event loop...")
    asyncio.run(main())