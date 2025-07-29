#!/usr/bin/env python3
"""
Minimal startup script without heavy dependencies
"""
import sys
import os
import asyncio
import logging
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MinimalBot')

class MinimalTradingBot:
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_API_TOKEN')
        if not self.bot_token:
            raise ValueError("TELEGRAM_API_TOKEN environment variable not set")
        self.application = None

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        await update.message.reply_text(
            "🤖 **AI Trading Bot** is online!\n\n"
            "Bot is running in minimal mode due to dependency issues.\n"
            "Type /help for available commands.",
            parse_mode='Markdown'
        )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = """
🤖 **AI Trading Bot - Help**

**Available Commands:**
• /start - Start the bot
• /help - Show this help message
• /status - Bot status

⚠️ **Note:** Bot is running in minimal mode.
Full features will be available once dependency issues are resolved.
"""
        await update.message.reply_text(help_text, parse_mode='Markdown')

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        await update.message.reply_text(
            "✅ Bot Status: **Online**\n"
            "🔧 Mode: **Minimal**\n"
            "⚠️ Issue: Dependency conflicts being resolved",
            parse_mode='Markdown'
        )

    def setup_handlers(self):
        """Setup basic command handlers"""
        app = self.application
        app.add_handler(CommandHandler("start", self.start_command))
        app.add_handler(CommandHandler("help", self.help_command))
        app.add_handler(CommandHandler("status", self.status_command))

    async def run(self):
        """Run the minimal bot"""
        try:
            logger.info("Starting Minimal AI Trading Bot...")
            
            # Create application
            self.application = Application.builder().token(self.bot_token).build()
            
            # Setup handlers
            self.setup_handlers()
            
            # Start bot
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling()
            
            logger.info("🤖 Minimal bot is running! Press Ctrl+C to stop.")
            
            # Keep running
            try:
                await asyncio.Event().wait()
            except KeyboardInterrupt:
                logger.info("Bot stopping...")
            finally:
                await self.application.stop()
                await self.application.shutdown()
                
        except Exception as e:
            logger.error(f"Bot crashed: {e}")
            raise

async def main():
    bot = MinimalTradingBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())
