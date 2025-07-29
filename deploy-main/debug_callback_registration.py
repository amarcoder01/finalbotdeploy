#!/usr/bin/env python3
"""
Debug script to test callback handler registration
"""

import asyncio
import logging
from telegram import Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CallbackQueryHandler, CommandHandler
from config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleCallbackTest:
    def __init__(self):
        self.config = Config()
        self.bot_token = self.config.TELEGRAM_BOT_TOKEN
        self.application = None
    
    async def simple_callback_handler(self, update: Update, context):
        """Simple callback handler for testing"""
        query = update.callback_query
        await query.answer()
        
        print(f"\n🎉 CALLBACK RECEIVED! 🎉")
        print(f"User ID: {update.effective_user.id}")
        print(f"Callback Data: {query.data}")
        print(f"Message ID: {query.message.message_id}")
        logger.info(f"CALLBACK SUCCESS: {query.data} from user {update.effective_user.id}")
        
        # Respond to the callback
        await query.edit_message_text(
            f"✅ **Callback Working!**\n\n"
            f"🔹 You clicked: `{query.data}`\n"
            f"🔹 User ID: `{update.effective_user.id}`\n"
            f"🔹 Time: {query.message.date}\n\n"
            f"🎯 **Result:** Inline keyboards are working correctly!",
            parse_mode='Markdown'
        )
    
    async def test_command(self, update: Update, context):
        """Send test message with inline keyboard"""
        keyboard = [
            [InlineKeyboardButton("✅ Test 1", callback_data="test_1")],
            [InlineKeyboardButton("🔥 Test 2", callback_data="test_2")],
            [InlineKeyboardButton("📊 Test 3", callback_data="test_3")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "🧪 **CALLBACK REGISTRATION TEST**\n\n"
            "Click any button below to test if callbacks work:\n\n"
            "If you see a success message, callbacks are working!\n"
            "If nothing happens, there's an issue with callback handling.",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
        
        print(f"\n📤 Test message sent to user {update.effective_user.id}")
        logger.info(f"Test message sent to user {update.effective_user.id}")
    
    async def setup_and_run(self):
        """Setup the test bot and run it"""
        try:
            # Create application
            builder = Application.builder().token(self.bot_token)
            self.application = builder.build()
            
            # Add handlers
            self.application.add_handler(CommandHandler("test_callback", self.test_command))
            self.application.add_handler(CallbackQueryHandler(self.simple_callback_handler))
            
            print("\n" + "="*60)
            print("🧪 CALLBACK REGISTRATION TEST BOT")
            print("="*60)
            print("\n📋 Instructions:")
            print("   1. Send /test_callback to the bot")
            print("   2. Click any of the buttons that appear")
            print("   3. Watch this console for callback messages")
            print("\n🔍 What to look for:")
            print("   - '🎉 CALLBACK RECEIVED! 🎉' message")
            print("   - Callback data and user info")
            print("   - Success message in Telegram")
            print("\n" + "="*60 + "\n")
            
            # Initialize and start
            await self.application.initialize()
            await self.application.updater.start_polling()
            await self.application.start()
            
            logger.info("🚀 Test bot is running! Send /test_callback to test callbacks.")
            print("🚀 Test bot is running! Send /test_callback to test callbacks.")
            
            # Keep running
            try:
                await asyncio.Event().wait()
            except KeyboardInterrupt:
                print("\n🛑 Test bot stopped by user")
            finally:
                await self.application.stop()
                await self.application.shutdown()
                
        except Exception as e:
            logger.error(f"❌ Error running test bot: {e}")
            print(f"❌ Error running test bot: {e}")

if __name__ == "__main__":
    test_bot = SimpleCallbackTest()
    asyncio.run(test_bot.setup_and_run())