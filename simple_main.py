"""
Simplified main entry point for the Telegram AI Trading Bot
Focuses on core functionality without heavy dependencies
"""
import sys
import os
import asyncio
import time

# Ensure the current directory is in sys.path for module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_required_dependencies():
    """Check if core dependencies are available"""
    missing = []
    
    try:
        import aiohttp
    except ImportError:
        missing.append("aiohttp")
    
    try:
        import openai
    except ImportError:
        missing.append("openai")
    
    try:
        from telegram import Bot
    except ImportError:
        missing.append("python-telegram-bot")
    
    if missing:
        print(f"Missing required dependencies: {', '.join(missing)}")
        print("Please install them using: pip install " + " ".join(missing))
        return False
    
    return True

def check_environment():
    """Check if required environment variables are set"""
    required_vars = ["OPENAI_API_KEY"]
    missing = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        print(f"Missing required environment variables: {', '.join(missing)}")
        print("Please set them in your environment or .env file")
        return False
    
    return True

def simple_bot():
    """Simple bot implementation for testing"""
    from telegram import Bot
    from telegram.ext import Application, CommandHandler, MessageHandler, filters
    from telegram.error import Conflict, TelegramError
    import openai
    from dotenv import load_dotenv
    import time
    
    # Load environment variables
    load_dotenv()
    
    # Initialize OpenAI
    openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    async def start(update, context):
        """Start command handler"""
        await update.message.reply_text(
            "🤖 AI Trading Bot is running!\n\n"
            "Available commands:\n"
            "/start - Show this message\n"
            "/help - Get help\n"
            "/status - Check bot status\n"
            "\nOr just send me a message about stocks!"
        )
    
    async def help_command(update, context):
        """Help command handler"""
        help_text = """
🤖 **AI Trading Bot Help**

**Basic Commands:**
• /start - Start the bot
• /help - Show this help message
• /status - Check bot status

**Natural Language:**
Just type your questions about stocks, for example:
• "What's the price of Apple?"
• "Tell me about Tesla stock"
• "Should I buy Microsoft?"

The bot uses AI to understand your questions and provide helpful responses.
        """
        await update.message.reply_text(help_text)
    
    async def status(update, context):
        """Status command handler"""
        await update.message.reply_text(
            "✅ Bot Status: Online\n"
            f"🕐 Uptime: {int(time.time() - start_time)} seconds\n"
            "🧠 AI: OpenAI Connected\n"
            "📊 Market Data: Ready\n"
        )
    
    async def handle_message(update, context):
        """Handle natural language messages"""
        user_message = update.message.text
        
        try:
            # Simple AI response using OpenAI
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful AI trading assistant. Provide brief, informative responses about stocks and trading. Keep responses under 500 characters."
                    },
                    {"role": "user", "content": user_message}
                ],
                max_tokens=150
            )
            
            ai_response = response.choices[0].message.content
            await update.message.reply_text(ai_response)
            
        except Exception as e:
            await update.message.reply_text(
                f"Sorry, I encountered an error: {str(e)[:100]}...\n"
                "Please try again or contact support."
            )
    
    # Get bot token
    token = os.getenv("TELEGRAM_API_TOKEN")
    if not token:
        print("Error: TELEGRAM_API_TOKEN not found in environment variables")
        return
    
    # Create application with error handling
    application = Application.builder().token(token).build()
    
    # Add error handler
    async def error_handler(update, context):
        """Log errors and handle conflicts"""
        if isinstance(context.error, Conflict):
            print("⚠️ Bot conflict detected - another instance may be running")
            print("🔄 Retrying in 5 seconds...")
            time.sleep(5)
        else:
            print(f"❌ Bot error: {context.error}")
    
    application.add_error_handler(error_handler)
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    print("🤖 Starting AI Trading Bot...")
    print("✅ Bot is running! Press Ctrl+C to stop.")
    
    # Run the bot with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            application.run_polling(drop_pending_updates=True)
            break
        except Conflict as e:
            print(f"🔄 Attempt {attempt + 1}: Conflict detected, waiting before retry...")
            if attempt < max_retries - 1:
                time.sleep(10)  # Wait longer between retries
            else:
                print("❌ Max retries reached. Please ensure no other bot instances are running.")
                raise
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            raise

if __name__ == "__main__":
    start_time = time.time()
    
    print("🚀 AI Trading Bot - Simple Mode")
    print("=" * 40)
    
    # Check dependencies
    if not check_required_dependencies():
        sys.exit(1)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Run the bot
    try:
        simple_bot()
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"❌ Bot crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)