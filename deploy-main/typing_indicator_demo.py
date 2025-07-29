#!/usr/bin/env python3
"""
Typing Indicator Demo Script
Demonstrates the enhanced typing indicator functionality in the Telegram bot.
"""

import asyncio
from telegram import Bot
from telegram.ext import ContextTypes
import os
from config import Config

class TypingIndicatorDemo:
    """Demo class to showcase typing indicator improvements"""
    
    def __init__(self):
        config = Config()
        self.bot_token = config.TELEGRAM_BOT_TOKEN
        self._typing_tasks = {}
    
    async def _send_typing_indicator(self, chat_id: int, bot: Bot) -> None:
        """Send a single typing indicator"""
        try:
            await bot.send_chat_action(chat_id=chat_id, action="typing")
            print(f"✅ Typing indicator sent to chat {chat_id}")
        except Exception as e:
            print(f"❌ Error sending typing indicator to chat {chat_id}: {e}")
    
    async def _start_persistent_typing(self, chat_id: int, bot: Bot) -> str:
        """Start a persistent typing indicator that refreshes every 4 seconds"""
        task_id = f"typing_{chat_id}_{asyncio.get_event_loop().time()}"
        
        async def typing_loop():
            try:
                count = 0
                while task_id in self._typing_tasks and count < 5:  # Demo: max 5 iterations
                    await self._send_typing_indicator(chat_id, bot)
                    print(f"🔄 Typing indicator refreshed (iteration {count + 1})")
                    await asyncio.sleep(4)  # Refresh every 4 seconds
                    count += 1
                print(f"⏹️ Typing loop completed for chat {chat_id}")
            except asyncio.CancelledError:
                print(f"🛑 Typing loop cancelled for chat {chat_id}")
            except Exception as e:
                print(f"❌ Error in typing loop for chat {chat_id}: {e}")
        
        # Start the typing task
        task = asyncio.create_task(typing_loop())
        self._typing_tasks[task_id] = task
        print(f"🚀 Started persistent typing for chat {chat_id} (task_id: {task_id})")
        
        return task_id
    
    async def _stop_persistent_typing(self, task_id: str) -> None:
        """Stop a persistent typing indicator"""
        if task_id in self._typing_tasks:
            task = self._typing_tasks.pop(task_id)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            print(f"🛑 Stopped persistent typing (task_id: {task_id})")
    
    async def demo_enhanced_typing(self, chat_id: int):
        """Demonstrate the enhanced typing indicator system"""
        print("\n🎯 Enhanced Typing Indicator Demo")
        print("=" * 40)
        
        bot = Bot(token=self.bot_token)
        
        # Demo 1: Single typing indicator
        print("\n📍 Demo 1: Single typing indicator")
        await self._send_typing_indicator(chat_id, bot)
        
        # Demo 2: Persistent typing indicator
        print("\n📍 Demo 2: Persistent typing indicator (20 seconds)")
        task_id = await self._start_persistent_typing(chat_id, bot)
        
        # Simulate long operation
        print("⏳ Simulating long operation...")
        await asyncio.sleep(20)
        
        # Stop typing
        await self._stop_persistent_typing(task_id)
        
        print("\n✅ Demo completed!")
        print("\n📋 Key Features Demonstrated:")
        print("• Immediate typing indicator response")
        print("• Persistent typing for long operations")
        print("• Automatic refresh every 4 seconds")
        print("• Clean task management and cleanup")
        print("• Error handling and logging")

async def main():
    """Main demo function"""
    print("🤖 Telegram Bot Enhanced Typing Indicator Demo")
    print("=" * 50)
    
    # Note: This is a demo script - in real usage, chat_id would come from user interaction
    demo_chat_id = input("Enter a chat ID for demo (or press Enter to skip): ").strip()
    
    if demo_chat_id:
        try:
            chat_id = int(demo_chat_id)
            demo = TypingIndicatorDemo()
            await demo.demo_enhanced_typing(chat_id)
        except ValueError:
            print("❌ Invalid chat ID format")
        except Exception as e:
            print(f"❌ Demo error: {e}")
    else:
        print("\n📝 Demo skipped - no chat ID provided")
        print("\n💡 To test with a real chat:")
        print("1. Start a conversation with your bot")
        print("2. Send a message to get your chat ID")
        print("3. Run this script again with that chat ID")
    
    print("\n🎉 Enhanced typing indicator features:")
    print("• ⚡ Immediate response - typing shows instantly")
    print("• 🔄 Persistent typing - refreshes every 4 seconds")
    print("• 🎯 Smart management - automatic cleanup")
    print("• 🛡️ Error handling - graceful failure recovery")
    print("• 📊 Consistent UX - works for all command types")

if __name__ == "__main__":
    asyncio.run(main())