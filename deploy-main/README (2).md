# 🤖 AI-Powered Telegram Trading Bot

A comprehensive Telegram bot powered by OpenAI's GPT-4o mini that provides intelligent trading insights, real-time market data, and dynamic chart generation with natural language processing capabilities.

## ✨ Features

### 🧠 **Natural Language Processing**
- **Smart Intent Detection**: Understands requests like "Show me Apple's chart" or "What's Tesla trading at?"
- **Context-Aware Responses**: Maintains conversation history and provides relevant suggestions
- **Dual Interface**: Supports both command-based (`/chart AAPL`) and natural language interactions

### 📊 **Advanced Chart Generation**
- **Professional Charts**: High-quality candlestick charts via Chart-IMG API
- **Technical Indicators**: RSI, MACD, moving averages, volume overlays
- **Multiple Timeframes**: 1 minute to 1 year data with customizable intervals
- **Fallback System**: Matplotlib backup for chart generation

### 💰 **Real-Time Market Data**
- **Live Stock Prices**: Current prices, changes, volume, market cap
- **Portfolio Tracking**: Real portfolio data via Alpaca API
- **Market Movers**: Top gainers and losers
- **Sector Analysis**: Performance across different sectors

### 🤖 **AI-Powered Analysis**
- **Stock Analysis**: AI-driven recommendations and price targets
- **Risk Assessment**: Comprehensive risk analysis for trading decisions

- **Trading Intelligence**: Advanced AI insights for better trading decisions

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Telegram Bot Token
- OpenAI API Key
- Alpaca API credentials (optional for portfolio features)
- Chart IMG API key (optional for enhanced charts)

### Installation

```bash
git clone https://github.com/amar01vidality/botdevelpm.git
cd botdevelpm
pip install python-telegram-bot openai yfinance matplotlib pandas numpy python-dotenv asyncio aiohttp
```

### Environment Setup

Set up environment variables (you can create a `.env` file or set them directly):

```env
# Required
TELEGRAM_API_TOKEN=your_telegram_bot_token
OPENAI_API_KEY=your_openai_api_key

# Optional (for enhanced features)
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_API_SECRET=your_alpaca_secret_key
CHART_IMG_API_KEY=your_chart_img_api_key
```

### Running the Bot

```bash
python main.py
```

## 🔧 API Keys Setup

### 1. Telegram Bot Token
1. Message [@BotFather](https://t.me/botfather) on Telegram
2. Send `/newbot` and follow instructions
3. Copy the token to your `.env` file

### 2. OpenAI API Key
1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create a new API key
3. Add to your environment variables

### 3. Alpaca API (Optional)
1. Sign up at [Alpaca Markets](https://alpaca.markets/)
2. Generate API keys from dashboard
3. Add to your environment variables

### 4. Chart IMG API (Optional)
1. Visit [Chart IMG](https://chart-img.com/)
2. Get your API key
3. Add to your environment variables

## 💬 Usage Examples

### Command-Based Interface
```
/start - Initialize the bot
/help - Show all available commands
/price AAPL - Get Apple's current stock price
/chart TSLA 1d - Generate Tesla's daily chart
/analyze MSFT - Get AI analysis for Microsoft
/movers - Top market gainers and losers
/portfolio - Your portfolio summary

```

### Natural Language Interface
```
"Show me Apple's chart"
"What's Tesla's current price?"
"Analyze Microsoft stock"
"Get today's market movers"
"How is my portfolio doing?"
"Show me tech stocks performance"
"Suggest some example questions about charts"

```

## 🏗️ Architecture

The bot follows a modular architecture with clear separation of concerns:

- **`main.py`** - Application entry point and orchestration
- **`telegram_handler.py`** - Telegram Bot API integration
- **`openai_service.py`** - OpenAI GPT-4o mini integration
- **`market_data_service.py`** - Real-time market data (yfinance + Alpaca)
- **`chart_service.py`** - Dynamic chart generation
- **`trading_intelligence.py`** - AI-powered trading analysis
- **`natural_language_processor.py`** - Intent detection and NLP
- **`conversation_memory.py`** - Session management and context
- **`config.py`** - Configuration and environment management
- **`logger.py`** - Structured logging system

## 🔒 Security

- All API keys stored as environment variables
- No sensitive data in source code
- Secure token handling for Telegram and external APIs
- Error handling prevents API key exposure

## 🌐 Cross-Platform Compatibility

This bot works across multiple development environments:

- **Local Development**: Any Python IDE (VS Code, PyCharm, etc.)
- **Cloud Platforms**: GitHub Codespaces, Google Colab
- **Package Managers**: pip, conda, poetry

## 📊 Supported Features

### Market Data
- Real-time stock prices
- Historical price data
- Volume and market cap information
- Technical indicators (RSI, MACD, SMA, EMA)
- Market movers and sector performance

### Chart Types
- Candlestick charts
- Line charts
- Volume overlays
- Technical indicator plots
- Comparison charts
- Portfolio allocation charts

### AI Capabilities
- Stock analysis and recommendations
- Risk assessment

- Trading opportunity detection
- Natural language understanding
- Context-aware conversations

## 🛠️ Development

### Project Structure
```
├── main.py                     # Application entry point
├── telegram_handler.py         # Telegram integration
├── openai_service.py          # OpenAI integration
├── market_data_service.py     # Market data APIs
├── chart_service.py           # Chart generation
├── trading_intelligence.py    # AI trading analysis
├── natural_language_processor.py # NLP and intent detection
├── conversation_memory.py     # Session management
├── config.py                  # Configuration
├── logger.py                  # Logging system
├── setup.py                   # Package setup
├── config.py                 # Configuration management
├── logger.py                 # Logging system
└── README.md                 # This file
```

### Adding New Features
1. Create service modules following the existing pattern
2. Add configuration in `config.py`
3. Register handlers in `telegram_handler.py`
4. Update natural language patterns in `natural_language_processor.py`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.

## 🆘 Support

For support and questions:
- Open an issue on GitHub
- Check the logs in `bot_errors.log` for debugging
- Ensure all API keys are properly configured

## 🔗 Related APIs

- [Telegram Bot API](https://core.telegram.org/bots/api)
- [OpenAI API](https://platform.openai.com/docs)
- [Alpaca Markets API](https://alpaca.markets/docs/)
- [Chart IMG API](https://chart-img.com/docs)
- [Yahoo Finance API](https://pypi.org/project/yfinance/)

---

**Built with ❤️ for traders who want intelligent market insights through conversational AI**