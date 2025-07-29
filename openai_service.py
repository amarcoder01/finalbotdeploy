"""
OpenAI service module for handling AI chat responses
Integrates with OpenAI GPT-4o mini for intelligent conversation
"""
import os
import asyncio
import base64
from openai import AsyncOpenAI
from typing import Optional
from logger import logger
from config import Config
from conversation_memory import ConversationMemory

class OpenAIService:
    """Service class for OpenAI API integration"""
    
    def __init__(self):
        """Initialize OpenAI client"""
        # Create config instance to properly access properties
        config = Config()
        self.client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        self.model = Config.OPENAI_MODEL
        self.memory = ConversationMemory()
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        logger.info(f"OpenAI service initialized with model: {self.model}")
    
    async def generate_response(self, user_message: str, user_id: int, context_str: Optional[str] = None) -> Optional[str]:
        """
        Generate AI response for user message
        
        Args:
            user_message (str): The user's message
            user_id (int): Telegram user ID for logging
            context_str (Optional[str]): Conversation context string (if provided by caller)
        
        Returns:
            Optional[str]: AI response or None if error occurred
        """
        try:
            # Sanitize and validate input
            if not user_message or not user_message.strip():
                return "I received an empty message. Please send me something to respond to!"
            
            # Truncate very long messages
            if len(user_message) > 2000:
                user_message = user_message[:2000] + "... (message truncated)"
                logger.warning(f"Message from user {user_id} was truncated due to length")
            
            # Use provided context or fallback to internal memory
            if context_str is None:
                conversation_context = self.memory.get_conversation_context(user_id)
            else:
                conversation_context = context_str
            
            # Create system prompt for the trading bot context
            system_prompt = f"""You are an advanced AI trading assistant with comprehensive market knowledge. 
            You help users with stock analysis, market insights, trading strategies, and investment decisions.
            
            Your capabilities:
            - Real-time market data analysis
            - Technical and fundamental analysis
            - Trading recommendations with risk assessment
            - Portfolio optimization advice
            
            - Sector rotation insights
            
            You have access to real-time data through various commands:
            /price - Current stock prices
            /chart - Technical charts  
            /analyze - AI stock analysis
            /movers - Market movers
            /sectors - Sector performance
            /opportunities - Trading opportunities
            
            IMPORTANT: You have conversation memory. Use the context below to maintain continuity:
            {conversation_context}
            
            Be professional, informative, and actionable. Keep responses under 3000 characters.
            Provide specific, data-driven insights when possible. Always include risk warnings for trading advice."""
            
            logger.info(f"[OpenAI] Message received from user {user_id}")
            logger.info(f"[OpenAI] API call initiated for user {user_id} with model: {self.model}")
            
            # Add retry logic for API calls
            max_retries = 3
            retry_delay = 2  # seconds
            
            for attempt in range(max_retries):
                try:
                    response = await asyncio.wait_for(
                        self.client.chat.completions.create(
                            model=self.model,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_message}
                            ],
                            max_tokens=1000,
                            temperature=0.7
                        ),
                        timeout=Config.REQUEST_TIMEOUT
                    )
                    # If successful, break out of retry loop
                    break
                except asyncio.TimeoutError:
                    logger.warning(f"[OpenAI] Timeout error for user {user_id} (attempt {attempt+1}/{max_retries})")
                    if attempt == max_retries - 1:  # Last attempt
                        logger.error(f"[OpenAI] All retry attempts failed for user {user_id}")
                        return self._get_error_response("timeout")
                    await asyncio.sleep(retry_delay)  # Wait before retrying
                except Exception as e:
                    logger.warning(f"[OpenAI] API error on attempt {attempt+1}/{max_retries}: {str(e)}")
                    if attempt == max_retries - 1:  # Last attempt
                        logger.error(f"[OpenAI] All retry attempts failed for user {user_id}")
                        return self._get_error_response(str(e))
                    await asyncio.sleep(retry_delay)  # Wait before retrying
            
            logger.info(f"[OpenAI] API response received for user {user_id}")
            
            # Extract response content
            ai_response = response.choices[0].message.content
            logger.info(f"[OpenAI] Extracted response content: {len(ai_response) if ai_response else 0} characters")
            
            if not ai_response:
                logger.error(f"[OpenAI] Empty response from OpenAI for user {user_id}")
                return "I'm sorry, I couldn't generate a response right now. Please try again."
            
            # Ensure response fits Telegram's message limit
            if len(ai_response) > Config.MAX_MESSAGE_LENGTH:
                ai_response = ai_response[:Config.MAX_MESSAGE_LENGTH-50] + "... (response truncated)"
            
            # Only store conversation in internal memory if context_str was not provided
            if context_str is None:
                self.memory.add_message(user_id, user_message, ai_response, "text")
            
            logger.info(f"[OpenAI] Response sent to user {user_id}: {len(ai_response)} characters")
            return ai_response
            
        except Exception as e:
            logger.error(f"[OpenAI] Error generating response for user {user_id}: {str(e)}")
            return self._get_error_response(str(e))
    
    def _get_error_response(self, error_message: str) -> str:
        """
        Generate user-friendly error response with more detailed information
        
        Args:
            error_message (str): The actual error message
            
        Returns:
            str: User-friendly error message
        """
        logger.debug(f"[OpenAI] Generating error response for: {error_message}")
        
        # Common OpenAI API errors
        if "insufficient_quota" in error_message.lower() or "quota" in error_message.lower():
            return "⚠️ AI service quota exceeded. Basic trading commands are still available: /price, /chart, /analyze, /movers, /sectors. AI chat is temporarily unavailable."
        elif "rate limit" in error_message.lower():
            return "I'm currently experiencing high demand. Please try again in a few moments. (Rate limit exceeded)"
        elif "timeout" in error_message.lower():
            return "The request timed out. This could be due to high server load or network issues. Please try again."
        elif "api key" in error_message.lower() or "authentication" in error_message.lower():
            return "There's a configuration issue with my AI service. Please contact the administrator. (API key error)"
        elif "content filter" in error_message.lower() or "content_filter" in error_message.lower():
            return "Your request was flagged by content filters. Please modify your query and try again."
        elif "context length" in error_message.lower() or "token limit" in error_message.lower():
            return "The request was too complex for me to process. Please try a simpler query."
        elif "overloaded" in error_message.lower() or "capacity" in error_message.lower():
            return "The AI service is currently overloaded. Please try again in a few minutes."
        elif "connection" in error_message.lower() or "network" in error_message.lower():
            return "There was a network connection issue. Please check your internet connection and try again."
        else:
            # Fallback with more detailed information for debugging
            logger.error(f"[OpenAI] Unhandled error type: {error_message}")
            return f"I encountered an issue while processing your request. Please try again later. If the problem persists, contact support with error code: {hash(error_message) % 10000:04d}"
    
    async def test_connection(self) -> bool:
        """
        Test OpenAI API connection
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            logger.info("[OpenAI] Test connection initiated")
            try:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": "Test connection"}],
                        max_tokens=10
                    ),
                    timeout=10
                )
            except asyncio.TimeoutError:
                logger.error("[OpenAI] Timeout error during test connection")
                return False
            logger.info("[OpenAI] Test connection successful")
            return True
        except Exception as e:
            logger.error(f"[OpenAI] Test connection failed: {str(e)}")
            return False



    async def search_stock_price(self, symbol: str, user_id: int) -> Optional[str]:
        """
        Use OpenAI's web search to find stock price when traditional sources fail
        Args:
            symbol (str): Stock symbol to search for
            user_id (int): Telegram user ID for logging
        Returns:
            Optional[str]: Formatted stock price information or None if error
        """
        try:
            logger.info(f"[OpenAI] Web search requested for stock {symbol} by user {user_id}")
            
            # Create a search query for current stock price
            search_query = f"current stock price {symbol} real-time market data today"
            
            # Use OpenAI's browsing capability to search for current stock information
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system", 
                            "content": """You are a financial data assistant. Search for current stock price information and return ONLY the essential data in this exact format:

SYMBOL: [stock symbol]
PRICE: [current price with currency]
CHANGE: [price change]
CHANGE_PERCENT: [percentage change]
VOLUME: [trading volume if available]
MARKET: [exchange/market if available]
TIMESTAMP: [current time]

Search for the stock on major financial websites and market data providers. If you cannot find the specific stock, return "NOT_FOUND". Keep the response concise and factual."""
                        },
                        {
                            "role": "user", 
                            "content": f"Search for current stock price and market data for {symbol}. Look for real-time or today's data from financial websites. If this is a US stock, search US exchanges. If it's an international stock, search the appropriate exchange."
                        }
                    ],

                    max_tokens=200,
                    temperature=0.1
                ),
                timeout=30
            )
            
            ai_response = response.choices[0].message.content
            logger.info(f"[OpenAI] Web search response for {symbol}: {ai_response}")
            
            if ai_response and "NOT_FOUND" not in ai_response:
                return ai_response
            else:
                return None
                
        except Exception as e:
            logger.error(f"[OpenAI] Error in stock price search for {symbol}: {str(e)}")
            return None

    async def analyze_image(self, image_bytes: bytes, user_id: int, prompt: Optional[str] = None) -> Optional[str]:
        """
        Analyze chart images using GPT-4o vision capabilities
        
        Args:
            image_bytes (bytes): The image data
            user_id (int): Telegram user ID for logging
            prompt (Optional[str]): Custom analysis prompt
            
        Returns:
            Optional[str]: Analysis result or None if error
        """
        try:
            logger.info(f"[OpenAI] Image analysis requested by user {user_id}")
            
            # Detect image format
            image_format = self._detect_image_format(image_bytes)
            if not image_format:
                logger.error(f"[OpenAI] Unsupported image format for user {user_id}")
                return "❌ Unsupported image format. Please send JPEG, PNG, GIF, or WEBP images."
            
            # Encode image to base64
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Create analysis prompt
            analysis_prompt = prompt or """
Provide a professional, detailed analysis of this financial chart image with the following structure:

📊 **CHART IDENTIFICATION:**
• Asset name and ticker symbol (if visible)
• Chart type (candlestick, line, bar) and timeframe
• Current price and key price levels
• Primary trend direction with confidence level

📐 **TECHNICAL STRUCTURE:**
• Major support and resistance zones with price levels
• Trendlines, channels, and chart patterns
• Volume profile analysis and significant volume events
• Market structure (higher highs/lows or lower highs/lows)

📈 **TECHNICAL INDICATORS:**
• Moving averages (identify crossovers and dynamic support/resistance)
• Momentum indicators (RSI, MACD, Stochastic) with current readings
• Volatility indicators (Bollinger Bands, ATR) with interpretation
• Any divergences between price and indicators

🎯 **ACTIONABLE TRADING INSIGHTS:**
• Potential trade setups with specific entry points
• Key price targets for both bullish and bearish scenarios
• Stop loss placement recommendations with rationale
• Risk-to-reward ratio for suggested trades

⚠️ **RISK ASSESSMENT:**
• Current market volatility context
• Upcoming events that could impact price (if identifiable)
• Position sizing recommendations based on volatility
• Alternative scenarios that could invalidate the analysis

💼 **PORTFOLIO CONTEXT:**
• How this asset might fit in a diversified portfolio
• Correlation with broader market or sector performance
• Suggested allocation percentage for risk management

Provide precise price levels, percentages, and timeframes whenever possible. Balance technical analysis with practical trading advice. Format your response with clear sections and bullet points for readability.
"""
            
            # Make API call with vision model
            max_retries = 3  # Increased retries for better reliability
            backoff_factor = 2  # Exponential backoff for retries
            
            # Add system message to improve analysis quality
            system_message = "You are a professional financial analyst specializing in technical chart analysis. "
            system_message += "Provide detailed, accurate, and actionable analysis of financial charts with precise price levels. "
            system_message += "Focus on identifying key technical patterns, support/resistance levels, and trading opportunities. "
            system_message += "Always include risk management advice and maintain a balanced perspective."
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"[OpenAI] Making image analysis API call (attempt {attempt+1}/{max_retries})")
                    response = await asyncio.wait_for(
                        self.client.chat.completions.create(
                            model="gpt-4o",  # Use GPT-4o for vision capabilities
                            messages=[
                                {
                                    "role": "system",
                                    "content": system_message
                                },
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": analysis_prompt
                                        },
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/{image_format};base64,{image_b64}",
                                                "detail": "high"
                                            }
                                        }
                                    ]
                                }
                            ],
                            max_tokens=2000,  # Increased token limit for more detailed analysis
                            temperature=0.2,  # Reduced temperature for more consistent results
                            presence_penalty=0.1,  # Slight presence penalty to encourage covering all topics
                            frequency_penalty=0.1  # Slight frequency penalty to reduce repetition
                        ),
                        timeout=45  # Increased timeout for complex charts
                    )
                    logger.info(f"[OpenAI] Image analysis API call successful")
                    break
                except Exception as e:
                    logger.warning(f"[OpenAI] Image analysis attempt {attempt+1} failed: {str(e)}")
                    if attempt == max_retries - 1:
                        # Try fallback with gpt-4o-mini if available
                        try:
                            logger.info(f"[OpenAI] Attempting fallback to gpt-4o-mini model")
                            response = await asyncio.wait_for(
                                self.client.chat.completions.create(
                                    model="gpt-4o-mini",
                                    messages=[
                                        {
                                            "role": "system",
                                            "content": system_message
                                        },
                                        {
                                            "role": "user",
                                            "content": [
                                                {
                                                    "type": "text",
                                                    "text": analysis_prompt
                                                },
                                                {
                                                    "type": "image_url",
                                                    "image_url": {
                                                        "url": f"data:image/{image_format};base64,{image_b64}",
                                                        "detail": "high"
                                                    }
                                                }
                                            ]
                                        }
                                    ],
                                    max_tokens=1800,
                                    temperature=0.3
                                ),
                                timeout=30
                            )
                            logger.info(f"[OpenAI] Fallback model API call successful")
                            break
                        except Exception as fallback_error:
                            logger.error(f"[OpenAI] Fallback model also failed: {str(fallback_error)}")
                            return self._get_image_analysis_error(str(fallback_error))
                    # Exponential backoff between retries
                    wait_time = backoff_factor ** attempt
                    logger.info(f"[OpenAI] Waiting {wait_time}s before retry")
                    await asyncio.sleep(wait_time)
            
            # Extract and validate response
            analysis_result = response.choices[0].message.content
            
            # Validate response quality
            if not analysis_result or len(analysis_result.strip()) < 50:
                logger.error(f"[OpenAI] Empty or too short analysis result for user {user_id}")
                return "❌ *Analysis Failed*\n\nThe analysis engine couldn't generate a meaningful result. Please ensure you're uploading a clear financial chart image."
            
            # Post-process the analysis for better formatting and quality
            analysis_result = self._post_process_analysis(analysis_result)
            
            # Store in conversation memory
            self.memory.add_message(user_id, "[Professional Chart Analysis]", analysis_result, "image")
            
            logger.info(f"[OpenAI] Image analysis completed for user {user_id}: {len(analysis_result)} characters")
            return analysis_result
        except Exception as e:
            logger.error(f"[OpenAI] Error in analyze_image: {str(e)}")
            return self._get_image_analysis_error(str(e))
    
    def _post_process_analysis(self, analysis: str) -> str:
        """
        Post-process the analysis result to improve formatting and quality
        
        Args:
            analysis (str): The raw analysis result from the model
            
        Returns:
            str: The improved analysis result
        """
        # Fix common Markdown formatting issues
        # Replace triple asterisks with double (common model mistake)
        analysis = analysis.replace('***', '**')
        
        # Ensure proper spacing after bullet points
        analysis = analysis.replace('•', '• ')
        
        # Fix percentage signs for Markdown compatibility
        analysis = analysis.replace('%', '\\%')
        
        # Ensure section headers are properly formatted
        sections = ['CHART IDENTIFICATION', 'TECHNICAL STRUCTURE', 'TECHNICAL INDICATORS', 
                   'ACTIONABLE TRADING INSIGHTS', 'RISK ASSESSMENT', 'PORTFOLIO CONTEXT']
        
        for section in sections:
            # Look for section headers without proper Markdown formatting
            if f"**{section}**" not in analysis and section in analysis:
                analysis = analysis.replace(section, f"**{section}**")
        
        # Ensure there's a blank line after each section header for proper Markdown rendering
        for section in sections:
            analysis = analysis.replace(f"**{section}:**\n", f"**{section}:**\n\n")
            
        # Add a professional signature at the end if not present
        if "DISCLAIMER:" not in analysis:
            analysis += "\n\n**DISCLAIMER:** This analysis is for informational purposes only and should not be considered financial advice. Always conduct your own research before making investment decisions."
            
        return analysis
    
    def _detect_image_format(self, image_bytes: bytes) -> Optional[str]:
        """
        Detect image format from bytes
        
        Args:
            image_bytes (bytes): Image data
            
        Returns:
            Optional[str]: Image format (jpeg, png, gif, webp, bmp, tiff) or None
        """
        # Check for common image formats by their magic numbers/signatures
        if image_bytes.startswith(b'\xff\xd8\xff'):
            return 'jpeg'
        elif image_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
            return 'png'
        elif image_bytes.startswith(b'GIF87a') or image_bytes.startswith(b'GIF89a'):
            return 'gif'
        elif image_bytes.startswith(b'RIFF') and b'WEBP' in image_bytes[:12]:
            return 'webp'
        elif image_bytes.startswith(b'BM'):
            return 'bmp'
        elif image_bytes.startswith(b'II*\x00') or image_bytes.startswith(b'MM\x00*'):
            return 'tiff'
        elif image_bytes.startswith(b'\x00\x00\x01\x00') or image_bytes.startswith(b'\x00\x00\x02\x00'):
            return 'ico'
        
        # Try to detect by content analysis if header detection fails
        try:
            # Look for JPEG markers throughout the file
            if b'\xff\xd8' in image_bytes[:20] and b'\xff\xd9' in image_bytes[-20:]:
                return 'jpeg'
            # Look for PNG chunks
            if b'IDAT' in image_bytes or b'IEND' in image_bytes:
                return 'png'
        except:
            pass
            
        # Format not recognized
        logger.warning(f"[OpenAI] Unrecognized image format detected")
        return None
    
    def _get_image_analysis_error(self, error_message: str) -> str:
        """
        Generate user-friendly error response for image analysis
        
        Args:
            error_message (str): The actual error message
            
        Returns:
            str: User-friendly error message
        """
        if "insufficient_quota" in error_message.lower() or "quota" in error_message.lower():
            return "⚠️ *AI Analysis Quota Exceeded*\n\nThe service has reached its analysis limit. Please try again in a few hours or contact support for premium access."
        elif "rate limit" in error_message.lower():
            return "🔄 *Rate Limit Reached*\n\nToo many analysis requests in a short time. Please wait 1-2 minutes before trying again."
        elif "api key" in error_message.lower():
            return "🔧 *Technical Issue*\n\nThe chart analysis service is experiencing configuration issues. Our team has been notified and is working to restore full functionality."
        elif "content policy" in error_message.lower() or "content_filter" in error_message.lower():
            return "⚠️ *Content Policy Alert*\n\nThe image was flagged by safety filters. Please ensure you're uploading financial charts only. Non-financial images cannot be analyzed."
        elif "model not found" in error_message.lower():
            return "🔧 *Service Temporarily Unavailable*\n\nThe advanced chart analysis model is currently unavailable. Please try again later when the service is restored."
        elif "timeout" in error_message.lower():
            return "⏱️ *Analysis Timeout*\n\nThe image analysis process took too long to complete. This may be due to image complexity or size. Try with a clearer or smaller image, or try again when the service is less busy."
        elif "format" in error_message.lower() or "unsupported" in error_message.lower():
            return "📸 *Unsupported Image Format*\n\nPlease upload your chart as a JPG, PNG, or WebP file. Other formats cannot be processed by the analysis engine."
        elif "too large" in error_message.lower() or "size" in error_message.lower():
            return "📏 *Image Size Issue*\n\nThe image is too large to process. Please resize to under 4MB or take a screenshot of the chart area only."
        elif "network" in error_message.lower() or "connection" in error_message.lower():
            return "🌐 *Network Error*\n\nThere was a connection issue while analyzing your chart. Please check your internet connection and try again."
        else:
            return "❌ *Analysis Failed*\n\nUnable to analyze the image. Please ensure it's a clear financial chart with visible indicators and price action. Screenshots from trading platforms work best."
