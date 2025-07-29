import difflib
import re
from typing import List, Dict, Optional, Tuple
from logger import logger

class EnhancedErrorHandler:
    """
    Enhanced error handling system that provides:
    - Helpful suggestions when commands fail
    - "Did you mean...?" suggestions for typos
    - Examples when users enter invalid formats
    """
    
    def __init__(self):
        # Common stock symbols for suggestions
        self.common_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'PYPL', 'UBER', 'LYFT',
            'SNAP', 'TWTR', 'SPOT', 'SQ', 'ROKU', 'ZM', 'DOCU', 'SHOP',
            'SPY', 'QQQ', 'VTI', 'IWM', 'DIA', 'XLF', 'XLK', 'XLE',
            'BTC-USD', 'ETH-USD', 'ADA-USD', 'DOT-USD'
        ]
        
        # Valid command patterns
        self.command_patterns = {
            'price': r'^/price\s+[A-Z]{1,5}(-USD)?$',
            'chart': r'^/chart\s+[A-Z]{1,5}(-USD)?\s*(1D|1W|1M|3M|6M|1Y|2Y|5Y)?$',
            'analyze': r'^/analyze\s+[A-Z]{1,5}(-USD)?$',
            'alert': r'^/alert\s+[A-Z]{1,5}(-USD)?\s+(above|below)\s+\d+(\.\d+)?$',
            'trade': r'^/trade\s+(buy|sell)\s+[A-Z]{1,5}(-USD)?\s+\d+\s+\d+(\.\d+)?$'
        }
        
        # Command examples
        self.command_examples = {
            'price': [
                '/price AAPL',
                '/price TSLA',
                '/price BTC-USD',
                '/price SPY'
            ],
            'chart': [
                '/chart AAPL',
                '/chart TSLA 1M',
                '/chart NVDA 6M',
                '/chart SPY 1Y'
            ],
            'analyze': [
                '/analyze AAPL',
                '/analyze TSLA',
                '/analyze tech'
            ],
            'alert': [
                '/alert AAPL above 150',
                '/alert TSLA below 200',
                '/alert BTC-USD above 50000',
                '/alert SPY below 400'
            ],
            'trade': [
                '/trade buy AAPL 10 150',
                '/trade sell TSLA 5 250',
                '/trade buy SPY 20 400',
                '/trade sell NVDA 3 500'
            ]
        }
        
        # Common error types and their solutions
        self.error_solutions = {
            'invalid_symbol': {
                'title': '❌ Invalid Stock Symbol',
                'reasons': [
                    'Symbol not found in US exchanges',
                    'Incorrect symbol format',
                    'Company may have been delisted'
                ],
                'solutions': [
                    'Use valid US stock symbols (e.g., AAPL, TSLA)',
                    'Check symbol spelling on financial websites',
                    'Try popular symbols like SPY, QQQ for ETFs'
                ]
            },
            'invalid_format': {
                'title': '❌ Invalid Command Format',
                'reasons': [
                    'Missing required parameters',
                    'Incorrect parameter order',
                    'Invalid parameter values'
                ],
                'solutions': [
                    'Check command syntax with /help',
                    'Use examples provided below',
                    'Ensure all parameters are included'
                ]
            },
            'api_error': {
                'title': '⚠️ Service Temporarily Unavailable',
                'reasons': [
                    'API quota exceeded',
                    'Network connectivity issues',
                    'External service downtime'
                ],
                'solutions': [
                    'Try again in a few minutes',
                    'Use alternative commands',
                    'Check if markets are open'
                ]
            },
            'data_error': {
                'title': '📊 Data Not Available',
                'reasons': [
                    'Insufficient historical data',
                    'Market closed or pre-market',
                    'Symbol recently listed'
                ],
                'solutions': [
                    'Try during market hours (9:30 AM - 4:00 PM ET)',
                    'Use more established stocks',
                    'Try again later when more data is available'
                ]
            }
        }
    
    def suggest_similar_symbol(self, invalid_symbol: str, threshold: float = 0.6) -> Optional[str]:
        """
        Suggest similar stock symbols for typos
        
        Args:
            invalid_symbol: The invalid symbol entered by user
            threshold: Similarity threshold (0.0 to 1.0)
            
        Returns:
            Best matching symbol or None
        """
        invalid_symbol = invalid_symbol.upper().replace('-USD', '')
        
        # Find closest matches
        matches = difflib.get_close_matches(
            invalid_symbol, 
            self.common_stocks, 
            n=3, 
            cutoff=threshold
        )
        
        return matches[0] if matches else None
    
    def suggest_command_correction(self, invalid_command: str) -> Optional[Tuple[str, str]]:
        """
        Suggest command corrections for typos
        
        Args:
            invalid_command: The invalid command entered by user
            
        Returns:
            Tuple of (suggested_command, example) or None
        """
        # Extract command name
        command_match = re.match(r'^/([a-zA-Z_]+)', invalid_command)
        if not command_match:
            return None
            
        command_name = command_match.group(1).lower()
        
        # Find closest command match
        available_commands = list(self.command_patterns.keys())
        matches = difflib.get_close_matches(command_name, available_commands, n=1, cutoff=0.6)
        
        if matches:
            suggested_command = matches[0]
            example = self.command_examples[suggested_command][0]
            return (suggested_command, example)
            
        return None
    
    def validate_command_format(self, command: str) -> Dict[str, any]:
        """
        Validate command format and provide specific feedback
        
        Args:
            command: The command to validate
            
        Returns:
            Dictionary with validation results
        """
        command = command.strip()
        
        # Check if it's a command
        if not command.startswith('/'):
            return {
                'valid': False,
                'error_type': 'not_command',
                'message': 'Commands must start with /',
                'suggestion': 'Try /help to see available commands'
            }
        
        # Extract command name
        parts = command.split()
        if not parts:
            return {
                'valid': False,
                'error_type': 'empty_command',
                'message': 'Empty command',
                'suggestion': 'Try /help to see available commands'
            }
        
        command_name = parts[0][1:].lower()  # Remove / and convert to lowercase
        
        # Check if command exists
        if command_name not in self.command_patterns:
            suggestion = self.suggest_command_correction(command)
            if suggestion:
                return {
                    'valid': False,
                    'error_type': 'unknown_command',
                    'message': f'Unknown command: /{command_name}',
                    'suggestion': f'Did you mean /{suggestion[0]}? Example: {suggestion[1]}'
                }
            else:
                return {
                    'valid': False,
                    'error_type': 'unknown_command',
                    'message': f'Unknown command: /{command_name}',
                    'suggestion': 'Try /help to see available commands'
                }
        
        # Validate command format
        pattern = self.command_patterns[command_name]
        if not re.match(pattern, command, re.IGNORECASE):
            return {
                'valid': False,
                'error_type': 'invalid_format',
                'command': command_name,
                'message': f'Invalid format for /{command_name}',
                'examples': self.command_examples[command_name]
            }
        
        return {'valid': True}
    
    def format_error_message(self, error_type: str, context: Dict = None) -> str:
        """
        Format a comprehensive error message
        
        Args:
            error_type: Type of error (invalid_symbol, invalid_format, etc.)
            context: Additional context like invalid symbol, command, etc.
            
        Returns:
            Formatted error message
        """
        if error_type not in self.error_solutions:
            return "❌ An unexpected error occurred. Please try again or contact support."
        
        error_info = self.error_solutions[error_type]
        
        message = f"{error_info['title']}\n\n"
        
        # Add specific context
        if context:
            if 'invalid_symbol' in context:
                symbol = context['invalid_symbol']
                suggestion = self.suggest_similar_symbol(symbol)
                if suggestion:
                    message += f"💡 **Did you mean '{suggestion}'?**\n\n"
            
            if 'invalid_command' in context:
                command = context['invalid_command']
                suggestion = self.suggest_command_correction(command)
                if suggestion:
                    message += f"💡 **Did you mean '/{suggestion[0]}'?**\n"
                    message += f"Example: `{suggestion[1]}`\n\n"
        
        # Add reasons
        message += "**Possible reasons:**\n"
        for reason in error_info['reasons']:
            message += f"• {reason}\n"
        
        message += "\n**Try this:**\n"
        for solution in error_info['solutions']:
            message += f"• {solution}\n"
        
        # Add examples if available
        if context and 'command' in context:
            command = context['command']
            if command in self.command_examples:
                message += "\n**Examples:**\n"
                for example in self.command_examples[command][:3]:
                    message += f"• `{example}`\n"
        
        return message
    
    def handle_command_error(self, command: str, error: Exception = None) -> str:
        """
        Handle command errors with intelligent suggestions
        
        Args:
            command: The command that failed
            error: The exception that occurred (optional)
            
        Returns:
            Formatted error message with suggestions
        """
        try:
            # First validate the command format
            validation = self.validate_command_format(command)
            
            if not validation['valid']:
                if validation['error_type'] == 'invalid_format':
                    context = {
                        'command': validation['command'],
                        'examples': validation['examples']
                    }
                    message = self.format_error_message('invalid_format', context)
                    message += "\n**Correct examples:**\n"
                    for example in validation['examples'][:3]:
                        message += f"• `{example}`\n"
                    return message
                else:
                    return f"{validation['message']}\n\n💡 {validation['suggestion']}"
            
            # If format is valid but command failed, analyze the error
            if error:
                error_str = str(error).lower()
                
                if any(keyword in error_str for keyword in ['symbol', 'ticker', 'not found']):
                    # Extract symbol from command
                    parts = command.split()
                    if len(parts) > 1:
                        symbol = parts[1].upper()
                        return self.format_error_message('invalid_symbol', {'invalid_symbol': symbol})
                
                elif any(keyword in error_str for keyword in ['quota', 'rate limit', 'api']):
                    return self.format_error_message('api_error')
                
                elif any(keyword in error_str for keyword in ['data', 'insufficient', 'empty']):
                    return self.format_error_message('data_error')
            
            # Generic error message
            return self.format_error_message('api_error')
            
        except Exception as e:
            logger.error(f"Error in enhanced error handler: {e}")
            return "❌ An unexpected error occurred. Please try again or use /help for assistance."
    
    def get_help_for_command(self, command_name: str) -> str:
        """
        Get specific help for a command
        
        Args:
            command_name: Name of the command
            
        Returns:
            Help message for the command
        """
        if command_name not in self.command_examples:
            return "Command not found. Use /help to see all available commands."
        
        examples = self.command_examples[command_name]
        
        help_text = f"**/{command_name.upper()} Command Help:**\n\n"
        help_text += "**Examples:**\n"
        
        for example in examples:
            help_text += f"• `{example}`\n"
        
        # Add specific tips based on command
        if command_name == 'price':
            help_text += "\n**Tips:**\n"
            help_text += "• Use stock symbols like AAPL, TSLA, MSFT\n"
            help_text += "• For crypto, add -USD (e.g., BTC-USD)\n"
            help_text += "• ETFs work too (SPY, QQQ, VTI)\n"
        
        elif command_name == 'chart':
            help_text += "\n**Time periods:**\n"
            help_text += "• 1D, 1W, 1M, 3M, 6M, 1Y, 2Y, 5Y\n"
            help_text += "• Default is 3M if not specified\n"
        
        elif command_name == 'alert':
            help_text += "\n**Tips:**\n"
            help_text += "• Use 'above' or 'below' for direction\n"
            help_text += "• Price can be decimal (e.g., 150.50)\n"
            help_text += "• You'll get notified when price hits target\n"
        
        return help_text