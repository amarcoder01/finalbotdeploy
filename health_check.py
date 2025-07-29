#!/usr/bin/env python3
"""
Health Check Script for TradeAI Companion Bot
Verifies that the deployment is working correctly
"""

import asyncio
import aiohttp
import sys
import os
import json
from datetime import datetime

class HealthChecker:
    """Health check utility for the deployed bot"""
    
    def __init__(self, base_url=None):
        self.base_url = base_url or os.getenv('RENDER_EXTERNAL_URL', 'http://localhost:10000')
        self.endpoints = {
            'root': '/',
            'health': '/health',
            'ready': '/ready',
            'metrics': '/metrics'
        }
    
    async def check_endpoint(self, session, name, path):
        """Check a single endpoint"""
        url = f"{self.base_url}{path}"
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                status = response.status
                content = await response.text()
                
                return {
                    'endpoint': name,
                    'url': url,
                    'status': status,
                    'success': 200 <= status < 300,
                    'response_length': len(content),
                    'content_preview': content[:200] if content else None
                }
        except asyncio.TimeoutError:
            return {
                'endpoint': name,
                'url': url,
                'status': 'timeout',
                'success': False,
                'error': 'Request timed out'
            }
        except Exception as e:
            return {
                'endpoint': name,
                'url': url,
                'status': 'error',
                'success': False,
                'error': str(e)
            }
    
    async def run_health_check(self):
        """Run comprehensive health check"""
        print(f"🔍 Running health check for: {self.base_url}")
        print(f"⏰ Timestamp: {datetime.now().isoformat()}")
        print("=" * 60)
        
        results = []
        
        async with aiohttp.ClientSession() as session:
            # Check all endpoints
            for name, path in self.endpoints.items():
                print(f"Checking {name} endpoint...")
                result = await self.check_endpoint(session, name, path)
                results.append(result)
                
                # Print result
                status_icon = "✅" if result['success'] else "❌"
                print(f"  {status_icon} {name}: {result['status']}")
                
                if not result['success'] and 'error' in result:
                    print(f"    Error: {result['error']}")
                elif result['success'] and result.get('content_preview'):
                    print(f"    Response: {result['content_preview']}")
        
        # Summary
        successful = sum(1 for r in results if r['success'])
        total = len(results)
        
        print("=" * 60)
        print(f"📊 Health Check Summary:")
        print(f"   Successful: {successful}/{total}")
        print(f"   Success Rate: {(successful/total)*100:.1f}%")
        
        if successful == total:
            print("🎉 All health checks passed! Deployment is healthy.")
            return True
        else:
            print("⚠️  Some health checks failed. Please review the deployment.")
            return False
    
    def print_deployment_info(self):
        """Print deployment information"""
        print("\n📋 Deployment Information:")
        print(f"   Base URL: {self.base_url}")
        print(f"   Environment: {os.getenv('ENVIRONMENT', 'unknown')}")
        print(f"   Python Path: {os.getenv('PYTHONPATH', 'not set')}")
        print(f"   Port: {os.getenv('PORT', 'not set')}")
        print(f"   Timezone: {os.getenv('TZ', 'not set')}")
        
        # Check for required environment variables
        required_vars = ['TELEGRAM_API_TOKEN', 'OPENAI_API_KEY']
        print("\n🔑 Required Environment Variables:")
        for var in required_vars:
            value = os.getenv(var)
            status = "✅ Set" if value else "❌ Missing"
            masked_value = f"{value[:8]}..." if value and len(value) > 8 else "Not set"
            print(f"   {var}: {status} ({masked_value})")
        
        # Check for optional environment variables
        optional_vars = ['DATABASE_URL', 'REDIS_URL', 'ALPACA_API_KEY']
        print("\n🔧 Optional Environment Variables:")
        for var in optional_vars:
            value = os.getenv(var)
            status = "✅ Set" if value else "⚪ Not set"
            print(f"   {var}: {status}")

async def main():
    """Main health check function"""
    # Parse command line arguments
    base_url = None
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    checker = HealthChecker(base_url)
    
    # Print deployment info
    checker.print_deployment_info()
    
    # Run health check
    success = await checker.run_health_check()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    print("🏥 TradeAI Companion Bot - Health Check")
    print("Usage: python health_check.py [base_url]")
    print("Example: python health_check.py https://your-app.onrender.com")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Health check interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Health check failed with error: {e}")
        sys.exit(1)