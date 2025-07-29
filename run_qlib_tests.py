import unittest
import sys
import logging
from datetime import datetime
from test_qlib_integration import TestQlibIntegration
from logger import logger

def run_tests():
    # Configure logging
    logger.setLevel(logging.INFO)
    
    # Start time
    start_time = datetime.now()
    logger.info(f'Starting Qlib integration tests at {start_time}')
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestQlibIntegration)
    
    # Run tests
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    
    # End time
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Log results
    logger.info(f'Tests completed at {end_time}')
    logger.info(f'Total duration: {duration}')
    logger.info(f'Tests run: {result.testsRun}')
    logger.info(f'Tests passed: {result.testsRun - len(result.failures) - len(result.errors)}')
    
    if result.failures:
        logger.error('Failed tests:')
        for test, traceback in result.failures:
            logger.error(f'\n{test}\n{traceback}')
    
    if result.errors:
        logger.error('Test errors:')
        for test, traceback in result.errors:
            logger.error(f'\n{test}\n{traceback}')
    
    # Return True if all tests passed
    return len(result.failures) == 0 and len(result.errors) == 0

def main():
    logger.info('Starting Qlib data analysis and validation')
    
    try:
        # Run tests
        tests_passed = run_tests()
        
        if tests_passed:
            logger.info('All Qlib data validation tests passed')
            logger.info('Data analysis pipeline is ready for user requests')
            sys.exit(0)
        else:
            logger.error('Some tests failed - check logs for details')
            sys.exit(1)
            
    except Exception as e:
        logger.error(f'Error running tests: {e}')
        sys.exit(1)

if __name__ == '__main__':
    main()