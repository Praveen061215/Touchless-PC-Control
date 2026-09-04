"""
run_tests.py
------------
Discovers and executes all unit tests in the tests/ directory.
Usage:
    py run_tests.py
"""

import sys
import unittest

def main():
    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

if __name__ == "__main__":
    main()
