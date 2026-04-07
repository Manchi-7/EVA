#!/usr/bin/env python3
"""
Rule Quality Evaluation Runner
Evaluates distilled attack rules by generating new attacks and testing them.

Usage:
    python run_evaluation.py [options]
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run from evaluation module
from evaluation.run import main

if __name__ == "__main__":
    main()
