#!/usr/bin/env python3
"""
Rule Distillation Runner
Extracts attack rules from successful adversarial injections using GPT-5.

Usage:
    python run_distillation.py [--output FILENAME] [--input FILEPATH] [--api-key KEY]
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run from distillation module
from distillation.run import main

if __name__ == "__main__":
    main()
