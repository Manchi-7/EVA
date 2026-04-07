#!/usr/bin/env python3
"""
Collection Module Entry Point
Runs the adversarial popup benchmark to collect successful attacks.

Usage:
    python run_collection.py [options]

Examples:
    # Default (GLM-4.5-flash attacker, GLM-4V-flash victim)
    python run_collection.py

    # Use GPT-4 as attacker, GPT-4V as victim
    python run_collection.py --attacker gpt-4 --victim gpt-4-vision-preview

    # Use Qwen models
    python run_collection.py --attacker qwen-plus --victim qwen-vl-plus
"""

from collection.run import main

if __name__ == "__main__":
    main()
