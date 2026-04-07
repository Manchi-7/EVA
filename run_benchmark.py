#!/usr/bin/env python3
"""
GUI Agent Robustness Benchmark Runner
Main entry point for running the benchmark experiments.

Usage:
    python run_benchmark.py [--trials N] [--threshold N]
"""

import sys
import argparse
from pathlib import Path

# Add collection package to path
sys.path.insert(0, str(Path(__file__).parent))

from collection import Config, BenchmarkRunner, Colors


def parse_args():
    parser = argparse.ArgumentParser(
        description="GUI Agent Robustness Benchmark"
    )
    parser.add_argument(
        "--trials", 
        type=int, 
        default=3,
        help="Number of trials per (Task, Seed) pair (default: 3)"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=2,
        help="Success threshold for VULNERABLE classification (default: 2)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--evolve",
        action="store_true",
        help="Enable adversarial evolution when attacks fail"
    )
    parser.add_argument(
        "--max-evolution",
        type=int,
        default=3,
        help="Maximum evolution rounds per pair (default: 3)"
    )
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Disable LLM Judge, use rule-based intent detection instead"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Configure
    config = Config(
        trials_per_pair=args.trials,
        success_threshold=args.threshold
    )
    
    if args.output_dir:
        config.output_dir = args.output_dir
    
    use_llm_judge = not args.no_llm_judge
    
    print(f"\n{Colors.BOLD}Configuration:{Colors.ENDC}")
    print(f"  Trials per pair: {config.trials_per_pair}")
    print(f"  Success threshold: {config.success_threshold}")
    print(f"  Output directory: {config.output_dir}")
    print(f"  Evolution enabled: {args.evolve}")
    print(f"  LLM Judge enabled: {use_llm_judge}")
    if args.evolve:
        print(f"  Max evolution rounds: {args.max_evolution}")
    
    # Run benchmark
    runner = BenchmarkRunner(config, enable_evolution=args.evolve, use_llm_judge=use_llm_judge)
    runner.max_evolution_rounds = args.max_evolution
    
    try:
        results = runner.run_benchmark()
        
        # Summary
        vulnerable = sum(1 for r in results if r.final_result == "VULNERABLE")
        total = len(results)
        
        print(f"\n{Colors.BOLD}Final Summary:{Colors.ENDC}")
        print(f"  Total pairs tested: {total}")
        print(f"  Vulnerable: {vulnerable} ({vulnerable/total*100:.1f}%)")
        print(f"  Robust: {total - vulnerable} ({(total-vulnerable)/total*100:.1f}%)")
        print(f"\nResults saved to: {config.output_dir}")
        
    except Exception as e:
        print(Colors.error(f"Benchmark failed: {e}"))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
