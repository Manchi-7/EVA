#!/usr/bin/env python3
"""
Collection Module Runner
Runs the adversarial popup benchmark to collect successful attacks.

Usage:
    python -m collection.run [options]
    
Or from project root:
    python run_collection.py [options]

Examples:
    # Default (GLM-4.5-flash attacker, GLM-4V-flash victim)
    python run_collection.py

    # Use GPT-4 as attacker, GLM-4V as victim
    python run_collection.py --attacker gpt-4 --victim glm-4v-flash

    # Use Qwen models
    python run_collection.py --attacker qwen-plus --victim qwen-vl-plus

    # Enable evolution
    python run_collection.py --evolution
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from collection.config import Config, Colors
from collection.benchmark_runner import BenchmarkRunner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run adversarial popup benchmark to collect successful attacks"
    )
    parser.add_argument(
        "--attacker",
        type=str,
        default="glm-4.5-flash",
        help="Attacker/Evolution LLM model (default: glm-4.5-flash)"
    )
    parser.add_argument(
        "--victim",
        type=str,
        default="glm-4v-flash",
        help="Victim VLM model (default: glm-4v-flash)"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of trials per (task, seed) pair (default: 3)"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=2,
        help="Success threshold (default: 2 out of trials)"
    )
    parser.add_argument(
        "--evolution",
        action="store_true",
        help="Enable attack evolution for failed attacks"
    )
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Disable LLM-based intent judgment"
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=None,
        help="Specific scenarios to run (default: all)"
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Log file path (e.g., collection.log). If not specified, logs only to console"
    )
    parser.add_argument(
        "--start-task",
        type=str,
        default=None,
        help="Start from a specific task ID (e.g., T3). Skips all tasks before this."
    )
    parser.add_argument(
        "--start-seed",
        type=str,
        default=None,
        help="Start from a specific seed ID (e.g., S5). Only used with --start-task."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Import model utilities
    from models import get_llm_provider, get_llm_api_config, get_vlm_provider, get_vlm_api_config
    
    print(f"\n{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}   ADVERSARIAL POPUP COLLECTION{Colors.ENDC}")
    print(f"{'='*60}\n")
    
    # Create config
    config = Config()
    
    # Determine providers from model names
    attacker_provider = get_llm_provider(args.attacker)
    victim_provider = get_vlm_provider(args.victim)
    
    attacker_api = get_llm_api_config(attacker_provider)
    victim_api = get_vlm_api_config(victim_provider)
    
    # Store model info in config for downstream use
    config.attacker_model = args.attacker
    config.attacker_provider = attacker_provider.value
    config.attacker_api_key = attacker_api["api_key"]
    config.attacker_base_url = attacker_api.get("base_url")
    
    config.victim_model = args.victim
    config.victim_provider = victim_provider.value
    config.victim_api_key = victim_api["api_key"]
    config.victim_base_url = victim_api.get("base_url")
    
    # Update experiment settings
    config.trials_per_pair = args.trials
    config.success_threshold = args.threshold
    
    # Set output directory based on model combination
    config.output_dir = str(Path(config.output_dir).parent / f"{args.attacker}_{args.victim}")
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    # Also create images subdirectory
    (Path(config.output_dir) / "images").mkdir(parents=True, exist_ok=True)
    
    print(Colors.info(f"Configuration:"))
    print(f"  Attacker model: {args.attacker} ({attacker_provider.value})")
    print(f"  Victim model: {args.victim} ({victim_provider.value})")
    print(f"  Trials per pair: {args.trials}")
    print(f"  Success threshold: {args.threshold}/{args.trials}")
    print(f"  Evolution: {'enabled' if args.evolution else 'disabled'}")
    print(f"  LLM Judge: {'disabled' if args.no_llm_judge else 'enabled'}")
    print(f"  Output dir: {config.output_dir}")
    
    if args.scenarios:
        print(f"  Scenarios: {args.scenarios}")
    
    # Setup logging to file if specified
    log_file = None
    if args.log:
        import logging
        from datetime import datetime
        
        # Create log file path
        log_path = Path(args.log)
        if not log_path.suffix:
            log_path = log_path.with_suffix('.log')
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup file logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(log_path, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        log_file = log_path
        print(f"  Log file: {log_path}")
    
    print()
    
    # Initialize runner
    runner = BenchmarkRunner(
        config=config,
        enable_evolution=args.evolution,
        use_llm_judge=not args.no_llm_judge
    )
    
    try:
        # Run benchmark (with optional resume point)
        runner.run_benchmark(
            start_task=args.start_task,
            start_seed=args.start_seed
        )
        
        print(f"\n{Colors.GREEN}Collection complete!{Colors.ENDC}")
        print(f"Results saved to: {config.output_dir}")
        
    except KeyboardInterrupt:
        print(Colors.warning("\nCollection interrupted by user"))
    except Exception as e:
        print(Colors.error(f"Collection failed: {e}"))
        import traceback
        traceback.print_exc()
        raise
    finally:
        runner.close()


if __name__ == "__main__":
    main()
