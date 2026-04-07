#!/usr/bin/env python3
"""
Rule Quality Evaluation Runner
Evaluates distilled attack rules by generating new attacks and testing them.

Usage:
    python evaluation/run.py [options]
    
Or from project root:
    python run_evaluation.py [options]
    
Examples:
    # Default: all scenarios with universal + scenario rules
    python run_evaluation.py
    
    # Use GPT-4 as generator, GPT-4V as victim
    python run_evaluation.py --generator gpt-4 --victim gpt-4o
    
    # Use Qwen model
    python run_evaluation.py --generator qwen-plus --victim qwen-vl-plus
    
    # Single scenario with universal rules (from full rules file)
    python run_evaluation.py --scenarios amazon --rules output/rules.json
    
    # Single scenario without universal rules (from full rules file)
    python run_evaluation.py --scenarios amazon --rules output/rules.json --no-universal-rules
    
    # Single scenario using scenario-only rules file (no universal rules available)
    python run_evaluation.py --scenarios amazon --rules output/amazon_rules.json
    python run_evaluation.py --scenarios youtube --rules output/youtube_rules.json
    python run_evaluation.py --scenarios gmail --rules output/gmail_rules.json
    python run_evaluation.py --scenarios discord --rules output/discord_rules.json
    
    # Baseline: no rules experiment
    python run_evaluation.py --scenarios amazon --no-rules-experiment
    
    # Baseline: use seed attacks
    python run_evaluation.py --scenarios amazon --use-seeds
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation import EvaluationConfig, RuleBasedAttackGenerator, RuleQualityEvaluator
from evaluation.config import Colors


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate distilled attack rules by testing generated attacks"
    )
    parser.add_argument(
        "--rules",
        type=str,
        default=None,
        help="Path to rules JSON file (default: output/rules.json)"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Path to tasks JSON file (default: evaluation/tasks.json)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename for evaluation report (default: auto-generated with model names)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of attack samples to generate per rule (default: 3)"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of trials per attack sample (default: 3)"
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=None,
        help="Scenarios to evaluate (default: all)"
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate attacks, don't test them"
    )
    parser.add_argument(
        "--generator",
        type=str,
        default="glm-4.5-flash",
        help="Generator LLM model for creating attacks (default: glm-4.5-flash). Supports: glm-*, gpt-*, qwen*"
    )
    parser.add_argument(
        "--victim",
        type=str,
        default="glm-4v-flash",
        help="Victim VLM model being tested (default: glm-4v-flash). Supports: glm-4v-*, gpt-4o, gpt-4-vision-preview, qwen-vl-*"
    )
    parser.add_argument(
        "--no-universal-rules",
        action="store_true",
        help="Disable universal rules, only use scenario-specific rules"
    )
    parser.add_argument(
        "--no-rules-experiment",
        action="store_true",
        help="Baseline: generate attacks without any rules (only scenario + task goal context)"
    )
    parser.add_argument(
        "--use-seeds",
        action="store_true",
        help="Baseline: do not call generator LLM, use seeds JSON as attacks"
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Path to seeds JSON (default: collection/seeds.json)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"\n{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}   RULE QUALITY EVALUATION{Colors.ENDC}")
    print(f"{'='*60}\n")
    
    # Create config
    config = EvaluationConfig()
    
    # Override config with command line args
    if args.rules:
        config.rules_file = args.rules
    if args.tasks:
        config.tasks_file = args.tasks
    if args.scenarios:
        config.scenarios = args.scenarios
    
    config.samples_per_rule = args.samples
    config.trials_per_sample = args.trials
    
    # Set generator model
    config.generator_model = args.generator
    
    # Set victim model
    config.victim_model = args.victim
    
    # Set universal rules option
    config.use_universal_rules = not args.no_universal_rules

    # Control modes
    if args.use_seeds:
        config.control_mode = "seeds"
    elif args.no_rules_experiment:
        config.control_mode = "no_rules"
    else:
        config.control_mode = "rules"

    # Seeds path override
    if args.seeds:
        config.seeds_file = args.seeds
    
    # Generate output filename with model names if not specified
    if args.output:
        output_filename = args.output
    else:
        generator_name = args.generator.replace("-", "_")
        victim_name = args.victim.replace("-", "_")
        output_filename = f"evaluation_report_{generator_name}_{victim_name}.json"
    
    print(Colors.info(f"Configuration:"))
    print(f"  Rules file: {config.rules_path}")
    print(f"  Tasks file: {config.tasks_path}")
    print(f"  Samples per rule: {config.samples_per_rule}")
    print(f"  Trials per sample: {config.trials_per_sample}")
    print(f"  Scenarios: {config.scenarios}")
    print(f"  Universal rules: {'enabled' if config.use_universal_rules else 'disabled'}")
    print(f"  Generator model: {config.generator_model}")
    print(f"  Victim model: {config.victim_model}")
    print(f"  Control mode: {config.control_mode}")
    if config.control_mode == "seeds":
        print(f"  Seeds file: {config.seeds_path}")
    print(f"  Output: {output_filename}")
    
    if args.generate_only:
        # Only generate attacks
        print(f"\n{Colors.BOLD}Mode: Generate Only{Colors.ENDC}")
        
        generator = RuleBasedAttackGenerator(config)
        
        all_attacks = generator.generate_all(config.samples_per_rule)
        
        total = sum(len(attacks) for attacks in all_attacks.values())
        print(Colors.success(f"\nGenerated {total} total attacks"))
        
        attacks_filename = f"generated_attacks_{generator_name}_{victim_name}.json"
        output_path = generator.save_attacks(all_attacks, attacks_filename)
        print(f"Attacks saved to: {output_path}")
        
    else:
        # Full evaluation
        print(f"\n{Colors.BOLD}Mode: Full Evaluation{Colors.ENDC}")
        
        evaluator = RuleQualityEvaluator(config)
        
        report = evaluator.evaluate_all_rules(
            samples_per_rule=config.samples_per_rule,
            trials_per_sample=config.trials_per_sample
        )
        
        # Print and save report
        evaluator.print_report(report)
        output_path = evaluator.save_report(report, output_filename)
        
        print(f"\n{Colors.SUCCESS}Evaluation complete!{Colors.ENDC}")
        print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
