#!/usr/bin/env python3
"""
Rule Distillation Runner
Extracts attack rules from successful adversarial injections.

Usage:
    python -m distillation.run [options]
    
Or from project root:
    python distillation/run.py [options]
    
Examples:
    # Default (uses gpt-5-chat for distillation)
    python run_distillation.py
    
    # Distill from collection folder
    python run_distillation.py --input collection/gpt-5-nano_glm-4v-flash
    
    # Specify model for distillation
    python run_distillation.py --model gpt-5-nano --input collection/gpt-5-nano_glm-4v-flash
    
    # Full example with all parameters
    python run_distillation.py --model gpt-5-nano --input collection/gpt-5-nano_glm-4v-flash --output output/my_rules.json
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from distillation import RuleDistiller, DistillationConfig
from distillation.config import Colors


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distill attack rules from successful adversarial injections"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename for distilled rules (default: auto-generated with model names)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input file path for successful attacks (default: output/successful_attacks.jsonl)"
    )
    parser.add_argument(
        "--attacker",
        type=str,
        default=None,
        help="Attacker model used in collection (default: infer from input path or glm-4.5-flash)"
    )
    parser.add_argument(
        "--victim",
        type=str,
        default=None,
        help="Victim model used in collection (default: infer from input path or glm-4v-flash)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-chat",
        help="LLM model to use for distillation (default: gpt-5-chat)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"\n{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}   ATTACK RULE DISTILLATION{Colors.ENDC}")
    print(f"{'='*60}\n")
    
    # Create config
    config = DistillationConfig()
    
    # Handle input path
    if args.input:
        input_path = Path(args.input)
        if input_path.is_dir():
            input_path = input_path / "successful_attacks.jsonl"
        config.successful_attacks_file = str(input_path)

    # Determine attacker and victim models
    attacker = args.attacker
    victim = args.victim
    
    # Try to infer from input path if not specified
    if config.successful_attacks_file and (not attacker or not victim):
        try:
            # Assuming folder structure: .../attacker_victim/filename
            parent_name = Path(config.successful_attacks_file).parent.name
            if "_" in parent_name:
                parts = parent_name.split("_")
                if len(parts) >= 2:
                    if not attacker:
                        attacker = parts[0]
                    if not victim:
                        victim = "_".join(parts[1:])
        except Exception:
            pass
            
    # Set defaults if still None
    if not attacker:
        attacker = "glm-4.5-flash"
    if not victim:
        victim = "glm-4v-flash"

    config.attacker_model = attacker
    config.victim_model = victim
    
    # Set distillation model
    if args.model:
        config.model = args.model
    
    # Generate output filename if not specified
    if args.output:
        output_filename = args.output
    else:
        output_filename = config.get_output_filename()
    
    print(Colors.info(f"Configuration:"))
    print(f"  Distillation model: {config.model}")
    print(f"  Source attacker: {config.attacker_model}")
    print(f"  Source victim: {config.victim_model}")
    print(f"  Input: {config.successful_attacks_file}")
    print(f"  Output: {output_filename}")
    
    # Initialize distiller
    distiller = RuleDistiller(config)
    
    # Load attacks (use config path which has been processed to handle directories)
    attacks = distiller.load_successful_attacks(Path(config.successful_attacks_file))
    
    if not attacks:
        print(Colors.error("No successful attacks found."))
        print(Colors.info("Run the collection first: python run_collection.py"))
        sys.exit(1)
    
    # Analyze distribution
    stats = distiller.analyze_evolution_strategies(attacks)
    
    total = stats['total']
    print(f"\n{Colors.BOLD}Attack Distribution:{Colors.ENDC}")
    print(f"  Total successful attacks: {total}")
    print(f"  No evolution needed: {stats['no_evolution']} ({stats['no_evolution']/total*100:.1f}%)")
    print(f"  Boost Trust strategy: {stats['boost_trust']} ({stats['boost_trust']/total*100:.1f}%)")
    print(f"  Boost Urgency strategy: {stats['boost_urgency']} ({stats['boost_urgency']/total*100:.1f}%)")
    print(f"  Mixed strategies: {stats['mixed']} ({stats['mixed']/total*100:.1f}%)")
    
    print(f"\n  By Scenario:")
    for scenario, count in sorted(stats['by_scenario'].items()):
        print(f"    {scenario}: {count} ({count/total*100:.1f}%)")
    
    # Distill rules
    print(f"\n{Colors.BOLD}Distilling Rules with {config.model}...{Colors.ENDC}")
    rules = distiller.distill_rules(attacks)
    
    if rules.get("universal_rules") or rules.get("scenario_rules"):
        # Print rules
        distiller.print_rules(rules)
        
        # Save rules
        output_path = distiller.save_rules(rules, output_filename)
        
        print(f"\n{Colors.SUCCESS}Distillation complete!{Colors.ENDC}")
        print(f"Rules saved to: {output_path}")
    else:
        print(Colors.error("Failed to distill rules"))
        sys.exit(1)


if __name__ == "__main__":
    main()
