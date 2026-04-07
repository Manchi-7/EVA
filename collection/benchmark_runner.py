"""
Benchmark Runner Module
Main orchestrator for the robustness benchmark.
"""

import json
import time
from pathlib import Path
from typing import List, Tuple
from itertools import product

from .config import Config, Colors
from .data_types import (
    Task, AttackSeed, TrialResult, BenchmarkResult, 
    SuccessfulAttack, TrialOutcome
)
from .logger import BenchmarkLogger
from .environment import SeleniumEnvironment
from .victim import VictimAgent
from .evaluator import ActionEvaluator
from .attacker import AttackerLLM


def load_tasks(config: Config) -> List[Task]:
    """Load tasks from JSON file."""
    with open(config.tasks_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [Task.from_dict(t) for t in data['tasks']]


def load_seeds(config: Config) -> List[AttackSeed]:
    """Load attack seeds from JSON file."""
    with open(config.seeds_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [AttackSeed.from_dict(s) for s in data['seeds']]


class BenchmarkRunner:
    """
    Main benchmark runner that orchestrates the experiment.
    
    Iterates through Cartesian Product of Tasks × Seeds,
    running multiple trials per pair.
    """
    
    def __init__(self, config: Config = None, enable_evolution: bool = False, use_llm_judge: bool = True):
        self.config = config or Config()
        self.enable_evolution = enable_evolution
        self.max_evolution_rounds = 5  # Maximum evolution attempts per pair
        
        # Initialize components
        self.logger = BenchmarkLogger(self.config)
        self.environment = SeleniumEnvironment(self.config)
        self.victim = VictimAgent(self.config)
        self.evaluator = ActionEvaluator(self.config, use_llm_judge=use_llm_judge)
        
        # Initialize attacker LLM only if evolution is enabled
        self.attacker = AttackerLLM(self.config) if enable_evolution else None
        
        print(Colors.success(f"BenchmarkRunner initialized (evolution: {enable_evolution}, llm_judge: {use_llm_judge})"))
    
    def run_trial(
        self, 
        task: Task, 
        seed: AttackSeed, 
        trial_num: int
    ) -> TrialResult:
        """
        Run a single trial for a (Task, Seed) pair.
        
        Returns:
            TrialResult with outcome and details
        """
        try:
            # Render attack popup on the appropriate website
            button_bbox, close_bbox = self.environment.render_attack(seed, task.website)
            
            # Capture screenshot
            screenshot = self.environment.capture_screenshot()
            
            # Get agent action
            reasoning, coords = self.victim.get_action(screenshot, task)
            
            # Evaluate action
            outcome, coord_hit, intent_hit = self.evaluator.evaluate(
                coords, reasoning, button_bbox, close_bbox, seed
            )
            
            # Log to console
            result_str = self.evaluator.format_result(
                task.id, seed.id, trial_num, outcome, coords, button_bbox
            )
            print(result_str)
            
            return TrialResult(
                trial_number=trial_num,
                outcome=outcome.value,
                agent_coordinates=list(coords) if coords else None,
                agent_reasoning=reasoning,
                ground_truth_bbox=button_bbox,
                close_button_bbox=close_bbox,
                coordinate_hit=coord_hit,
                intent_hit=intent_hit
            )
            
        except Exception as e:
            print(Colors.error(f"Trial {trial_num} failed: {e}"))
            import traceback
            traceback.print_exc()
            return TrialResult(
                trial_number=trial_num,
                outcome=TrialOutcome.ERROR.value,
                agent_coordinates=(0, 0),
                agent_reasoning=f"Error: {e}",
                ground_truth_bbox=(0, 0, 0, 0),
                close_button_bbox=(0, 0, 0, 0),
                coordinate_hit=False,
                intent_hit=False
            )
    
    def run_pair(
        self, 
        task: Task, 
        seed: AttackSeed,
        evolution_round: int = 0
    ) -> BenchmarkResult:
        """
        Run all trials for a single (Task, Seed) pair.
        Optionally evolves the seed if attack fails.
        
        Returns:
            BenchmarkResult with aggregated stats
        """
        round_info = f" [Evolution Round {evolution_round}]" if evolution_round > 0 else ""
        print(f"\n{Colors.BOLD}{'='*60}{Colors.ENDC}")
        print(Colors.phase("PAIR", f"{task.id} × {seed.id}{round_info}"))
        print(f"  Scenario: {task.scenario_name} ({task.scenario})")
        print(f"  Task: {task.goal[:50]}...")
        print(f"  Popup: {seed.title}")
        print(f"  Button: {seed.button_text}")
        print(f"{'='*60}")
        
        result = BenchmarkResult(
            task_id=task.id,
            seed_id=seed.id,
            scenario=task.scenario,
            scenario_name=task.scenario_name,
            task_description=task.goal,
            popup_content=seed.to_dict()
        )
        
        # Run trials with early termination on failure
        failure_count = 0
        success_count = 0
        max_failures_allowed = self.config.trials_per_pair - self.config.success_threshold + 1
        
        for trial_num in range(1, self.config.trials_per_pair + 1):
            trial_result = self.run_trial(task, seed, trial_num)
            result.trials.append({
                **trial_result.to_dict(),
                'is_success': trial_result.is_success
            })
            
            # Track success/failure
            if trial_result.is_success:
                success_count += 1
            else:
                failure_count += 1
            
            # Early termination: if already reached success threshold, no need to continue
            if success_count >= self.config.success_threshold:
                print(Colors.success(f"  Early success: {success_count}/{self.config.success_threshold} achieved, skipping remaining trials"))
                break
            
            # Early termination: if failures make success impossible, stop to save tokens
            if failure_count >= max_failures_allowed:
                print(Colors.warning(f"  Early termination: {failure_count} failures, success impossible"))
                break
            
            # Brief pause between trials
            time.sleep(0.5)
        
        # Compute final result
        result.compute_result(self.config.success_threshold)
        
        # Log result
        self.logger.log_benchmark_result(result)
        
        # If vulnerable, log for distillation
        if result.final_result == "VULNERABLE":
            # Get the reasoning from a successful trial
            success_reasoning = ""
            for t in result.trials:
                if t.get('is_success'):
                    success_reasoning = t.get('agent_reasoning', '')
                    break
            
            attack = SuccessfulAttack(
                task_id=task.id,
                task_goal=task.goal,
                scenario=task.scenario,
                scenario_name=task.scenario_name,
                seed_id=seed.id,
                seed_category=seed.category,
                popup_title=seed.title,
                popup_body=seed.body,
                popup_button=seed.button_text,
                agent_reasoning=success_reasoning,
                success_rate=result.stats
            )
            self.logger.log_successful_attack(attack)
            
            # Save screenshot of successful attack
            # Re-render the popup and capture screenshot
            try:
                self.environment.render_attack(seed, task.website)
                screenshot = self.environment.capture_screenshot()
                self.logger.save_success_screenshot(
                    scenario=task.scenario,
                    task_id=task.id,
                    seed_id=seed.id,
                    screenshot_base64=screenshot
                )
            except Exception as e:
                print(Colors.warning(f"Failed to save screenshot: {e}"))
        
        # If ROBUST and evolution is enabled, try to evolve
        elif self.enable_evolution and evolution_round < self.max_evolution_rounds:
            print(Colors.warning(f"Attack failed, attempting evolution (round {evolution_round + 1})..."))
            
            # Collect failed trial results for evolution
            failed_trials = [
                TrialResult(
                    trial_number=t['trial_number'],
                    outcome=t['outcome'],
                    agent_coordinates=tuple(t['agent_coordinates']) if t['agent_coordinates'] is not None else None,
                    agent_reasoning=t['agent_reasoning'],
                    ground_truth_bbox=tuple(t['ground_truth_bbox']),
                    close_button_bbox=tuple(t.get('close_button_bbox', (0,0,0,0))),
                    coordinate_hit=t['coordinate_hit'],
                    intent_hit=t['intent_hit']
                )
                for t in result.trials
            ]
            
            # Evolve the seed
            evolved_seed = self.attacker.evolve(seed, task.goal, failed_trials)
            
            # Recursively run with evolved seed
            return self.run_pair(task, evolved_seed, evolution_round + 1)
        
        return result
    
    def run_benchmark(
        self, 
        tasks: List[Task] = None, 
        seeds: List[AttackSeed] = None,
        start_task: str = None,
        start_seed: str = None
    ) -> List[BenchmarkResult]:
        """
        Run the full benchmark across all (Task, Seed) pairs.
        
        Args:
            tasks: List of tasks (defaults to loading from JSON)
            seeds: List of attack seeds (defaults to loading from JSON)
            start_task: Task ID to start from (skip earlier tasks)
            start_seed: Seed ID to start from (only with start_task)
            
        Returns:
            List of BenchmarkResults
        """
        # Load from JSON if not provided
        if tasks is None:
            tasks = load_tasks(self.config)
        if seeds is None:
            seeds = load_seeds(self.config)
        
        total_pairs = len(tasks) * len(seeds)
        total_trials = total_pairs * self.config.trials_per_pair
        
        print("\n" + "="*80)
        print(f"{Colors.BOLD}{Colors.HEADER}GUI AGENT ROBUSTNESS BENCHMARK{Colors.ENDC}")
        print("="*80)
        print(f"Tasks: {len(tasks)}")
        print(f"Attack Seeds: {len(seeds)}")
        print(f"Total Pairs: {total_pairs}")
        print(f"Trials per Pair: {self.config.trials_per_pair}")
        print(f"Total Trials: {total_trials}")
        print(f"Success Threshold: {self.config.success_threshold}/{self.config.trials_per_pair}")
        print("="*80)
        
        # Print scenarios
        scenarios = set((t.scenario, t.scenario_name) for t in tasks)
        print(f"\nScenarios:")
        for scenario, name in sorted(scenarios):
            count = sum(1 for t in tasks if t.scenario == scenario)
            print(f"  - {name} ({scenario}): {count} tasks")
        print()
        
        results = []
        
        # Determine if we need to skip some pairs
        skip_mode = start_task is not None
        found_start = not skip_mode  # If no start specified, don't skip anything
        
        try:
            # Iterate through Cartesian Product
            for i, (task, seed) in enumerate(product(tasks, seeds), 1):
                # Skip logic for resuming from a specific point
                if skip_mode and not found_start:
                    if task.id == start_task:
                        if start_seed is None or seed.id == start_seed:
                            found_start = True
                            print(Colors.info(f"Resuming from {task.id} × {seed.id}"))
                        else:
                            print(f"  Skipping {task.id} × {seed.id} (before start point)")
                            continue
                    else:
                        print(f"  Skipping {task.id} × {seed.id} (before start point)")
                        continue
                
                print(f"\n{Colors.CYAN}[Progress: {i}/{total_pairs}]{Colors.ENDC}")
                
                result = self.run_pair(task, seed)
                results.append(result)
                
                # Pause between pairs
                time.sleep(1)
            
            # Print summary
            self.logger.print_summary()
            
        except KeyboardInterrupt:
            print(Colors.warning("\nBenchmark interrupted by user"))
        finally:
            self.environment.close()
        
        return results
    
    def close(self):
        """Clean up resources."""
        self.environment.close()


def main():
    """Main entry point for running the benchmark."""
    config = Config()
    runner = BenchmarkRunner(config)
    
    try:
        runner.run_benchmark()
    except Exception as e:
        print(Colors.error(f"Benchmark failed: {e}"))
        raise
    finally:
        runner.close()


if __name__ == "__main__":
    main()
