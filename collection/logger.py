"""
Logging Module for Benchmark Results
Handles saving experiment data for future distillation.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

from .config import Config, Colors
from .data_types import BenchmarkResult, SuccessfulAttack


class BenchmarkLogger:
    """
    Logger for benchmark results.
    Saves both detailed results and successful attacks for distillation.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.results_path = config.benchmark_results_path
        self.success_path = config.successful_attacks_path
        
        # Ensure output directory exists
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(Colors.info(f"BenchmarkLogger initialized"))
        print(Colors.info(f"  Results file: {self.results_path}"))
        print(Colors.info(f"  Success file: {self.success_path}"))
    
    def log_benchmark_result(self, result: BenchmarkResult) -> None:
        """Append a benchmark result to the results file."""
        with open(self.results_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + '\n')
        
        status = Colors.GREEN if result.final_result == "VULNERABLE" else Colors.YELLOW
        print(f"{status}[LOGGED]{Colors.ENDC} {result.task_id}-{result.seed_id}: "
              f"{result.stats} -> {result.final_result}")
    
    def log_successful_attack(self, attack: SuccessfulAttack) -> None:
        """
        Log a successful attack for future rule distillation.
        Only called when attack is VULNERABLE.
        """
        with open(self.success_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(attack.to_dict(), ensure_ascii=False) + '\n')
        
        print(Colors.success(f"Successful attack logged for distillation: {attack.seed_id}"))
    
    def save_success_screenshot(self, scenario: str, task_id: str, seed_id: str, screenshot_base64: str) -> str:
        """
        Save screenshot of a successful attack.
        
        Args:
            scenario: Scenario name (e.g., 'amazon', 'youtube')
            task_id: Task ID
            seed_id: Evolved seed ID (e.g., 'urgent_001_v2')
            screenshot_base64: Screenshot as base64 string
            
        Returns:
            Path to saved image file
        """
        import base64
        
        # Create filename: scenario_taskid_seedid.png
        filename = f"{scenario}_{task_id}_{seed_id}.png"
        filepath = self.config.images_dir / filename
        
        # Decode and save
        image_data = base64.b64decode(screenshot_base64)
        with open(filepath, 'wb') as f:
            f.write(image_data)
        
        print(Colors.success(f"Screenshot saved: {filepath}"))
        return str(filepath)
    
    def get_all_results(self) -> List[Dict[str, Any]]:
        """Read all benchmark results from file."""
        results = []
        if self.results_path.exists():
            with open(self.results_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))
        return results
    
    def get_successful_attacks(self) -> List[Dict[str, Any]]:
        """Read all successful attacks for distillation."""
        attacks = []
        if self.success_path.exists():
            with open(self.success_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        attacks.append(json.loads(line))
        return attacks
    
    def print_summary(self) -> None:
        """Print summary of benchmark results."""
        results = self.get_all_results()
        
        if not results:
            print(Colors.warning("No benchmark results found"))
            return
        
        total = len(results)
        vulnerable = sum(1 for r in results if r['final_result'] == 'VULNERABLE')
        robust = total - vulnerable
        
        print("\n" + "="*80)
        print(f"{Colors.BOLD}BENCHMARK SUMMARY{Colors.ENDC}")
        print("="*80)
        print(f"Total (Task, Seed) Pairs: {total}")
        print(f"{Colors.RED}VULNERABLE:{Colors.ENDC} {vulnerable} ({vulnerable/total*100:.1f}%)")
        print(f"{Colors.GREEN}ROBUST:{Colors.ENDC} {robust} ({robust/total*100:.1f}%)")
        
        # Group by scenario
        print(f"\n{Colors.BOLD}By Scenario:{Colors.ENDC}")
        scenarios = {}
        for r in results:
            scenario_name = r.get('scenario_name', 'Unknown')
            if scenario_name not in scenarios:
                scenarios[scenario_name] = {'vulnerable': 0, 'total': 0}
            scenarios[scenario_name]['total'] += 1
            if r['final_result'] == 'VULNERABLE':
                scenarios[scenario_name]['vulnerable'] += 1
        
        for scenario_name, stats in sorted(scenarios.items()):
            v = stats['vulnerable']
            t = stats['total']
            print(f"  {scenario_name}: {v}/{t} vulnerable ({v/t*100:.1f}%)")
        
        # Group by task
        print(f"\n{Colors.BOLD}By Task:{Colors.ENDC}")
        tasks = {}
        for r in results:
            tid = r['task_id']
            if tid not in tasks:
                tasks[tid] = {'vulnerable': 0, 'total': 0}
            tasks[tid]['total'] += 1
            if r['final_result'] == 'VULNERABLE':
                tasks[tid]['vulnerable'] += 1
        
        for tid, stats in sorted(tasks.items()):
            v = stats['vulnerable']
            t = stats['total']
            print(f"  {tid}: {v}/{t} vulnerable ({v/t*100:.1f}%)")
        
        # Group by seed
        print(f"\n{Colors.BOLD}By Seed:{Colors.ENDC}")
        seeds = {}
        for r in results:
            sid = r['seed_id']
            if sid not in seeds:
                seeds[sid] = {'vulnerable': 0, 'total': 0}
            seeds[sid]['total'] += 1
            if r['final_result'] == 'VULNERABLE':
                seeds[sid]['vulnerable'] += 1
        
        for sid, stats in sorted(seeds.items()):
            v = stats['vulnerable']
            t = stats['total']
            print(f"  {sid}: {v}/{t} vulnerable ({v/t*100:.1f}%)")
        
        print("="*80)
