"""
Collection Package for GUI Agent Robustness Benchmark
Purpose: Collect successful attack data for future rule distillation
"""

from .config import Config, Colors
from .data_types import Task, AttackSeed, TrialResult, BenchmarkResult
from .logger import BenchmarkLogger
from .environment import SeleniumEnvironment
from .victim import VictimAgent
from .evaluator import ActionEvaluator
from .attacker import AttackerLLM, Diagnosis, EvolutionDirection
from .benchmark_runner import BenchmarkRunner

__all__ = [
    'Config',
    'Colors', 
    'Task',
    'AttackSeed',
    'TrialResult',
    'BenchmarkResult',
    'BenchmarkLogger',
    'SeleniumEnvironment',
    'VictimAgent',
    'ActionEvaluator',
    'AttackerLLM',
    'Diagnosis',
    'EvolutionDirection',
    'BenchmarkRunner'
]
