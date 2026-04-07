"""
Distillation Module
Extracts attack rules from successful adversarial injections using GPT-5.
"""

from .distiller import RuleDistiller
from .config import DistillationConfig

__all__ = ["RuleDistiller", "DistillationConfig"]
