"""
Victim Agent Module
Simulates a naive user interacting with the GUI.
Uses centralized VLM from models module.
"""

import re
import json
from typing import Tuple, Optional

from .config import Config, Colors
from .data_types import Task

# Import from centralized models
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.vlm import VisionLanguageModel, VLMConfig, VLMProvider, get_provider_from_model, get_api_config


class VictimAgent:
    """
    Simulates a naive user (victim) using VLM.
    
    IMPORTANT: The prompt is PURELY TASK-ORIENTED.
    We do NOT mention security, popups, or warnings.
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # Determine provider and API config from model name
        # Use victim model settings from config if available
        model = getattr(config, 'victim_model', config.vision_model)
        provider_str = getattr(config, 'victim_provider', None)
        
        if provider_str:
            provider = VLMProvider(provider_str)
            api_key = getattr(config, 'victim_api_key', config.api_key)
            base_url = getattr(config, 'victim_base_url', None)
        else:
            provider = get_provider_from_model(model)
            api_config = get_api_config(provider)
            api_key = api_config["api_key"]
            base_url = api_config.get("base_url")
        
        # Initialize VLM using centralized model
        vlm_config = VLMConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.7,
            max_tokens=512,
            max_retries=10
        )
        self.vlm = VisionLanguageModel(vlm_config)
        
        print(Colors.info(f"VictimAgent initialized ({model}, {provider.value})"))
    
    def get_action(self, screenshot_base64: str, task: Task) -> Tuple[str, Optional[Tuple[int, int]]]:
        """
        Get agent's action for the current screen state.
        
        Args:
            screenshot_base64: Current screen as base64 image
            task: The user's task goal
            
        Returns:
            (reasoning_text, (x, y) coordinates or None if parsing failed)
        """
        reasoning, coords = self.vlm.get_click_action(screenshot_base64, task.goal)
        return reasoning, coords
