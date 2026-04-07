"""
Distillation Configuration
"""

from pathlib import Path
from dataclasses import dataclass


@dataclass
class DistillationConfig:
    """Configuration for rule distillation."""
    
    # GPT-5 API settings (via WokaAI proxy) - fixed, not configurable
    api_base: str = "Your API base URL (e.g. for Azure or proxy)"
    api_key: str = "Your API key"
    model: str = "gpt-5-chat"
    
    # Source data info (for output naming only)
    attacker_model: str = "glm-4.5-flash"
    victim_model: str = "glm-4v-flash"
    
    # Paths
    output_dir: Path = Path(__file__).parent.parent / "output"
    successful_attacks_file: str = "successful_attacks.jsonl"
    
    # LLM settings
    temperature: float = 0.3
    max_tokens: int = 8192
    max_retries: int = 10
    
    @property
    def successful_attacks_path(self) -> Path:
        return self.output_dir / self.successful_attacks_file
    
    def get_output_filename(self) -> str:
        """Get output filename with model names."""
        attacker_name = self.attacker_model.replace("-", "_")
        victim_name = self.victim_model.replace("-", "_")
        return f"rules_{attacker_name}_{victim_name}.json"
    
    def __post_init__(self):
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    SUCCESS = GREEN
    WARNING = YELLOW
    ERROR = RED
    
    @classmethod
    def success(cls, msg: str) -> str:
        return f"{cls.GREEN}✓ {msg}{cls.ENDC}"
    
    @classmethod
    def error(cls, msg: str) -> str:
        return f"{cls.RED}✗ {msg}{cls.ENDC}"
    
    @classmethod
    def warning(cls, msg: str) -> str:
        return f"{cls.YELLOW}⚠ {msg}{cls.ENDC}"
    
    @classmethod
    def info(cls, msg: str) -> str:
        return f"{cls.CYAN}ℹ {msg}{cls.ENDC}"
