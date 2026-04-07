"""
Evaluation Configuration
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EvaluationConfig:
    """Configuration for rule quality evaluation."""
    
    # Generator model (LLM for generating attacks from rules)
    generator_model: str = "glm-4.5-flash"
    generator_provider: Optional[str] = None  # Auto-detected from model name
    generator_api_key: Optional[str] = None   # Auto-configured from provider
    generator_base_url: Optional[str] = None  # Auto-configured from provider
    
    # Victim model (VLM agent being tested)
    victim_model: str = "glm-4v-flash"
    victim_provider: Optional[str] = None     # Auto-detected from model name
    victim_api_key: Optional[str] = None      # Auto-configured from provider
    victim_base_url: Optional[str] = None     # Auto-configured from provider
    
    # Judge model (qwen3-32b for intent judgment)
    judge_model: str = "glm-4.5-flash"
    
    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    websites_dir: str = "websites"  # Directory containing website HTML files
    rules_file: str = "output/rules.json"
    tasks_file: str = "evaluation/tasks.json"  # Evaluation-specific tasks (12 tasks)
    seeds_file: str = "collection/seeds.json"   # Seed-based baseline attacks
    output_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "evaluation_results")
    screenshots_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "evaluation_results" / "screenshots")
    
    # ChromeDriver
    chromedriver_path: str = "path/to/chromedriver.exe"
    
    # Viewport settings
    viewport_width: int = 1920
    viewport_height: int = 1080
    
    # Evaluation settings
    samples_per_rule: int = 3          # Number of attack samples to generate per rule
    trials_per_sample: int = 3         # Number of trials per attack sample
    scenarios: List[str] = field(default_factory=lambda: ["amazon", "youtube", "gmail", "discord"])
    use_universal_rules: bool = True   # Whether to include universal rules in attack generation
    control_mode: str = "rules"       # modes: rules | no_rules | seeds
    
    # LLM settings
    temperature: float = 0.7           # Higher for more diverse attack generation
    max_tokens: int = 8192             # Increased to handle larger rule sets
    max_retries: int = 10
    
    @property
    def rules_path(self) -> Path:
        # Support both absolute and relative paths
        path = Path(self.rules_file)
        if path.is_absolute():
            return path
        return self.project_root / self.rules_file
    
    @property
    def tasks_path(self) -> Path:
        # Support both absolute and relative paths
        path = Path(self.tasks_file)
        if path.is_absolute():
            return path
        return self.project_root / self.tasks_file

    @property
    def seeds_path(self) -> Path:
        path = Path(self.seeds_file)
        if path.is_absolute():
            return path
        return self.project_root / self.seeds_file
    
    def get_output_report_path(self) -> Path:
        """Get output path with model names."""
        generator_name = self.generator_model.replace("-", "_")
        victim_name = self.victim_model.replace("-", "_")
        return self.output_dir / f"evaluation_report_{generator_name}_{victim_name}.json"
    
    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)


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
