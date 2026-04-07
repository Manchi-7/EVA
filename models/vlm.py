"""
Vision Language Model (VLM) / GUI Agent API
Centralized vision-language model calls for GUI automation agents.
Supports: GLM-4V-Flash, GPT-4V, Qwen-VL, etc.
"""

import time
import json
import re
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

import requests
from zhipuai import ZhipuAI
from openai import OpenAI


class VLMProvider(Enum):
    """Supported VLM providers."""
    ZHIPU = "zhipu"
    OPENAI = "openai"
    QWEN = "qwen"
    VLLM = "vllm"  # 本地vLLM服务


# API configurations for each provider
PROVIDER_CONFIGS = {
    VLMProvider.ZHIPU: {
        "api_key": "Your API key",
        "base_url": None,
    },
    VLMProvider.OPENAI: {
        "api_key": "Your API key",
        "base_url": "Your API base URL (optional, e.g. for Azure or proxy)",
    },
    VLMProvider.QWEN: {
        "api_key": "Your API key",
        "base_url": "Your API base URL (optional, e.g. for proxy)",
    },
    VLMProvider.VLLM: {
        "api_key": "EMPTY",  # vLLM不需要API key
        "base_url": "http://localhost:8000/v1",  # 默认端口
    },
}

# vLLM服务器配置
VLLM_SERVER_HOST = "localhost"  # SSH端口转发到本地

# vLLM本地模型端口映射
VLLM_MODEL_PORTS = {
    "qwen2.5-vl-7b-instruct": 8000,
    "qwen2-vl-7b-instruct": 8001,
    "qwen3-vl-8b-instruct": 8002,
    "ui-tars-1.5-7b": 8003,
    "os-atlas-pro-7b": 8004,
    "os-genesis-8b-ac": 8005,
    "gui-owl-7b": 8006,
}

# vLLM模型名称映射(本地名称 -> 服务器路径)
VLLM_MODEL_NAME_MAPPING = {
    "os-atlas-pro-7b": "/path/to/OS-Atlas-Pro-7B",
    "os-genesis-8b-ac": "/path/to/OS-Genesis-8B-AC",
    "qwen2.5-vl-7b-instruct": "/path/to/Qwen2.5-VL-7B-Instruct",
    "qwen2-vl-7b-instruct": "/path/to/Qwen2-VL-7B-Instruct",
    "qwen3-vl-8b-instruct": "/path/to/Qwen3-VL-8B-Instruct",
    "ui-tars-1.5-7b": "/path/to/UI-TARS-1.5-7B",
    "gui-owl-7b": "/path/to/GUI-Owl-7B",
}


def get_provider_from_model(model_name: str) -> VLMProvider:
    """Auto-detect provider from model name."""
    model_lower = model_name.lower()
    if model_lower.startswith("glm-"):
        return VLMProvider.ZHIPU
    elif model_lower.startswith("gpt-") or model_lower.startswith("o1"):
        return VLMProvider.OPENAI
    elif model_lower.startswith("qwen") and "vl" not in model_lower:
        return VLMProvider.QWEN
    elif any(name in model_lower for name in ["qwen2.5-vl", "qwen2-vl", "qwen3-vl", "ui-tars", "os-atlas", "os-genesis", "gui-owl"]):
        return VLMProvider.VLLM  # 本地vLLM模型
    else:
        # Default to Zhipu
        return VLMProvider.ZHIPU


def get_api_config(provider: VLMProvider) -> Dict[str, Any]:
    """Get API configuration for a provider."""
    return PROVIDER_CONFIGS.get(provider, PROVIDER_CONFIGS[VLMProvider.ZHIPU])


def get_vllm_base_url(model_name: str) -> str:
    """Get vLLM base URL for local model based on model name."""
    model_lower = model_name.lower()
    for model_key, port in VLLM_MODEL_PORTS.items():
        if model_key in model_lower:
            return f"http://{VLLM_SERVER_HOST}:{port}/v1"
    return f"http://{VLLM_SERVER_HOST}:8000/v1"  # 默认端口


def _detect_image_mime(image_base64: str) -> str:
    """Detect MIME type from base64-encoded image magic bytes."""
    import base64 as _b64
    header = _b64.b64decode(image_base64[:16])
    if header[:3] == b'\xff\xd8\xff':
        return "image/jpeg"
    if header[:8] == b'\x89PNG\r\n\x1a\n':
        return "image/png"
    if header[:6] in (b'GIF87a', b'GIF89a'):
        return "image/gif"
    if header[:4] == b'RIFF' and header[8:12] == b'WEBP':
        return "image/webp"
    return "image/jpeg"   # safe fallback


@dataclass
class VLMConfig:
    """Configuration for vision language model calls."""
    provider: VLMProvider = VLMProvider.ZHIPU
    model: str = "glm-4v-flash"
    api_key: str = ""
    api_url: str = ""
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 512
    max_retries: int = 10


class VisionLanguageModel:
    """
    Unified Vision Language Model interface for GUI Agents.
    
    Usage:
        vlm = VisionLanguageModel(config)
        reasoning, coords = vlm.get_click_action(screenshot_b64, goal)
    """
    
    # Default GUI Agent prompt
    GUI_AGENT_PROMPT = """You are a GUI automation agent. Your current goal is: {goal}

Look at the screen carefully and decide your next action.
What would you click to make progress on your goal?

Respond in JSON format:
{{
    "observation": "What do you see on the screen",
    "reasoning": "Why you choose to click this element", 
    "click_x": <number 0-1000>,
    "click_y": <number 0-1000>
}}

Coordinates are normalized: (0,0) is top-left, (1000,1000) is bottom-right."""

    def __init__(self, config: VLMConfig):
        self.config = config
        
        # For vLLM, override base_url based on model name
        if config.provider == VLMProvider.VLLM:
            config.base_url = get_vllm_base_url(config.model)
            config.api_key = "EMPTY"  # vLLM不需要API key
        
        if config.provider == VLMProvider.ZHIPU:
            # For Zhipu, we use requests directly for more control
            self.headers = {
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json"
            }
            self.client = ZhipuAI(api_key=config.api_key)
        elif config.provider in [VLMProvider.OPENAI, VLMProvider.QWEN, VLMProvider.VLLM]:
            self.client = OpenAI(
                api_key=config.api_key,
                base_url=config.base_url
            )
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")
    
    def chat_with_image(
        self,
        image_base64: str,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Optional[str]:
        """
        Send image and prompt, get text response.
        
        Args:
            image_base64: Base64 encoded image
            prompt: Text prompt
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            
        Returns:
            Response content string or None if failed
        """
        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        
        for attempt in range(self.config.max_retries):
            try:
                if self.config.provider == VLMProvider.ZHIPU:
                    return self._zhipu_request(image_base64, prompt, temp, tokens)
                else:
                    return self._openai_request(image_base64, prompt, temp, tokens)
                    
            except Exception as e:
                if self._is_rate_limit_error(e):
                    wait_time = min(2 ** attempt, 60)
                    print(f"[VLM] Rate limited, retry {attempt + 1}/{self.config.max_retries} after {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"[VLM] Error: {e}")
                    if attempt == self.config.max_retries - 1:
                        return None
                    continue
        
        return None
    
    def _zhipu_request(
        self,
        image_base64: str,
        prompt: str,
        temperature: float,
        max_tokens: int
    ) -> Optional[str]:
        """Make request to Zhipu API using requests library."""
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{_detect_image_mime(image_base64)};base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(
            self.config.api_url,
            json=payload,
            headers=self.headers,
            timeout=60
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    
    def _zhipu_sdk_request(
        self,
        image_base64: str,
        prompt: str,
        temperature: float,
        max_tokens: int
    ) -> Optional[str]:
        """Make request to Zhipu API using SDK."""
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                        },
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    
    def _openai_request(
        self,
        image_base64: str,
        prompt: str,
        temperature: float,
        max_tokens: int
    ) -> Optional[str]:
        """Make request to OpenAI-compatible API."""
        # 对于vLLM模型,使用映射后的服务器路径
        model_name = self.config.model
        if self.config.provider == VLMProvider.VLLM:
            model_lower = model_name.lower()
            for local_name, server_path in VLLM_MODEL_NAME_MAPPING.items():
                if local_name in model_lower:
                    model_name = server_path
                    break
        
        response = self.client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{_detect_image_mime(image_base64)};base64,{image_base64}"}
                        },
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    
    def get_click_action(
        self,
        screenshot_base64: str,
        goal: str,
        custom_prompt: Optional[str] = None
    ) -> Tuple[str, Optional[Tuple[int, int]]]:
        """
        Get GUI agent's click action for the current screen.
        
        Args:
            screenshot_base64: Screenshot as base64 string
            goal: The user's task goal
            custom_prompt: Optional custom prompt template (must include {goal})
            
        Returns:
            (reasoning, (x, y)) where reasoning is the agent's explanation
            and (x, y) are click coordinates in 0-1000 scale, or None if parsing failed
        """
        prompt = custom_prompt or self.GUI_AGENT_PROMPT
        prompt = prompt.format(goal=goal)
        
        response = self.chat_with_image(screenshot_base64, prompt)
        
        if response:
            return self._parse_action_response(response)
        
        return "Error: Failed to get VLM response", None
    
    def _parse_action_response(self, content: str) -> Tuple[str, Optional[Tuple[int, int]]]:
        """Parse VLM response to extract reasoning and coordinates."""
        reasoning = content
        coords = None  # 解析失败返回 None
        
        # Debug: 打印完整的原始响应
        print(f"\n{'='*60}")
        print(f"[VLM Raw Response (first 500 chars)]:")
        print(content[:500])
        print(f"{'='*60}\n")
        
        try:
            # 特殊格式1: <number>[[x,y]]</number> 或 <point>[[x,y]]</point> (OS-Atlas等模型)
            xml_coord_match = re.search(r'<(?:number|point)>\[\[(\d+),(\d+)\]\]</(?:number|point)>', content)
            if xml_coord_match:
                coords = (int(xml_coord_match.group(1)), int(xml_coord_match.group(2)))
                print(f"[Parsed] XML/Point tag format: {coords}")
                return reasoning, coords
            
            # 特殊格式2: {"action_type":"click","x":481.0,"y":318.5} (OS-Genesis等模型)
            action_json_match = re.search(r'\{"action_type"\s*:\s*"click"\s*,\s*"x"\s*:\s*(\d+(?:\.\d+)?)\s*,\s*"y"\s*:\s*(\d+(?:\.\d+)?)\s*\}', content)
            if action_json_match:
                coords = (int(float(action_json_match.group(1))), int(float(action_json_match.group(2))))
                print(f"[Parsed] Action JSON format: {coords}")
                return reasoning, coords
            
            # 特殊格式3: {"click_x": 77.0,"y":1272.0} 或 {"x":123,"click_y":456} (混合格式)
            # 尝试提取任意包含 x/click_x 和 y/click_y 的JSON片段
            mixed_coord_match = re.search(
                r'\{\s*"(?:click_)?([xy])"\s*:\s*(\d+(?:\.\d+)?)\s*,\s*"(?:click_)?([xy])"\s*:\s*(\d+(?:\.\d+)?)\s*\}',
                content
            )
            if mixed_coord_match:
                # 确定哪个是x,哪个是y
                first_key, first_val, second_key, second_val = mixed_coord_match.groups()
                if first_key == 'x':
                    x_val, y_val = first_val, second_val
                else:
                    x_val, y_val = second_val, first_val
                coords = (int(float(x_val)), int(float(y_val)))
                print(f"[Parsed] Mixed coordinate format: {coords}")
                return reasoning, coords
            
            # 特殊格式4: "click_x": <number 607,420 (不完整的JSON)
            broken_coord_match = re.search(r'"click_x":\s*<number\s+(\d+),(\d+)', content)
            if broken_coord_match:
                coords = (int(broken_coord_match.group(1)), int(broken_coord_match.group(2)))
                print(f"[Parsed] Broken JSON format: {coords}")
                return reasoning, coords
            
            # 标准JSON格式
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                print(f"[JSON Found] Length: {len(json_str)}, First 200 chars: {json_str[:200]}")
                data = json.loads(json_str)
                reasoning = data.get("reasoning", data.get("observation", content))
                
                # 处理坐标:可能是单个数字或列表
                x_val = data.get("click_x")
                y_val = data.get("click_y")
                
                print(f"[Extracted] click_x={x_val} (type={type(x_val).__name__}), click_y={y_val} (type={type(y_val).__name__})")
                
                if x_val is None or y_val is None:
                    print(f"[Parse Warning] Missing coordinates in JSON - click_x={x_val}, click_y={y_val}")
                    print(f"[JSON Keys] {list(data.keys())}")
                    return reasoning, None
                
                # 如果是列表,取第一个元素
                if isinstance(x_val, list):
                    x_val = x_val[0] if x_val else None
                if isinstance(y_val, list):
                    y_val = y_val[0] if y_val else None
                
                if x_val is not None and y_val is not None:
                    coords = (int(x_val), int(y_val))
                    print(f"[Parsed] JSON format: {coords}")
                    return reasoning, coords
            else:
                print(f"[Parse Warning] No JSON found in response")
                
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            print(f"[Parse Error] {type(e).__name__}: {e}")
        
        # 通用fallback: 提取任意两个连续数字
        coord_match = re.search(r'(\d+)\s*[,\s]\s*(\d+)', content)
        if coord_match:
            coords = (int(coord_match.group(1)), int(coord_match.group(2)))
            print(f"[Parsed] Fallback regex: {coords}")
        else:
            print(f"[Parse Failed] No coordinates found, returning None")
        
        return reasoning, coords
    
    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if error is a rate limit error."""
        error_str = str(error).lower()
        return any(keyword in error_str for keyword in [
            "429", "1305", "rate", "limit", "请求过多", "too many"
        ])


# Convenience factory functions
def create_zhipu_vlm(
    api_key: str,
    model: str = "glm-4v-flash",
    temperature: float = 0.7,
    max_tokens: int = 512
) -> VisionLanguageModel:
    """Create a Zhipu AI vision language model."""
    config = VLMConfig(
        provider=VLMProvider.ZHIPU,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return VisionLanguageModel(config)


def create_openai_vlm(
    api_key: str,
    model: str = "gpt-4-vision-preview",
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 512
) -> VisionLanguageModel:
    """Create an OpenAI vision language model."""
    config = VLMConfig(
        provider=VLMProvider.OPENAI,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return VisionLanguageModel(config)
