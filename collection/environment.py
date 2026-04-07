"""
Selenium Environment Module
Handles browser automation and attack injection.
"""

import time
from pathlib import Path
from typing import Tuple

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait

from .config import Config, Colors
from .data_types import AttackSeed


class SeleniumEnvironment:
    """
    Selenium-based environment for rendering attacks.
    Handles page loading, popup injection, and screenshot capture.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.driver = None
        self.device_pixel_ratio = 1.0
        self._setup_driver()
    
    def _setup_driver(self) -> None:
        """Initialize Chrome WebDriver in headless mode."""
        print(Colors.info("Setting up Selenium WebDriver (headless mode)..."))
        
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")  # 无头模式
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--disable-popup-blocking")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument(f"--window-size={self.config.viewport_width},{self.config.viewport_height}")
        
        service = Service(self.config.chromedriver_path)
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Get device pixel ratio for coordinate alignment
        self.device_pixel_ratio = self.driver.execute_script("return window.devicePixelRatio || 1;")
        print(Colors.info(f"Device Pixel Ratio: {self.device_pixel_ratio}"))
        print(Colors.success("WebDriver initialized successfully"))
    
    def load_page(self, website: str = None) -> None:
        """Load the target HTML page."""
        if website:
            html_path = Path(self.config.websites_dir) / website
        else:
            html_path = Path(self.config.target_html_path)
        
        file_uri = html_path.as_uri()
        self.driver.get(file_uri)
        
        WebDriverWait(self.driver, 10).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )
        time.sleep(0.3)
    
    def render_attack(self, seed: AttackSeed, website: str = None) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]:
        """
        Inject attack popup onto page.
        
        CRITICAL: Popup is FIXED at 480px width, 200px height.
        Only one confirm button + X close button in top-right corner.
        
        Returns:
            (button_bbox, close_bbox): Both normalized to 0-1000 scale
        """
        # Refresh page to clean state
        self.load_page(website)
        
        # Calculate center position
        popup_width = self.config.popup_width
        popup_height = self.config.popup_height
        
        # Escape HTML content
        title = self._escape_html(seed.title)
        body = self._escape_html(seed.body)
        button = self._escape_html(seed.button_text)
        
        # Popup HTML with auto-expand - Only ONE button + X close
        popup_html = f"""
        <div id="attack-overlay" style="
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            z-index: 99999;
            display: flex;
            justify-content: center;
            align-items: center;
        ">
            <div id="attack-popup" style="
                width: {popup_width}px;
                min-height: {popup_height}px;
                max-width: 90%;
                max-height: 80%;
                background: white;
                border-radius: 12px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.3);
                display: flex;
                flex-direction: column;
                padding: 24px;
                box-sizing: border-box;
                position: relative;
                overflow: auto;
            ">
                <button id="attack-close-btn" style="
                    position: absolute;
                    top: 12px;
                    right: 12px;
                    width: 28px;
                    height: 28px;
                    background: #f5f5f5;
                    border: none;
                    border-radius: 50%;
                    cursor: pointer;
                    font-size: 16px;
                    color: #666;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    transition: background 0.2s;
                    flex-shrink: 0;
                ">✕</button>
                <div style="padding-right: 30px; flex: 1; margin-bottom: 16px;">
                    <h2 id="attack-title" style="
                        margin: 0 0 12px 0;
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
                        color: #1a1a1a;
                        font-size: 18px;
                        font-weight: 600;
                    ">{title}</h2>
                    <p id="attack-body" style="
                        margin: 0;
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
                        color: #666;
                        font-size: 14px;
                        line-height: 1.6;
                        word-wrap: break-word;
                    ">{body}</p>
                </div>
                <div style="display: flex; justify-content: center; flex-shrink: 0;">
                    <button id="attack-confirm-btn" style="
                        padding: 12px 48px;
                        background: linear-gradient(135deg, #4285F4 0%, #1a73e8 100%);
                        color: white;
                        border: none;
                        border-radius: 8px;
                        cursor: pointer;
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
                        font-size: 15px;
                        font-weight: 600;
                        box-shadow: 0 2px 8px rgba(66,133,244,0.3);
                        transition: transform 0.2s, box-shadow 0.2s;
                        white-space: nowrap;
                    ">{button}</button>
                </div>
            </div>
        </div>
        """
        
        # Remove existing overlay if any
        self.driver.execute_script("""
            const existing = document.getElementById('attack-overlay');
            if (existing) existing.remove();
        """)
        
        # Inject new popup
        self.driver.execute_script(f"""
            document.body.insertAdjacentHTML('beforeend', `{popup_html}`);
        """)
        
        time.sleep(0.2)
        
        # Get button bounding boxes
        button_bbox = self._get_element_bbox("attack-confirm-btn")
        close_bbox = self._get_element_bbox("attack-close-btn")
        
        return button_bbox, close_bbox
    
    def get_close_button_bbox(self) -> Tuple[float, float, float, float]:
        """Get close button bounding box."""
        return self._get_element_bbox("attack-close-btn")
    
    def _get_element_bbox(self, element_id: str) -> Tuple[float, float, float, float]:
        """Get element bounding box normalized to 0-1000 scale."""
        script = f"""
            const elem = document.getElementById('{element_id}');
            if (!elem) return null;
            const rect = elem.getBoundingClientRect();
            return {{
                x: rect.x,
                y: rect.y,
                width: rect.width,
                height: rect.height
            }};
        """
        
        rect = self.driver.execute_script(script)
        if not rect:
            return (0, 0, 0, 0)
        
        # Normalize to 0-1000 scale
        scale = self.config.coordinate_scale
        vw = self.config.viewport_width
        vh = self.config.viewport_height
        
        x1 = (rect['x'] / vw) * scale
        y1 = (rect['y'] / vh) * scale
        x2 = ((rect['x'] + rect['width']) / vw) * scale
        y2 = ((rect['y'] + rect['height']) / vh) * scale
        
        return (x1, y1, x2, y2)
    
    def capture_screenshot(self) -> str:
        """Capture screenshot as base64."""
        return self.driver.get_screenshot_as_base64()
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
            .replace("`", "&#96;"))
    
    def close(self) -> None:
        """Close the browser."""
        if self.driver:
            self.driver.quit()
            print(Colors.info("WebDriver closed"))
