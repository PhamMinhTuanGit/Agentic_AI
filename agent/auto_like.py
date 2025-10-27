#!/usr/bin/env python3
"""
Auto-like script for Padlet using Selenium WebDriver
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
import time
import logging
import sys

# Configuration
CONFIG = {
    'url': "https://padlet.com/vonguyetanh1/happy-b-t-y-s-ng-nh-s-ng-kh-e-myxx1sbk0s3ezqi2",
    'button_selector': 'button[data-testid="surfacePostReactionEmojiAccumulatedReactionsButton-2764"]',
    'alternative_selector': 'button[data-testid^="surfacePostReactions"]',
    'interval': 10,  # seconds
    'page_load_timeout': 30,
    'element_wait_timeout': 20,
    'headless': True,
    'retry_attempts': 3,
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def create_chrome_driver():
    """Create and configure Chrome WebDriver with robust options"""
    chrome_options = Options()
    
    if CONFIG['headless']:
        chrome_options.add_argument("--headless=new")  # Use new headless mode
    
    # Essential options to fix DevToolsActivePort error
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-software-rasterizer")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-setuid-sandbox")
    
    # Additional stability options
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    # Suppress logging
    chrome_options.add_argument("--log-level=3")
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(CONFIG['page_load_timeout'])
        return driver
    except WebDriverException as e:
        logger.error(f"Failed to create Chrome driver: {e}")
        logger.error("Make sure Chrome and ChromeDriver are installed and compatible")
        raise


def click_like_button(driver):
    """Find and click the like button on Padlet"""
    try:
        # Try primary selector first
        logger.info("Waiting for like button to appear...")
        wait = WebDriverWait(driver, CONFIG['element_wait_timeout'])
        
        try:
            button = wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, CONFIG['button_selector']))
            )
            logger.info("Found button with primary selector")
        except TimeoutException:
            logger.warning("Primary selector not found, trying alternative...")
            button = wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, CONFIG['alternative_selector']))
            )
            logger.info("Found button with alternative selector")
        
        # Scroll into view and click
        driver.execute_script("arguments[0].scrollIntoView(true);", button)
        time.sleep(0.5)
        button.click()
        
        logger.info(f"💖 Like button clicked successfully at {time.strftime('%H:%M:%S')}")
        return True
        
    except TimeoutException:
        logger.error("❌ Timeout: Like button not found")
        return False
    except NoSuchElementException:
        logger.error("❌ Like button element not found")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error clicking button: {e}")
        return False


def auto_like_iteration():
    """Perform one auto-like iteration"""
    driver = None
    success = False
    
    for attempt in range(1, CONFIG['retry_attempts'] + 1):
        try:
            logger.info(f"Starting attempt {attempt}/{CONFIG['retry_attempts']}")
            
            driver = create_chrome_driver()
            logger.info(f"🌐 Opening Padlet page...")
            
            driver.get(CONFIG['url'])
            logger.info("Page loaded successfully")
            
            # Wait a bit for dynamic content
            time.sleep(2)
            
            success = click_like_button(driver)
            
            if success:
                break
            else:
                logger.warning(f"Attempt {attempt} failed, retrying...")
                
        except WebDriverException as e:
            logger.error(f"WebDriver error on attempt {attempt}: {e}")
            if attempt < CONFIG['retry_attempts']:
                logger.info("Retrying in 3 seconds...")
                time.sleep(3)
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt}: {e}")
            if attempt < CONFIG['retry_attempts']:
                logger.info("Retrying in 3 seconds...")
                time.sleep(3)
        finally:
            if driver:
                try:
                    driver.quit()
                    logger.info("🔒 Browser closed")
                except Exception as e:
                    logger.warning(f"Error closing browser: {e}")
    
    return success


def main():
    """Main execution loop"""
    logger.info("🚀 Auto-like Padlet script started")
    logger.info(f"Interval: {CONFIG['interval']} seconds")
    logger.info(f"Headless mode: {CONFIG['headless']}")
    logger.info(f"URL: {CONFIG['url']}\n")
    
    iteration = 0
    
    try:
        while True:
            iteration += 1
            logger.info(f"\n{'='*50}")
            logger.info(f"Iteration {iteration}")
            logger.info(f"{'='*50}")
            
            success = auto_like_iteration()
            
            if success:
                logger.info("✅ Task completed successfully")
            else:
                logger.warning("⚠️  Task completed with errors")
            
            logger.info(f"\n⏳ Waiting {CONFIG['interval']} seconds before next run...")
            time.sleep(CONFIG['interval'])
            
    except KeyboardInterrupt:
        logger.info("\n\n⏹️  Script stopped by user (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n\n💥 Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
