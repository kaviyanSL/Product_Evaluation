import requests
from bs4 import BeautifulSoup
import time
import re

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

class TagFinderService:
    def __init__(self, URL, use_selenium=False, driver_path="chromedriver.exe"):
        """
        Initialize the scraper.
        :param URL: The webpage URL to scrape.
        :param use_selenium: If True, uses Selenium; otherwise, uses Requests + BeautifulSoup.
        :param driver_path: Path to the Chromedriver binary.
        """
        self.URL = URL
        self.use_selenium = use_selenium
        self.driver_path = driver_path

    def read_html_component(self):
        """
        Fetches HTML content using either Requests (for static sites) or Selenium (for JavaScript-heavy sites).
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }

        # === Option 1: Use Requests (For Static HTML) ===
        if not self.use_selenium:
            try:
                response = requests.get(self.URL, headers=headers, timeout=10)
                if 200 <= response.status_code < 300:
                    return response.text
            except requests.RequestException as e:
                print(f"Error fetching {self.URL} using Requests: {e}")
            return None

        # === Option 2: Use Selenium (For JavaScript-heavy sites) ===
        else:
            print("Using Selenium for dynamic content scraping...")
            options = Options()
            options.add_argument("--headless")  # Run without opening a browser
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1920x1080")
            options.add_argument("--log-level=3")  # Reduce Selenium logs
            
            service = Service(self.driver_path)  # Ensure chromedriver is correctly set
            driver = webdriver.Chrome(service=service, options=options)
            
            try:
                driver.get(self.URL)
                time.sleep(5)  # Wait for JavaScript content to load
                page_source = driver.page_source
                driver.quit()
                return page_source
            except Exception as e:
                print(f"Error fetching {self.URL} using Selenium: {e}")
                driver.quit()
            return None

    def find_reviews(self):
        """
        Extracts reviews and comments from the webpage.
        Works with both static and dynamic Amazon pages.
        """
        html = self.read_html_component()
        if not html:
            print("No HTML content found.")
            return None

        soup = BeautifulSoup(html, 'html.parser')
        comments = []

        # === Extract Amazon Reviews ===
        review_divs = soup.find_all('div', {'data-hook': 'review-collapsed'})
        for review in review_divs:
            try:
                comments.append(review.get_text(strip=True))
            except Exception as e:
                print(f"Error in finding Amazon-specific tags {self.URL}: {e}")

        # === Generic Comment Extraction ===
        comment_pattern = re.compile(r'comment', re.IGNORECASE)
        try:
            comment_section = soup.find_all(
                lambda tag: tag.name in ['div', 'section', 'ul', 'ol', 'article'] and 
                (comment_pattern.search(' '.join(tag.get('class', []))) or 
                 comment_pattern.search(tag.get('id', '')))
            )

            for section in comment_section:
                comments.extend([c.get_text(strip=True) for c in section.find_all(['p', 'li'])])

        except Exception as e:
            print(f"Error extracting general comments: {e}")

        return comments if comments else None
