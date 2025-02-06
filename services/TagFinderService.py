import requests
from bs4 import BeautifulSoup
import time
import re

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

class TagFinderService:
    def __init__(self, URL, use_selenium=False):
        self.URL = URL
        self.use_selenium = use_selenium

    def read_html_component(self):

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
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
            
            service = Service("chromedriver.exe")  # Ensure chromedriver is in your path
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
        html = self.read_html_component()
        if not html:
            print("No HTML content found.")
            return None

        soup = BeautifulSoup(html, 'html.parser')
        reviews = []

        # === Extract Amazon Reviews ===
        review_divs = soup.find_all('div', {'data-hook': 'review'})
        for review in review_divs:
            try:
                title = review.find('a', {'data-hook': 'review-title'}).get_text(strip=True)
                rating = review.find('i', {'data-hook': 'review-star-rating'}).get_text(strip=True)
                review_text = review.find('span', {'data-hook': 'review-body'}).get_text(strip=True)
                reviews.append({
                    'title': title,
                    'rating': rating,
                    'review': review_text
                })
            except AttributeError:
                continue

        return reviews if reviews else None

