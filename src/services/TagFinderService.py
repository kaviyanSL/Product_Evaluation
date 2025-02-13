import requests
from bs4 import BeautifulSoup
import time
import re

from playwright.sync_api import sync_playwright

class TagFinderService:
    def __init__(self, URL, playwright=False):
        self.URL = URL
        self.playwright = playwright

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
        if not self.playwright:
            try:
                response = requests.get(self.URL, headers=headers, timeout=10)
                if 200 <= response.status_code < 300:
                    return response.text
            except requests.RequestException as e:
                print(f"Error fetching {self.URL} using Requests: {e}")
            return None

        # === Option 2: Use Playwright (For JavaScript-heavy sites) ===
        else:
            print("Using Playwright for dynamic content scraping...")
            try:
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    page = browser.new_page()
                    page.set_extra_http_headers(headers)
                    page.goto(self.URL, timeout=60000)  # Set a timeout of 60 seconds
                    time.sleep(5)  # Wait for initial JavaScript content to load

                    # Scroll down to load more comments
                    last_height = page.evaluate("document.body.scrollHeight")
                    while True:
                        page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
                        time.sleep(2)  # Wait for new comments to load
                        new_height = page.evaluate("document.body.scrollHeight")
                        if new_height == last_height:
                            break
                        last_height = new_height

                    # Click "See more reviews" buttons if present, up to 10 times
                    click_count = 0
                    while click_count < 10:
                        try:
                            load_more_buttons = page.query_selector_all("span:has-text('See more reviews')")
                            if not load_more_buttons:
                                break
                            for button in load_more_buttons:
                                if re.search(r'See more reviews', button.inner_text(), re.IGNORECASE):
                                    button.click()
                                    time.sleep(2)  # Wait for new comments to load
                                    click_count += 1
                                    if click_count >= 10:
                                        break
                        except Exception:
                            break

                    page_source = page.content()
                    browser.close()
                    return page_source
            except Exception as e:
                print(f"Error fetching {self.URL} using Playwright: {e}")
            return None

    def find_reviews(self):
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
        if len(comments) == 0:
            comments = self.extract_general_comments(soup)
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