import requests
from bs4 import BeautifulSoup
import re

class TagFinderService ():
    def __init__(self,URL):
        self.URL = URL

    def readig_html_component(self):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        """Fetch HTML content from a given URL."""
        try:
            response = requests.get(self.URL, headers=headers, timeout=10)
            if response.status_code == 200:
                return response.text
        except requests.RequestException as e:
            print(f"Error fetching {self.URL}: {e}")
        return None
    
    
    def find_tag(self):
        html = self.readig_html_component()
        comment_pattern = re.compile(r'comment', re.IGNORECASE)
        try:
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                comment_section = soup.find_all(
                            lambda tag: tag.name in ['div', 'section', 'ul', 'ol', 'article'] and 
                            (comment_pattern.search(' '.join(tag.get('class', []))) or comment_pattern.search(tag.get('id', ''))))
                comments = []
                for section in comment_section:
                    comments.extend([c.get_text(strip=True) for c in section.find_all(['p', 'li'])])
                
                # Additional logic to handle Amazon-specific comment tags
                try:
                    for review in soup.find_all('div', {'data-hook': 'review'}):  
                        title = review.find('a', {'data-hook': 'review-title'}).text.strip()  
                        rating = review.find('i', {'data-hook': 'review-star-rating'}).text.strip()  
                        review_text = review.find('span', {'data-hook': 'review-body'}).text.strip()  
                        comments.append({'title': title, 'rating': rating, 'review': review_text})  

                    return comments if comments else None
                
                except Exception as e:
                    print(f"Error in finding Amazon-specific tags {self.URL}: {e}")
                    return comments if comments else None
                
        except Exception as e:
            print(f"Error in finding tags {self.URL}: {e}")
        return None