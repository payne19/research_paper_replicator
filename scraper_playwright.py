import playwright
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup


class WebScraper():

    def __init__(self, url):
        self.url = url

    def extract_text_from_webpage(self):
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(self.url)
            data = page.content()
            browser.close()
        return data

if __name__ == "__main__":
    url = 'https://arxiv.org/html/2504.19874v1'
    scraper = WebScraper(url)
    webpage_content = scraper.extract_text_from_webpage()
    webpage_content = BeautifulSoup(webpage_content, 'html.parser').get_text()
    with open('webpage_content.txt', 'w', encoding='utf-8') as f:
        f.write(webpage_content)