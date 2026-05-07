import requests
from bs4 import BeautifulSoup


class WebScraper:

    def __init__(self, url):
        self.url = url

    def extract_text_from_webpage(self):
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        response = requests.get(self.url, headers=headers, timeout=10)
        response.raise_for_status() 
        return response.text

    def scrape_arxiv(self):
        webpage_content = self.extract_text_from_webpage()
        soup = BeautifulSoup(webpage_content, "html.parser")

        for tag in soup(["script", "style"]):
            tag.decompose()

        text = soup.get_text(separator=" ")
        return text


if __name__ == "__main__":
    url = "https://arxiv.org/html/2504.19874v1"
    scraper = WebScraper(url)

    webpage_content = scraper.scrape_arxiv()

    webpage_content = webpage_content.strip()
    webpage_content = webpage_content.replace("\n", " ")
    webpage_content = webpage_content.lower()

    with open("webpage_content.txt", "w", encoding="utf-8") as f:
        f.write(webpage_content)