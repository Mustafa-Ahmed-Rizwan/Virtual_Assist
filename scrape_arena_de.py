import scrapy
from scrapy.crawler import CrawlerProcess
from urllib.parse import urlparse
import re
import json
import logging
from twisted.internet.error import DNSLookupError, TimeoutError, TCPTimedOutError

class ArenaDeSpider(scrapy.Spider):
    name = "arena_de"
    allowed_domains = ["arena2036.de"]
    start_urls = ["https://arena2036.de/de"]  # Start with German version
    visited_urls = set()
    base_url = "https://arena2036.de"

    custom_settings = {
        'LOG_FILE': 'arena_de.log',
        'LOG_FORMAT': '%(asctime)s %(levelname)s: %(message)s',
        'LOG_DATEFORMAT': '%Y-%m-%d %H:%M:%S',
        'LOG_LEVEL': 'INFO',
        'ROBOTSTXT_OBEY': True,
        'DOWNLOAD_DELAY': 0.5,
        'AUTOTHROTTLE_ENABLED': True,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 4,
        'RETRY_TIMES': 5,
        'RETRY_HTTP_CODES': [500, 502, 503, 504, 429, 403],
        'HTTPCACHE_ENABLED': True,
        'FEEDS': {
            'arena_data_de.jsonl': {
                'format': 'jsonlines',
                'encoding': 'utf8',
                'store_empty': False,
                'overwrite': False
            }
        }
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.failed_file = open('failed_de_links.jsonl', 'a', encoding='utf-8')
        self.logger.info("Opened failed_de_links.jsonl for appending")

    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(
                url,
                callback=self.parse,
                errback=self.errback,
                meta={'is_german': True}  # Mark initial request as German
            )

    def parse(self, response):
        self.logger.info(f"SCRAPED (200): {response.url}")

        # Skip if not HTML or not German
        content_type = response.headers.get('Content-Type', b'').decode('utf-8', 'ignore').lower()
        if 'text/html' not in content_type or not self.is_german(response.url):
            return

        title = response.css('title::text').get('').strip()
        content = self.extract_content(response)

        if title or content:
            yield {
                "url": response.url,
                "title": title,
                "content": content,
                "language": "de"
            }

        # Extract and follow links
        for link in response.css('a::attr(href)').getall():
            next_url = response.urljoin(link)
            if self.should_follow(next_url):
                yield scrapy.Request(
                    next_url,
                    callback=self.parse,
                    errback=self.errback,
                    meta={'is_german': self.is_german(next_url)}
                )

    def errback(self, failure):
        request = failure.request
        url = request.url
        if failure.check(DNSLookupError):
            reason = "DNS lookup failed"
        elif failure.check(TimeoutError, TCPTimedOutError):
            reason = "Timeout"
        else:
            reason = failure.getErrorMessage()

        self.logger.error(f"FAILED ({reason}): {url}")
        record = {"url": url, "reason": reason}
        self.failed_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self, reason):
        self.failed_file.close()
        self.logger.info("Closed failed_de_links.jsonl")
        super().close(self, reason)

    def is_german(self, url):
        parsed = urlparse(url)
        path = parsed.path.lower()
        
        # Explicit German paths
        if '/de/' in path or path.endswith('/de') or '/de?' in path:
            return True
            
        # Default pages (no language prefix)
        if not any(f'/{lang}/' in path for lang in ['en', 'fr', 'es']):
            return parsed.netloc == 'arena2036.de'
            
        return False

    def should_follow(self, url):
        parsed = urlparse(url)
        
        # Only follow arena2036.de links
        if parsed.netloc != 'arena2036.de':
            return False
            
        # Skip non-German content
        if not self.is_german(url):
            return False
            
        # Skip file extensions
        path = parsed.path.lower()
        if any(path.endswith(ext) for ext in ['.pdf', '.jpg', '.png', '.zip', '.docx', '.mp4']):
            return False
            
        # Skip already visited
        if url in self.visited_urls:
            return False
            
        self.visited_urls.add(url)
        return True

    def extract_content(self, response):
        """Improved content extraction focusing on German content"""
        content_selectors = [
            'main', 'article', '.main-content', '.content',
            '.page-content', '#content', '.entry-content',
            'section', '.container', '.text-content'
        ]

        # Try specific selectors first
        for selector in content_selectors:
            content = response.css(selector)
            if content:
                break
        else:
            content = response.css('body')

        # Extract text with better German language handling
        text_nodes = content.xpath('''
            .//text()[not(
                ancestor::script or
                ancestor::style or
                ancestor::nav or
                ancestor::header or
                ancestor::footer or
                ancestor::*[contains(@class, 'cookie')] or
                ancestor::*[contains(@class, 'banner')] or
                ancestor::*[contains(@class, 'modal')]
            )]
        ''').getall()

        # Clean and process German text
        processed_text = []
        for text in text_nodes:
            if isinstance(text, str):
                cleaned = re.sub(r'\s+', ' ', text.strip())
                cleaned = re.sub(r'[^\wäöüßÄÖÜ\s\.,;:!?\-()[\]{}"\'/]', ' ', cleaned)
                if len(cleaned) > 2 and not cleaned.isspace():
                    # German-specific noise patterns
                    noise_patterns = [
                        r'^(zurück|weiter|mehr|menü|navigation|impressum|datenschutz)',
                        r'^[\d\s\-\./]+$',
                        r'^(cookie|bildnachweis)',
                        r'^©'
                    ]
                    if not any(re.search(p, cleaned.lower()) for p in noise_patterns):
                        processed_text.append(cleaned)

        final_content = ' '.join(processed_text)
        final_content = re.sub(r'\s+', ' ', final_content)
        final_content = re.sub(r'(\s*[.!?]\s*){2,}', '. ', final_content)
        return final_content.strip()


if __name__ == "__main__":
    process = CrawlerProcess()
    process.crawl(ArenaDeSpider)
    process.start()