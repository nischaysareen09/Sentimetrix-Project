import feedparser
import functools

class NewsEngine:
    def __init__(self):
        print("News Engine (RSS) Loaded.")

    @functools.lru_cache(maxsize=32)
    def fetch_company_news(self, ticker, limit=5):
        """
        Fetches news from Yahoo Finance AND Google News.
        """
        articles = []
        
        # 1. Yahoo Finance RSS
        try:
            yf_url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
            feed_yf = feedparser.parse(yf_url)
            for entry in feed_yf.entries[:3]: # Top 3
                articles.append({
                    "source": "Yahoo",
                    "title": entry.title,
                    "link": entry.link,
                    "published": entry.get("published", "Recent")
                })
        except Exception as e:
            print(f"Yahoo News Error: {e}")

        # 2. Google News RSS
        try:
            gn_url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
            feed_gn = feedparser.parse(gn_url)
            for entry in feed_gn.entries[:3]: # Top 3
                articles.append({
                    "source": "Google",
                    "title": entry.title,
                    "link": entry.link,
                    "published": entry.get("published", "Recent")
                })
        except Exception as e:
            print(f"Google News Error: {e}")

        if not articles:
            return "No recent news found.", []
            
        # Create context string
        context_string = "\n".join([f"- [{a['source']}] {a['title']}" for a in articles])
        
        return context_string, articles
