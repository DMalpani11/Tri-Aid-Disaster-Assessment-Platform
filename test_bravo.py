import json
import re
import requests
from typing import Dict, List
from newspaper import Article
from urllib.parse import urlparse
from bravo import (
    discoverty_agent,  # Note: there's a typo in your original code ("discoverty" instead of "discovery")
    static_summarizer,
    classification_agent
)
from google.adk.agents import WorkflowAgent

# Custom fetch and parse function to replace http_fetch
def fetch_and_parse_article(url: str) -> Dict:
    """
    Fetch and parse an article using newspaper3k
    
    Args:
        url: URL to fetch and parse
        
    Returns:
        Dict containing article title, text, publish date and authors
    """
    try:
        # For classification checks, we need raw HTML
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        raw_html = response.text
        
        # Use newspaper3k for article extraction
        article = Article(url)
        article.download()
        article.parse()
        
        return {
            "title": article.title,
            "text": article.text,
            "html": raw_html,
            "publish_date": article.publish_date,
            "authors": article.authors,
            "url": url
        }
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        # Fallback to basic request if newspaper fails
        try:
            response = requests.get(url, headers=headers, timeout=10)
            return {
                "title": "Unknown",
                "text": response.text[:10000],  # Limit text size
                "html": response.text,
                "publish_date": None,
                "authors": [],
                "url": url
            }
        except:
            return {
                "title": "Error",
                "text": f"Failed to fetch content from {url}",
                "html": "",
                "publish_date": None,
                "authors": [],
                "url": url
            }

# Custom classification function to replace the one in bravo.py
def classify_page(url: str) -> str:
    """
    Classify a page as 'live' or 'static' based on URL and content
    """
    try:
        # Check URL patterns first
        if any(term in url.lower() for term in ["live", "updates", "breaking", "latest"]):
            return "live"
            
        # Fetch the content
        article_data = fetch_and_parse_article(url)
        html = article_data["html"]
        
        # Look for live update indicators in the HTML
        live_indicators = [
            "setInterval", 
            "WebSocket", 
            "live-updates",
            "live-blog", 
            "liveBlog",
            "refreshContent",
            "auto-refresh"
        ]
        
        if any(indicator in html for indicator in live_indicators):
            return "live"
            
        # Additional heuristics
        if "minutes ago" in html.lower() or "seconds ago" in html.lower():
            return "live"
            
        # Default to static
        return "static"
    except Exception as e:
        print(f"Error classifying {url}: {e}")
        return "static"  # Default to static on error

def test_alappuzha_floods():
    """
    Test the integration of agents for analyzing Alappuzha floods information.
    """
    print("🔍 Starting investigation on Alappuzha floods...")
    
    # Step 1: Use discovery agent to find relevant URLs
    discovery_query = "Recent Alappuzha Kerala floods disaster relief efforts and impact"
    print(f"\n🌐 Running discovery query: '{discovery_query}'")
    
    try:
        discovery_response = discoverty_agent.invoke({
            "input": discovery_query,
            "instructions": "Find the most relevant and recent information about Alappuzha floods in Kerala, India. Include news articles, government updates, and relief efforts. Return 5 most relevant URLs with brief descriptions."
        })
        
        print("\n📋 Discovery Results:")
        print(discovery_response)
    except Exception as e:
        print(f"⚠️ Error using discovery agent: {e}")
        discovery_response = """
        Here are some sources about Alappuzha floods:
        1. https://www.thehindu.com/news/national/kerala/flood-situation-grim-in-alappuzha/article24709251.ece - Detailed report on flood situation
        2. https://www.ndtv.com/kerala-news/kerala-floods-alappuzha-worst-hit-2570321 - Impact assessment
        3. https://kerala.gov.in/alappuzha-district - Official district information
        4. https://timesofindia.indiatimes.com/city/kochi/kerala-floods-updates-alappuzha/articleshow/86534217.cms - Latest updates
        5. https://www.downtoearth.org.in/news/natural-disasters/kerala-floods-alappuzha-residents-return-to-damaged-houses-face-snake-menace-61244 - Long-term impacts
        """
    
    # Extract URLs from the discovery response
    urls = re.findall(r'https?://[^\s]+', discovery_response)
    if not urls:
        print("⚠️ No URLs found in the discovery response. Using manual examples.")
        # Fallback to example URLs if none found
        urls = [
            "https://www.thehindu.com/news/national/kerala/flood-situation-grim-in-alappuzha/article24709251.ece",
            "https://www.ndtv.com/kerala-news/kerala-floods-alappuzha-worst-hit-2570321",
            "https://kerala.gov.in/alappuzha-district",
            "https://timesofindia.indiatimes.com/city/kochi/kerala-floods-updates-alappuzha/articleshow/86534217.cms",
            "https://www.downtoearth.org.in/news/natural-disasters/kerala-floods-alappuzha-residents-return-to-damaged-houses-face-snake-menace-61244"
        ]
    
    print(f"\n🔗 Found {len(urls)} URLs to analyze:")
    for url in urls[:5]:  # Limit to 5 URLs
        print(f" - {url}")
    
    # Step 2: Analyze each URL
    summaries = []
    
    for i, url in enumerate(urls[:5]):
        print(f"\n🔍 Analyzing URL {i+1}/{min(5, len(urls))}: {url}")
        
        # Fetch and parse article
        print("📥 Fetching article...")
        article_data = fetch_and_parse_article(url)
        
        # Classify the page
        try:
            page_type = classify_page(url)
            print(f"📊 Page classified as: {page_type}")
            
            # Process based on page type
            if page_type == "static":
                print("📝 Processing with static summarizer...")
                try:
                    # Create a summary input with the article text
                    summary_input = f"""
                    Title: {article_data['title']}
                    URL: {url}
                    
                    Content:
                    {article_data['text'][:3000]}  # Limit text to 3000 chars
                    
                    Summarize the key information about Alappuzha floods from this article.
                    """
                    
                    summary = static_summarizer.invoke({
                        "input": summary_input,
                        "instructions": "Extract key facts about impact, casualties, relief efforts, and current situation. Be concise but comprehensive."
                    })
                    
                except Exception as e:
                    print(f"❌ Error using static summarizer: {e}")
                    # Fallback summary
                    summary = f"Failed to generate summary. Article title: {article_data['title']}"
                
                summaries.append({
                    "url": url,
                    "title": article_data['title'],
                    "type": "static",
                    "summary": summary
                })
            else:  # live
                print("🔴 Live page detected - noting for real-time monitoring")
                summaries.append({
                    "url": url,
                    "title": article_data['title'],
                    "type": "live",
                    "note": "Live updates page - should be monitored for real-time information"
                })
        except Exception as e:
            print(f"❌ Error processing URL: {e}")
    
    # Step 3: Compile final report
    print("\n\n📊 ALAPPUZHA FLOODS SITUATION REPORT 📊")
    print("=" * 50)
    
    # Static information summaries
    print("\n📚 STATIC INFORMATION:")
    static_count = 0
    for item in summaries:
        if item["type"] == "static":
            static_count += 1
            print(f"\n📄 SOURCE {static_count}: {item['title']}")
            print(f"URL: {item['url']}")
            print("-" * 40)
            print(item['summary'])
    
    # Live sources for monitoring
    print("\n🔴 LIVE SOURCES FOR MONITORING:")
    live_count = 0
    for item in summaries:
        if item["type"] == "live":
            live_count += 1
            print(f" {live_count}. {item['title']} - {item['url']}")
    
    print("\n" + "=" * 50)
    print("End of report")

if __name__ == "__main__":
    test_alappuzha_floods()