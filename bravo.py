import asyncio
import json
import requests
from newspaper import Article
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from typing import Dict, List
from dotenv import load_dotenv
import os

# ADK core
from google.adk.agents import Agent, SequentialAgent    # Agent (alias for LlmAgent) and workflow agent :contentReference[oaicite:0]{index=0}
from google.adk.tools import google_search              # Built-in Google Search tool :contentReference[oaicite:1]{index=1}
from google.adk.runners import Runner                   # Manages execution & streaming :contentReference[oaicite:2]{index=2}
from google.adk.sessions import InMemorySessionService    # Session API interface :contentReference[oaicite:3]{index=3}
from google.genai import types as genai_types           # For composing Content parts
from google.genai import Client as genai_client      # For API client


load_dotenv()  # Load environment variables from .env file
def fetch_content(url: str) -> Dict:
    """
    Try newspaper3k, then fall back to BeautifulSoup.
    Returns title, text, raw HTML, metadata, and status.
    """
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        article = Article(url); article.download(); article.parse()
        raw_html = requests.get(url, headers=headers, timeout=10).text
        return {
            "title": article.title,
            "text": article.text,
            "html": raw_html,
            "publish_date": article.publish_date,
            "authors": article.authors,
            "status": "success",
            "method": "newspaper3k"
        }
    except Exception:
        # BS4 fallback
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        title = soup.title.string if soup.title else ""
        text = soup.get_text(separator=' ', strip=True)
        return {
            "title": title,
            "text": text,
            "html": resp.text,
            "publish_date": None,
            "authors": [],
            "status": "success",
            "method": "bs4"
        }

# Update the html_scraper function to save content to files

def html_scraper(url: str) -> str:
    """Extracts text body for summarization and saves both HTML and text for debugging."""
    import os
    import hashlib
    import requests
    from datetime import datetime
    from newspaper import Article
    from bs4 import BeautifulSoup
    
    # Create logs directory if it doesn't exist
    logs_dir = "scraper_logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create unique filename based on URL and timestamp
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{logs_dir}/{timestamp}_{url_hash}"
    
    print(f"\n🔍 Scraping URL: {url}")
    
    try:
        # First try with newspaper3k
        article = Article(url)
        article.download()
        article.parse()
        
        # Get raw HTML for backup and classification
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=20)
        raw_html = response.text
        
        # Extract content from the article
        title = article.title
        text = article.text
        publish_date = article.publish_date
        authors = article.authors
        method = "newspaper3k"
        
        # Fall back to BS4 if newspaper3k doesn't get good content
        if not text or len(text) < 100:
            print("⚠️ Newspaper3k extracted minimal text, trying BS4")
            soup = BeautifulSoup(raw_html, 'html.parser')
            
            # Try to get title if missing
            if not title and soup.title:
                title = soup.title.string
                
            # Extract main content using common patterns
            main_content = soup.find(['main', 'article', 'div', 'section'], 
                                     class_=['content', 'article', 'main-content', 'story', 'entry-content'])
            
            if main_content:
                # Remove scripts, styles, and other non-content elements
                for tag in main_content(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                    tag.decompose()
                
                text = main_content.get_text(separator=' ', strip=True)
            else:
                # Fall back to full page text with some cleaning
                text = soup.get_text(separator=' ', strip=True)
            
            method = "bs4"
        
        # Save raw HTML
        html_filename = f"{base_filename}_raw.html"
        with open(html_filename, "w", encoding="utf-8") as f:
            f.write(raw_html)
        
        # Save extracted text
        text_filename = f"{base_filename}_text.txt"
        with open(text_filename, "w", encoding="utf-8") as f:
            f.write(f"TITLE: {title}\n\n")
            f.write(f"URL: {url}\n\n")
            f.write(text)
        
        # Save metadata
        meta_filename = f"{base_filename}_meta.json"
        import json
        meta = {
            "url": url,
            "title": title,
            "timestamp": timestamp,
            "method": method,
            "status": "success",
            "publish_date": str(publish_date) if publish_date else None,
            "authors": authors,
            "text_length": len(text)
        }
        with open(meta_filename, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        
        print(f"✅ Extracted {len(text)} characters from {url}")
        print(f"📄 Saved to: {text_filename}")
        
        # Return structured content for the agent to use
        return f"""
TITLE: {title}

URL: {url}

CONTENT:
{text[:3000]}  # Limit to 3000 chars to avoid token limits

[Text continues - {len(text)} total characters]
"""
        
    except Exception as e:
        print(f"❌ Error scraping {url}: {str(e)}")
        error_filename = f"{base_filename}_error.txt"
        with open(error_filename, "w", encoding="utf-8") as f:
            f.write(f"URL: {url}\n\nError: {str(e)}")
        
        return f"Failed to scrape content from {url}. Error: {str(e)}"
def process_url(url: str) -> dict:
    """Process a single URL to extract content and generate a summary"""
    import json
    
    print(f"\n📝 Processing URL: {url}")
    
    try:
        # 1. Extract content
        content = html_scraper(url)
        
        # 2. Generate summary using Gemini
        client = genai_client(api_key=os.environ.get("GEMINI_API_KEY"))  # or your preferred model
        
        prompt = f"""
        Summarize the key information about the disaster event from this content:
        
        {content}
        
        Focus on:
        1. What happened
        2. When it happened
        3. Impact (casualties, damage, displaced people)
        4. Response efforts
        5. Current status
        
        Provide a detailed but concise summary (250-300 words).
        """
        
        response = client.models.generate_content(
            contents = prompt,
            model = "gemini-2.0-flash"
        )
        summary = response.text
        
        print(f"✅ Successfully summarized content from {url}")
        
        # Save summary to file
        import hashlib
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        summary_filename = f"summaries/{url_hash}_summary.txt"
        
        os.makedirs("summaries", exist_ok=True)
        with open(summary_filename, "w", encoding="utf-8") as f:
            f.write(f"URL: {url}\n\n")
            f.write(summary)
        
        # Return structured result
        return {
            "url": url,
            "summary": summary,
            "summary_file": summary_filename
        }
        
    except Exception as e:
        print(f"❌ Error processing {url}: {str(e)}")
        return {
            "url": url,
            "error": str(e)
        }
def classify_page(url: str) -> str:
    """
    Heuristic classification as 'live' vs 'static'.
    Checks URL tokens and inline JS indicators.
    """
    u = url.lower()
    if "live" in u or "updates" in u:
        return "live"
    html = fetch_content(url)["html"] or ""
    if any(token in html for token in ["setInterval", "WebSocket", "live-updates"]):
        return "live"
    return "static"

# 3.1 Discovery: Google‐search for URLs within the date range
discovery_agent = Agent(
    name="DiscoveryAgent",
    model="gemini-2.0-flash",
    instruction="""
        You will receive a query about a disaster event.
        Extract the event name and date range from the query.
        Search Google News for relevant articles about this event within the date range.
        Return a JSON array of URLs with brief descriptions.
    """,
    tools=[google_search]
)

# 3.2 Static Summarizer: Scrape & summarize one‐off articles
static_summarizer = Agent(
    name="StaticSummarizer",
    model="gemini-2.0-flash",
    instruction="""
        For each URL provided, use the html_scraper tool to extract the text.
        Then extract key facts and produce a structured summary of the disaster information.
    """,
    tools=[html_scraper]
)

# 3.3 Tracker: Monitor live pages for diffs and summarize changes
tracker_agent = Agent(
    name="TrackerAgent",
    model="gemini-2.0-flash",
    instruction="""
        For the provided URLs:
        1. Use classify_page to determine if each URL is a live page with updates
        2. For live pages, use html_scraper to monitor content changes
        3. Summarize each update with its timestamp
    """,
    tools=[html_scraper, classify_page]
)

# 3.4 Report Generator: Combine all summaries into a final report
report_agent = Agent(
    name="ReportAgent",
    model="gemini-2.0-flash",
    instruction="""
        You will receive JSON data about a disaster event with summaries for each source.
        
        Your task is to create a comprehensive disaster report using ONLY the information in these summaries.
        
        The report MUST include:
        1. Title: Clear title for the disaster report
        2. Executive summary: Brief overview of the event (2-3 paragraphs)
        3. Impact assessment: Details on affected areas, casualties, damage
        4. Response efforts: What actions were taken by authorities and organizations
        5. Current status: The situation as of the latest information
        6. Sources: List ALL URLs from the input JSON as numbered references
        
        IMPORTANT: For each fact in your report, include the source URL in [brackets].
        
        Format the report in Markdown with clear headings (##) and good structure.
        
        THIS IS A FINAL REPORT that will be presented to decision makers. Make it professional,
        comprehensive, and well-organized.
    """
)
# Update the pipeline definition to improve data flow

pipeline = SequentialAgent(
    name="DisasterPipeline",
    sub_agents=[
        Agent(                 # Step 1: find URLs with clearer output format
            name="DiscoveryAgent",
            model="gemini-2.0-flash",
            instruction="""
                You will receive a query about a disaster event.
                Extract the event name and date range from the query.
                Search Google News for relevant articles about this event within the date range.
                Return a structured JSON with this exact format:
                {
                  "event_name": "name of disaster",
                  "start_date": "YYYY-MM-DD",
                  "end_date": "YYYY-MM-DD",
                  "sources": [
                    {"url": "full_url_1", "title": "article title", "description": "brief description"},
                    {"url": "full_url_2", "title": "article title", "description": "brief description"}
                  ]
                }
            """,
            tools=[google_search]
        ),
        Agent(                 # Step 2: classify each URL
            name="Classifier",
            model="gemini-2.0-flash",
            instruction="""
                You will receive JSON data about a disaster event with URLs.
                Use the classify_page tool to check if each URL is 'static' or 'live'.
                Return the original JSON with an added 'type' field for each source:
                {
                  "event_name": "...",
                  "start_date": "...",
                  "end_date": "...",
                  "sources": [
                    {"url": "...", "title": "...", "description": "...", "type": "static or live"},
                    ...
                  ]
                }
            """,
            tools=[classify_page]
        ),
        Agent(                 # Step 3: summarize content from each URL
            name="ContentProcessor",
            model="gemini-2.0-flash",
            instruction="""
                You will receive JSON data about a disaster event with classified URLs.
                
                For EACH URL in the sources list:
                1. Use the html_scraper tool to extract the full text from that URL
                2. After extracting text from EACH URL, create a concise but detailed 
                   summary (200-300 words) of the most important information about the disaster
                3. ADD this summary as a new 'summary' field to each source object
                
                Process all URLs (up to 5 maximum).
                
                IMPORTANT: You MUST maintain the original JSON structure and add the summary field to EACH source.
                
                Example of expected output format:
                {
                  "event_name": "...",
                  "start_date": "...",
                  "end_date": "...",
                  "sources": [
                    {
                      "url": "...", 
                      "title": "...", 
                      "description": "...", 
                      "type": "...",
                      "summary": "Detailed summary of information from this source..."
                    },
                    ...
                  ]
                }
            """,
            tools=[html_scraper]
        ),
        Agent(                 # Step 4: generate final report
            name="ReportAgent",
            model="gemini-2.0-flash", 
            instruction="""
                You will receive JSON data about a disaster event with summaries for each source.
                Create a comprehensive report about the disaster event including:
                
                1. Executive summary of the event
                2. Main impacts and affected areas
                3. Response efforts
                4. Current status
                
                IMPORTANT: For each section, cite the specific sources used by including the URL in [brackets].
                Include a "Sources" section at the end with a numbered list of all URLs.
                
                Format the report in Markdown with clear headings and structure.
            """
        )
    ]
)

# 5.1 Create Runner
# 5.1 Session Service & Runner
session_service = InMemorySessionService()   # In-memory storage for testing :contentReference[oaicite:4]{index=4}
runner = Runner(
    agent=pipeline,
    app_name="disaster_reporting_app",
    session_service=session_service        # Required keyword-only argument :contentReference[oaicite:5]{index=5}
)

# 5.2 Create or retrieve a session
session = session_service.create_session(
    app_name="disaster_reporting_app",
    user_id="user123",
    session_id="session_001"
)
SESSION_ID = session.id
USER_ID = session.user_id

# Update the call_agent function to properly display intermediate results and handle errors

def call_agent(query: str) -> str:
    """
    Call the agent pipeline and properly handle all events, showing intermediate results.
    
    Args:
        query: The disaster query to process
        
    Returns:
        The final disaster report
    """
    print("\n🔍 Starting disaster information pipeline...")
    print(f"Query: {query}")
    
    content = genai_types.Content(role="user", parts=[genai_types.Part(text=query)])
    
    # Store intermediate and final results
    discovery_results = None
    classification_results = None
    content_processing_results = None
    final_report = None
    
    # Process all events
    try:
        events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)
        
        for evt in events:
            # Handle interim results to show progress
            if hasattr(evt, 'agent_name') and hasattr(evt, 'content'):
                agent_name = evt.agent_name
                if evt.content and hasattr(evt.content, 'parts') and evt.content.parts:
                    result = evt.content.parts[0].text
                    
                    # Store results based on agent name
                    if "DiscoveryAgent" in agent_name:
                        print("\n📰 Discovery Agent completed")
                        discovery_results = result
                        
                    elif "Classifier" in agent_name:
                        print("\n🔍 Classifier Agent completed")
                        classification_results = result
                        
                    elif "ContentProcessor" in agent_name:
                        print("\n📝 Content Processing completed")
                        content_processing_results = result
                        
                    elif "ReportAgent" in agent_name:
                        print("\n📊 Report Generation completed")
                        final_report = result
            
            # Final response handling
            if evt.is_final_response():
                if not final_report:  # If we haven't stored it yet
                    final_report = evt.content.parts[0].text
                    print("\n✅ Final report generated")
        
        # If we somehow didn't get a final report but have content processing results
        if not final_report and content_processing_results:
            print("\n⚠️ Final report not generated by pipeline, generating manually...")
            
            # Create a manual report using the content processing results
            client = genai_client(api_key=os.environ.get("GEMINI_API_KEY"))
            
            prompt = f"""
            Generate a comprehensive disaster report from this JSON data:
            
            {content_processing_results}
            
            Your report should include:
            1. Executive summary of the event
            2. Main impacts and affected areas
            3. Response efforts
            4. Current status
            
            IMPORTANT: For each section, cite the specific sources used by including the URL in [brackets].
            Include a "Sources" section at the end with a numbered list of all URLs.
            
            Format the report in Markdown with clear headings and structure.
            """
            genai_client = genai_client(api_key=os.environ.get("GEMINI_API_KEY"))
            response = genai_client.models.generate_content(
                contents = prompt,
                model = "gemini-2.0-flash"
            )
            final_report = response.candidates[0].content.parts[0].text
            print("✅ Manual report generated")
        
        # Add this debugging code in your call_agent function

        # After processing events, add this validation check:
        if content_processing_results:
            try:
                # Check if it's valid JSON
                data = json.loads(content_processing_results)
                
                # Check if sources have summaries
                has_summaries = False
                if "sources" in data:
                    for source in data["sources"]:
                        if "summary" in source and source["summary"]:
                            has_summaries = True
                            break
                
                if not has_summaries:
                    print("⚠️ Warning: No summaries found in processed content")
                    # Generate summaries manually if needed
            except:
                print("⚠️ Warning: Content processing results not valid JSON")
        
        return final_report or "Error: No report generated"
        
    except Exception as e:
        import traceback
        print(f"\n❌ Error in pipeline: {str(e)}")
        traceback.print_exc()
        return f"Error processing query: {str(e)}"

# Update the main execution block to save the report to a file
def manual_pipeline(json_file_path: str) -> str:
    """
    Manual pipeline to process disaster information and generate a report
    
    Args:
        json_file_path: Path to the JSON file with discovered URLs
        
    Returns:
        Generated report text
    """
    import json
    import os
    
    print("\n" + "="*60)
    print("🔄 STARTING MANUAL DISASTER INFORMATION PIPELINE")
    print("="*60)
    
    # 1. Load the JSON data
    try:
        with open(json_file_path, 'r') as f:
            content = f.read()
            # Remove markdown code blocks if present
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            data = json.loads(content)
            
        print(f"✅ Loaded JSON data with {len(data['sources'])} sources")
    except Exception as e:
        print(f"❌ Error loading JSON: {str(e)}")
        return f"Failed to load JSON: {str(e)}"
    
    # 2. Process each URL to get summaries
    summaries = []
    for i, source in enumerate(data['sources'][:5]):  # Limit to 5 sources
        print(f"\n📊 Processing source {i+1}/{min(5, len(data['sources']))}")
        url = source['url']
        
        if "google.com/search" in url:
            print(f"⏩ Skipping Google search URL: {url}")
            continue
            
        result = process_url(url)
        if "error" not in result:
            source["summary"] = result["summary"]
            summaries.append(result["summary"])
        else:
            source["summary"] = f"Failed to process: {result['error']}"
    
    # 3. Save the enhanced JSON with summaries
    enhanced_json_path = "enhanced_disaster_data.json"
    with open(enhanced_json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\n✅ Enhanced JSON saved to {enhanced_json_path}")
    
    # 4. Generate the final report
    print("\n📝 Generating final disaster report...")
    
    client = genai_client(api_key=os.environ.get("GEMINI_API_KEY"))  # or your preferred model
    
    # Create a comprehensive prompt with all the summaries
    summaries_text = "\n\n".join([f"SOURCE {i+1}:\n{summary}" for i, summary in enumerate(summaries)])
    
    prompt = f"""
    Generate a comprehensive disaster report about the {data['event_name']} that occurred between 
    {data['start_date']} and {data['end_date']}.
    
    Use the following summaries from various sources:
    
    {summaries_text}
    
    Your report should include:
    
    ## Disaster Event: {data['event_name']}
    ## Executive Summary
    [2-3 paragraphs overview]
    
    ## Impact Assessment
    - Affected areas
    - Casualties and injuries
    - Property damage
    - Displacement statistics
    
    ## Response Efforts
    - Government actions
    - Relief operations
    - Rescue efforts
    
    ## Current Status
    - Latest situation
    - Recovery progress
    - Ongoing challenges
    
    ## Sources
    [List all sources with URLs]
    
    For each fact, cite the corresponding source URL in [brackets].
    Format the report in Markdown with clear headings and good structure.
    """
    client = genai_client(api_key=os.environ.get("GEMINI_API_KEY"))  # or your preferred model
    try:
        response = client.models.generate_content(
            contents = prompt,
            model = "gemini-2.0-flash"
        )
        report = response.text
        
        # Save the report
        report_filename = f"disaster_report_{data['event_name'].replace(' ', '_')}.md"
        with open(report_filename, 'w') as f:
            f.write(report)
            
        print(f"\n✅ Final report saved to {report_filename}")
        return report
        
    except Exception as e:
        print(f"❌ Error generating report: {str(e)}")
        return f"Failed to generate report: {str(e)}"

if __name__ == "__main__":
    import sys
    
    # Check if a JSON file path is provided as an argument
    if len(sys.argv) > 1:
        json_file_path = sys.argv[1]
    else:
        # Use the most recent disaster report file
        import glob
        json_files = sorted(glob.glob("disaster_report_*.txt"), reverse=True)
        if json_files:
            json_file_path = json_files[0]
            print(f"Using most recent discovery file: {json_file_path}")
        else:
            print("No discovery files found. Please run the discovery agent first.")
            sys.exit(1)
    
    # Run the manual pipeline
    report = manual_pipeline(json_file_path)
    
    print("\n" + "="*60)
    print("📋 FINAL DISASTER REPORT")
    print("="*60)
    print(report)



