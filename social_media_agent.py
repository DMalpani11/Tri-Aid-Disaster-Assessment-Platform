import urllib.request
from PIL import Image
import io
import json
import pandas as pd
import base64
import httpx
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from apify_client import ApifyClient
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

import os
from dotenv import load_dotenv
load_dotenv()

# API Configuration
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN", "")
RAG_SERVER_URL = os.getenv("RAG_SERVER_URL", "http://localhost:8000")

# Data models for RAG integration
class SocialPost(BaseModel):
    content: str
    platform: str
    username: Optional[str] = None
    tags: Optional[List[str]] = None

class DisasterQuery(BaseModel):
    event_name: str
    start_date: str
    end_date: str
    query: Optional[str] = None
    max_results: int = 10

class ReportResponse(BaseModel):
    report: str
    sources: List[str]
    event_name: str
    start_date: str
    end_date: str

async def fetch_social_media_posts(query: DisasterQuery) -> pd.DataFrame:
    """
    Fetches social media posts related to a disaster using APIFY
    """
    print(f"🔍 Fetching social media content for: {query.event_name}")
    
    # Format search query
    location = query.event_name.split()[0]  # Extract location name (e.g., "Alappuzha" from "Alappuzha floods")
    search_query = f"flood OR damage OR rescue OR trapped {location}"
    
    # Initialize the ApifyClient
    client = ApifyClient(APIFY_API_TOKEN)
    
    # Twitter/X scraper actor ID
    ACTOR = "rBaTEHzveTxZPraGv"
    
    all_data = []
    
    # Build date ranges to search (work backwards from end_date to start_date)
    start = datetime.strptime(query.start_date, "%Y-%m-%d")
    end = datetime.strptime(query.end_date, "%Y-%m-%d")
    
    # Create date ranges to search (Twitter's API limits)
    date_points = []
    current = end
    while current >= start:
        date_points.append(current.strftime("%Y-%m-%d"))
        current -= timedelta(days=2)  # Search in 2-day chunks to get more results
    
    # Process each date point
    for date_str in date_points:
        print(f"📅 Searching social media until: {date_str}")
        
        # Prepare the Actor input
        run_input = {
            "query": f"{search_query} until:{date_str}",
            "resultsCount": query.max_results,
            "searchType": "latest"
        }
        
        try:
            # Run the Actor and wait for it to finish
            run = client.actor(ACTOR).call(run_input=run_input)
            
            # Fetch and process Actor results
            for item in client.dataset(run["defaultDatasetId"]).iterate_items():
                media_urls = []
                for media in item.get("media", []):
                    if "mediaUrlHttps" in media:
                        media_urls.append(media["mediaUrlHttps"])
                
                all_data.append({
                    "postText": item.get("postText", ""),
                    "timestamp": datetime.fromtimestamp(item.get("timestamp", 0) / 1000).isoformat(),
                    "postId": item.get("postId", ""),
                    "author": item.get("author", {}).get("name", "Unknown"),
                    "media_urls": media_urls,
                    "url": item.get("url", "")
                })
        except Exception as e:
            print(f"⚠️ Error fetching social media data: {str(e)}")
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Remove duplicates
    if not df.empty:
        df.drop_duplicates(subset=["postId"], keep="first", inplace=True)
    
    print(f"✅ Found {len(df)} unique social media posts")
    return df

async def store_posts_in_rag(df: pd.DataFrame, event_name: str) -> List[str]:
    """
    Store social media posts in RAG database
    """
    stored_ids = []
    
    for _, row in df.iterrows():
        try:
            # Prepare hashtags from content
            hashtags = []
            words = row["postText"].split()
            for word in words:
                if word.startswith('#'):
                    hashtags.append(word[1:])  # Remove the # symbol
            
            # Create post object
            post = SocialPost(
                content=row["postText"],
                platform="Twitter",  # Assuming Twitter/X for now
                username=row["author"],
                tags=hashtags
            )
            
            # Send to RAG
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{RAG_SERVER_URL}/add/social",
                    json=post.dict()
                )
                
                if response.status_code == 200:
                    result = response.json()
                    stored_ids.append(result.get("id", "unknown"))
                    print(f"✅ Stored post from {post.username}")
                else:
                    print(f"⚠️ Failed to store post: {response.status_code}")
                    print(response.text)
        
        except Exception as e:
            print(f"❌ Error storing post: {str(e)}")
    
    print(f"📊 Stored {len(stored_ids)} posts in RAG")
    return stored_ids

async def generate_social_media_report(query: DisasterQuery) -> Dict:
    """
    Generate a report based on social media posts
    """
    # Step 1: Query RAG for social media content
    print(f"🔍 Querying RAG for social media content about: {query.event_name}")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{RAG_SERVER_URL}/query",
            json={
                "query": f"{query.event_name} disaster {query.start_date} {query.end_date}",
                "collections": ["socials"],
                "limit": 25  # Fetch more social posts for better coverage
            }
        )
        
        if response.status_code != 200:
            print(f"⚠️ Failed to query RAG: {response.status_code}")
            return {
                "report": "Error fetching social media content",
                "sources": [],
                "event_name": query.event_name
            }
        
        results = response.json()
    
    # Step 2: Check if we have sufficient data
    if not results or "results" not in results or len(results["results"]) < 3:
        print("⚠️ Insufficient social media data in RAG")
        
        # Fallback: Fetch fresh social media content
        df = await fetch_social_media_posts(query)
        
        if df.empty:
            return {
                "report": "No social media content available for this disaster",
                "sources": [],
                "event_name": query.event_name
            }
        
        # Store posts in RAG
        await store_posts_in_rag(df, query.event_name)
        
        # Query RAG again
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{RAG_SERVER_URL}/query",
                json={
                    "query": f"{query.event_name} disaster {query.start_date} {query.end_date}",
                    "collections": ["socials"],
                    "limit": 25
                }
            )
            
            if response.status_code != 200:
                print(f"⚠️ Failed to query RAG after storing: {response.status_code}")
                
                # Use directly fetched data instead
                social_posts = []
                sources = []
                
                for _, row in df.iterrows():
                    social_posts.append(f"User @{row['author']} posted: {row['postText']}")
                    if "url" in row and row["url"]:
                        sources.append(row["url"])
            else:
                results = response.json()
                
                # Extract posts and sources
                social_posts = []
                sources = []
                
                for item in results.get("results", []):
                    post_content = item["content"]
                    username = item["metadata"].get("username", "Unknown")
                    social_posts.append(f"User @{username} posted: {post_content}")
                    
                    # No source URLs in social media posts in this implementation
    else:
        # Extract posts and sources from RAG results
        social_posts = []
        sources = []
        
        for item in results.get("results", []):
            post_content = item["content"]
            username = item["metadata"].get("username", "Unknown")
            social_posts.append(f"User @{username} posted: {post_content}")
            
            # No source URLs in social media posts in this implementation
    
    # Step 3: Generate social media report
    from google.genai import Client as GenAIClient
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    client = GenAIClient(api_key=os.environ.get("GEMINI_API_KEY"))
    model = client.get_model("gemini-2.0-flash")
    
    social_content = "\n\n".join(social_posts)
    
    prompt = f"""
    Generate a social media analysis report about the {query.event_name} disaster that occurred between 
    {query.start_date} and {query.end_date}.
    
    Use ONLY these social media posts as your source:
    
    {social_content}
    
    Your report should focus on:
    
    ## Social Media Insights: {query.event_name}
    
    ## Public Sentiment
    - How are people feeling and reacting to the disaster?
    - What emotions are prevalent in the posts?
    
    ## Eyewitness Reports
    - Extract firsthand accounts from people on the ground
    - Highlight specific incidents reported by users
    
    ## Urgent Needs & Requests
    - What help are people asking for?
    - What resources are most needed?
    
    ## Misinformation Analysis
    - Note any contradictory information in the posts
    - Highlight information that needs verification
    
    ## Timeline of Events
    - Create a chronological summary of developments as reported on social media
    
    Format the report in Markdown with clear headings and structure.
    """
    
    try:
        response = model.generate_content(prompt)
        social_report = response.text
    except Exception as e:
        print(f"❌ Error generating social media report: {str(e)}")
        social_report = f"Error generating social media report: {str(e)}"
    
    # Return the report
    return {
        "report": social_report,
        "sources": sources,
        "event_name": query.event_name
    }

async def generate_combined_report(query: DisasterQuery) -> Dict:
    """
    Generate a comprehensive report combining news and social media
    """
    # Step 1: Get news report from bravo-backend
    print("🔍 Fetching news report...")
    news_report = {}
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "http://localhost:8080/compile_report_from_rag",
                json=query.dict()
            )
            
            if response.status_code == 200:
                news_report = response.json()
            else:
                print(f"⚠️ Failed to get news report: {response.status_code}")
                print(response.text)
                news_report = {
                    "report": "Error fetching news report",
                    "sources": [],
                }
    except Exception as e:
        print(f"❌ Error getting news report: {str(e)}")
        news_report = {
            "report": "Error fetching news report",
            "sources": [],
        }
    
    # Step 2: Get social media report
    print("📱 Fetching social media report...")
    social_report = await generate_social_media_report(query)
    
    # Step 3: Combine the reports
    from google.genai import Client as GenAIClient
    import os
    
    client = GenAIClient(api_key=os.environ.get("GEMINI_API_KEY"))
    model = client.get_model("gemini-2.0-flash")
    
    prompt = f"""
    Create a comprehensive disaster report by combining the official news coverage with social media insights.
    
    OFFICIAL NEWS REPORT:
    {news_report.get("report", "No news report available.")}
    
    SOCIAL MEDIA INSIGHTS:
    {social_report.get("report", "No social media insights available.")}
    
    Your combined report should:
    1. Start with a comprehensive executive summary incorporating both official and social perspectives
    2. Present the official information about the disaster (from news sources)
    3. Complement it with eyewitness accounts and public sentiment (from social media)
    4. Highlight any discrepancies or additional information from social media not covered in news
    5. Create a comprehensive timeline merging both official and social media sources
    
    USE THIS STRUCTURE:
    
    # Comprehensive Report: {query.event_name}
    
    ## Executive Summary
    [Integrated summary from both sources]
    
    ## Official Impact Assessment
    [From news sources]
    
    ## Public Experience & Eyewitness Accounts
    [From social media]
    
    ## Response Efforts
    ### Official Response
    [From news]
    ### Community Response
    [From social media]
    
    ## Areas of Concern
    [Highlight needs, challenges from both sources]
    
    ## Information Verification
    [Note any contradictory information between sources]
    
    ## Comprehensive Timeline
    [Merged timeline from both official and social sources]
    
    ## Sources
    [List all sources]
    
    Format the report in Markdown with clear headings and structure.
    """
    
    try:
        response = model.generate_content(prompt)
        combined_report = response.text
    except Exception as e:
        print(f"❌ Error generating combined report: {str(e)}")
        combined_report = f"Error generating combined report: {str(e)}"
    
    # Combine sources
    all_sources = []
    all_sources.extend(news_report.get("sources", []))
    all_sources.extend(social_report.get("sources", []))
    
    # Save the combined report
    report_filename = f"combined_report_{query.event_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_filename, 'w') as f:
        f.write(combined_report)
    
    print(f"✅ Combined report saved to {report_filename}")
    
    # Return the report
    return {
        "report": combined_report,
        "sources": all_sources,
        "event_name": query.event_name,
        "start_date": query.start_date,
        "end_date": query.end_date
    }

# Synchronous wrapper for running the async functions
def fetch_and_store_social_media(query_dict):
    """Wrapper to run the async function synchronously"""
    query = DisasterQuery(**query_dict)
    loop = asyncio.get_event_loop()
    df = loop.run_until_complete(fetch_social_media_posts(query))
    stored_ids = loop.run_until_complete(store_posts_in_rag(df, query.event_name))
    return {"stored_count": len(stored_ids), "posts": len(df)}

def generate_social_report(query_dict):
    """Generate social media report synchronously"""
    query = DisasterQuery(**query_dict)
    loop = asyncio.get_event_loop()
    report = loop.run_until_complete(generate_social_media_report(query))
    return report

def generate_complete_report(query_dict):
    """Generate combined report synchronously"""
    query = DisasterQuery(**query_dict)
    loop = asyncio.get_event_loop()
    report = loop.run_until_complete(generate_combined_report(query))
    return report

# Main execution for testing
if __name__ == "__main__":
    test_query = {
        "event_name": "Kerala floods",
        "start_date": "2018-08-08",
        "end_date": "2018-08-17",
        "max_results": 10
    }
    
    print("Starting social media processing...")
    result = fetch_and_store_social_media(test_query)
    print(f"Stored {result['stored_count']} posts from {result['posts']} found posts")
    
    print("\nGenerating social media report...")
    social_report = generate_social_report(test_query)
    print(f"Social media report length: {len(social_report['report'])}")
    
    print("\nGenerating combined report...")
    combined_report = generate_complete_report(test_query)
    print(f"Combined report saved with {len(combined_report['report'])} characters")