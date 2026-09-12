import json
import os
import asyncio
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional

# FastAPI imports
from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel
import uvicorn
import httpx

# Web scraping
from newspaper import Article
from bs4 import BeautifulSoup

# Google services
from google import genai
from dotenv import load_dotenv

# Import working functions directly from bravo.py
from bravo import (
    process_url,
    html_scraper,
    google_search,
    call_agent,
    manual_pipeline
)

# Load environment variables
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
# Initialize FastAPI
app = FastAPI(title="Disaster API with RAG Integration")

# RAG server configuration
RAG_SERVER_URL = "http://localhost:8000"  # Default URL for RAG.py

#-------------------------------------------
# Data Models
#-------------------------------------------

class DisasterQuery(BaseModel):
    event_name: str
    start_date: str
    end_date: str
    query: Optional[str] = None
    max_results: int = 10

class NewsItem(BaseModel):
    title: str
    content: str
    source: Optional[str] = None
    url: Optional[str] = None

class ReportResponse(BaseModel):
    report: str
    sources: List[str]
    event_name: str
    start_date: str
    end_date: str

#-------------------------------------------
# RAG Integration Functions
#-------------------------------------------

async def store_in_rag(title: str, content: str, url: str) -> bool:
    """Store article facts in RAG"""
    try:
        news_item = NewsItem(
            title=title,
            content=content,
            source="disaster_api",
            url=url
        )
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{RAG_SERVER_URL}/add/news",
                json=news_item.dict()
            )
            
            if response.status_code == 200:
                print(f"✅ Stored in RAG: {title}")
                return True
            else:
                print(f"⚠️ RAG storage failed: {response.status_code}")
                print(response.text)
                return False
    except Exception as e:
        print(f"❌ RAG storage error: {str(e)}")
        return False

async def query_rag(query_text: str, limit: int = 15) -> Dict:
    """Query RAG for facts about disasters"""
    try:
        print(f"🔍 Querying RAG: {query_text}")
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f"{RAG_SERVER_URL}/query",
                json={
                    "query": query_text,
                    "collections": ["news"],
                    "limit": limit
                }
            )
            
            if response.status_code != 200:
                print(f"⚠️ RAG query failed: {response.status_code}")
                print(response.text)
                return {"results": []}
            
            return response.json()
    except Exception as e:
        print(f"❌ RAG query error: {str(e)}")
        return {"results": []}

#-------------------------------------------
# Main API Endpoints
#-------------------------------------------

@app.post("/generate_report")
async def generate_report_endpoint(query: DisasterQuery):
    """Generate disaster report with fresh search results and store in RAG"""
    try:
        print(f"🔍 Generating report for: {query.event_name}")
        
        # STEP 1: Search for fresh information
        search_query = f"{query.event_name} disaster {query.start_date} {query.end_date} news reports"
        print(f"🌐 Searching: {search_query}")
        
        # Use Google Search from bravo.py to get fresh results
        search_results = google_search.run(
            search_query,
            num_results=query.max_results
        )
        
        # STEP 2: Prepare data structure for processing
        temp_data = {
            "event_name": query.event_name,
            "start_date": query.start_date,
            "end_date": query.end_date,
            "sources": []
        }
        
        # Extract URLs and add to sources
        urls_found = 0
        for result in search_results:
            if hasattr(result, 'url') and result.url:
                urls_found += 1
                temp_data["sources"].append({
                    "url": result.url,
                    "title": getattr(result, 'title', result.url),
                    "description": getattr(result, 'snippet', "")
                })
                
        print(f"✅ Found {urls_found} relevant URLs")
        
        if urls_found == 0:
            return {"error": "No relevant URLs found for this disaster"}
            
        # STEP 3: Save temp data for processing
        temp_file = f"temp_{query.event_name.replace(' ', '_')}.json"
        with open(temp_file, "w") as f:
            json.dump(temp_data, f, indent=2)
        
        # STEP 4: Process each URL immediately (not in background)
        # Process URLs synchronously to ensure data is in RAG before report generation
        print("📊 Processing and storing URLs in RAG...")
        for source in temp_data["sources"]:
            if "url" in source:
                url = source["url"]
                try:
                    # Use bravo's extraction
                    content = html_scraper(url)
                    
                    # Extract facts using process_url
                    processed = process_url(url)
                    
                    # Store in RAG synchronously
                    if "facts" in processed:
                        # Use synchronous request for immediate storage
                        import requests
                        response = requests.post(
                            f"{RAG_SERVER_URL}/add/news",
                            json={
                                "title": processed.get("title", url),
                                "content": processed["facts"],
                                "source": "disaster_api",
                                "url": url
                            },
                            timeout=10
                        )
                        
                        if response.status_code == 200:
                            print(f"✅ Stored {url} in RAG")
                        else:
                            print(f"⚠️ Failed to store {url} in RAG: {response.status_code}")
                    else:
                        print(f"⚠️ No facts extracted from {url}")
                        
                except Exception as e:
                    print(f"❌ Error processing {url}: {str(e)}")
        
        # STEP 5: Generate the report using manual_pipeline from bravo
        print("📝 Generating report...")
        report = manual_pipeline(temp_file)
        
        # STEP 6: Return the report with sources
        sources = [source["url"] for source in temp_data["sources"] if "url" in source]
        
        return ReportResponse(
            report=report,
            sources=sources,
            event_name=query.event_name,
            start_date=query.start_date,
            end_date=query.end_date
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

async def process_and_store_in_rag(url: str):
    """Process a URL and store the facts in RAG"""
    try:
        # Use existing process_url from bravo.py
        result = process_url(url)
        
        if "facts" in result:
            # Store facts in RAG
            await store_in_rag(
                title=result.get("title", url),
                content=result["facts"],
                url=url
            )
            print(f"✅ Processed and stored {url} in RAG")
        else:
            print(f"⚠️ No facts extracted from {url}")
    except Exception as e:
        print(f"❌ Error processing {url}: {str(e)}")

@app.post("/add_url")
async def add_url_endpoint(url: str = Query(...)):
    """Process a specific URL and add to RAG"""
    try:
        # Start background task to process and store URL
        background_task = BackgroundTasks()
        background_task.add_task(process_and_store_in_rag, url)
        
        return {
            "status": "processing",
            "url": url,
            "message": "URL is being processed and will be added to RAG"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/query_facts")
async def query_facts_endpoint(query: str = Query(...), limit: int = 10):
    """Query RAG for disaster facts - this is the endpoint that wasn't working"""
    try:
        print(f"📊 Querying facts: {query}")
        # Make a direct POST request to RAG with debugging
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f"{RAG_SERVER_URL}/query",
                json={
                    "query": query,
                    "collections": ["news"],
                    "limit": limit
                }
            )
            
            if response.status_code != 200:
                print(f"⚠️ RAG API error: {response.status_code}")
                print(f"Error details: {response.text}")
                raise HTTPException(status_code=response.status_code, detail=response.text)
                
            result = response.json()
            print(f"✅ Query returned {len(result.get('results', []))} results")
            return result
    except httpx.HTTPError as e:
        print(f"❌ HTTP error querying RAG: {str(e)}")
        raise HTTPException(status_code=502, detail=f"RAG service error: {str(e)}")
    except Exception as e:
        print(f"❌ Error querying facts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Check health of API and RAG connection"""
    try:
        # Check RAG connection
        rag_status = "unknown"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{RAG_SERVER_URL}/health")
                if response.status_code == 200:
                    rag_status = "connected"
                    rag_data = response.json()
                    # Test a simple query to ensure full functionality
                    test_query = await client.post(
                        f"{RAG_SERVER_URL}/query", 
                        json={"query": "test", "collections": ["news"], "limit": 1}
                    )
                    if test_query.status_code == 200:
                        rag_status = "fully_functional"
                else:
                    rag_status = f"error: {response.status_code}"
                    rag_data = {"error": response.text}
        except Exception as e:
            rag_status = f"disconnected: {str(e)}"
            rag_data = {"error": str(e)}
            
        return {
            "status": "healthy",
            "timestamp": str( datetime.datetime.now()),
            "rag_connection": rag_status,
            "rag_details": rag_data,
            "environment": {
                "gemini_api": "configured" if os.getenv("GEMINI_API_KEY") else "missing"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/search_and_report")
async def search_and_report_endpoint(query: DisasterQuery):
    """
    Search for fresh information AND use existing RAG data to generate a comprehensive report
    """
    try:
        # First, generate a report with fresh search data
        result = await generate_report_endpoint(query)
        
        # Then, supplement with any existing RAG data
        rag_query = f"{query.event_name} disaster {query.start_date} {query.end_date}"
        rag_data = await query_rag(rag_query)
        
        # If we have additional data from RAG, augment the report
        if rag_data and "results" in rag_data and len(rag_data["results"]) > 0:
            print(f"📚 Found {len(rag_data['results'])} additional facts in RAG")
            
            # Use Gemini to augment the report with existing RAG data
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            
            prompt = f"""
            I have a disaster report about {query.event_name} that needs to be augmented with additional facts.
            
            ORIGINAL REPORT:
            {result.report}
            
            ADDITIONAL FACTS FROM DATABASE:
            {json.dumps([r["content"] for r in rag_data["results"]])}
            
            Please integrate these additional facts into the report where relevant, maintaining the same structure.
            Mark new information added to the report as [ADDED FROM DATABASE: information].
            
            Return the complete augmented report.
            """
            
            response = client.models.generate_content(prompt)
            augmented_report = response.text
            
            # Update the result
            result.report = augmented_report
            # Add the additional sources
            for item in rag_data["results"]:
                if "url" in item["metadata"] and item["metadata"]["url"] not in result.sources:
                    result.sources.append(item["metadata"]["url"])
        
        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error in search and report: {str(e)}")

if __name__ == "__main__":
    print("Starting Disaster API with RAG Integration...")
    print("🌐 Endpoints:")
    print("  - POST /generate_report - Generate a disaster report")
    print("  - POST /add_url - Add a URL to RAG")
    print("  - GET /query_facts - Query RAG for facts")
    print("  - GET /health - Check API health")
    uvicorn.run("disaster_api:app", host="0.0.0.0", port=8080, reload=True)