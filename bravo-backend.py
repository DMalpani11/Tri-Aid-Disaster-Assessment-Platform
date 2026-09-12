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

# Import working functions from bravo.py
from bravo import (
    process_url,
    html_scraper,
    call_agent,
    manual_pipeline,
    fetch_content
)
from google.adk.tools import google_search

# Google services
from google.genai import Client as genai_client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="Disaster Reporter API with RAG Integration")

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
    """Store summary in RAG database"""
    try:
        news_item = NewsItem(
            title=title,
            content=content,
            source="disaster_reporter",
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
    """Query RAG for stored summaries"""
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
# API Endpoints
#-------------------------------------------

@app.post("/search")
async def search_endpoint(query: DisasterQuery):
    """Search for disaster-related articles"""
    try:
        search_query = f"{query.event_name} disaster {query.start_date} {query.end_date} news reports"
        print(f"🔍 Searching for: {search_query}")
        
        # Use call_agent from bravo.py instead of google_search.run
        results = call_agent(search_query)
        
        # Parse the results from the agent response
        try:
            # Extract JSON from the response if it's embedded in markdown
            if "```json" in results:
                json_str = results.split("```json")[1].split("```")[0].strip()
                data = json.loads(json_str)
            else:
                data = json.loads(results)
                
            formatted_results = []
            if "sources" in data:
                for source in data["sources"]:
                    if "url" in source:
                        formatted_results.append({
                            "url": source["url"],
                            "title": source.get("title", source["url"]),
                            "description": source.get("description", "")
                        })
            
            print(f"✅ Found {len(formatted_results)} results")
            return {"results": formatted_results}
            
        except json.JSONDecodeError:
            print("⚠️ Could not parse JSON from agent response")
            return {"error": "Invalid response format", "raw_response": results}
            
    except Exception as e:
        print(f"❌ Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_url")
async def process_url_endpoint(url: str = Query(...)):
    """Process a single URL, store summary in RAG, and return result"""
    try:
        # Use the process_url function from bravo.py
        result = process_url(url)
        
        if "error" in result:
            return {"status": "error", "message": result["error"]}
        
        # Extract title from the scraped content
        title = "Article summary"
        content = html_scraper(url)
        for line in content.split("\n"):
            if line.startswith("TITLE:"):
                title = line.replace("TITLE:", "").strip()
                break
        
        # Store in RAG
        success = await store_in_rag(title, result["summary"], url)
        
        return {
            "url": url,
            "title": title,
            "summary": result["summary"],
            "stored_in_rag": success
        }
    except Exception as e:
        print(f"❌ Error processing URL: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_report")
async def generate_report_endpoint(query: DisasterQuery, background_tasks: BackgroundTasks):
    """Complete workflow to generate a report"""
    try:
        # Step 1: Search for articles using call_agent
        search_query = f"{query.event_name} disaster {query.start_date} {query.end_date} news reports"
        print(f"🔍 Searching for: {search_query}")
        
        # Get results using call_agent
        search_results = call_agent(search_query)
        
        # Parse the results
        try:
            # Extract JSON if wrapped in markdown
            if "```json" in search_results:
                json_str = search_results.split("```json")[1].split("```")[0].strip()
                data = json.loads(json_str)
            else:
                data = json.loads(search_results)
                
            # Create JSON structure for processing
            temp_data = {
                "event_name": query.event_name,
                "start_date": query.start_date,
                "end_date": query.end_date,
                "sources": data.get("sources", [])
            }
            
            # Continue with the rest of your function...
            # [rest of function implementation]
        
        except json.JSONDecodeError:
            print("⚠️ Could not parse JSON from agent response")
            raise HTTPException(status_code=500, detail="Invalid response format from search agent")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compile_report_from_rag")
async def compile_report_from_rag(query: DisasterQuery):
    """
    Generate a report with a 'search first, RAG second' approach 
    to ensure relevance and freshness
    """
    try:
        # Validate dates and use defaults if empty
        start_date = query.start_date if query.start_date else "past year"
        end_date = query.end_date if query.end_date else "present"
        
        # Step 1: ALWAYS search for fresh, relevant information first
        search_query = f"{query.event_name} disaster news reports"
        if start_date != "past year" or end_date != "present":
            search_query += f" from {start_date} to {end_date}"
            
        print(f"🔍 Searching for fresh information: {search_query}")
        
        # Use call_agent from bravo.py with better error handling
        try:
            search_results = call_agent(search_query)
            print(f"✅ Search completed, processing results...")
        except Exception as e:
            print(f"❌ Search agent error: {str(e)}")
            search_results = """{"sources": []}"""  # Provide fallback empty result
        
        # Parse the search results with much more robust error handling
        search_urls = []
        search_summaries = []
        search_sources = []
        
        try:
            # Check if the response is a string and parse it accordingly
            if isinstance(search_results, str):
                # Try multiple parsing strategies
                if "```json" in search_results:
                    # Extract JSON from markdown code block
                    json_parts = search_results.split("```json")
                    if len(json_parts) > 1:
                        json_str = json_parts[1].split("```")[0].strip()
                        data = json.loads(json_str)
                    else:
                        # Try other code block formats
                        json_parts = search_results.split("```")
                        if len(json_parts) > 1:
                            json_str = json_parts[1].strip()
                            data = json.loads(json_str)
                        else:
                            # No code blocks, try the whole string
                            data = json.loads(search_results)
                elif search_results.strip().startswith("{") and search_results.strip().endswith("}"):
                    # Directly parse as JSON
                    data = json.loads(search_results)
                else:
                    # If it's not JSON at all, create a simple structure for the agent's response
                    data = {
                        "sources": [],
                        "message": search_results[:1000]  # Include part of the message for debugging
                    }
            else:
                # If not a string, try to convert to dict
                data = dict(search_results) if hasattr(search_results, '__iter__') else {"sources": []}
            
            # Process each URL found in search results
            processed_count = 0
            
            if "sources" in data and isinstance(data["sources"], list):
                for source in data["sources"]:
                    if isinstance(source, dict) and "url" in source:
                        url = source["url"]
                        search_urls.append(url)
                        try:
                            # Process URL
                            result = process_url(url)
                            
                            if "summary" in result:
                                # Get title
                                title = source.get("title", "Article summary")
                                
                                # Store summary for report generation
                                search_summaries.append({
                                    "content": result["summary"],
                                    "url": url,
                                    "title": title,
                                    "source": "search" 
                                })
                                search_sources.append(url)
                                
                                # Store in RAG for future reference
                                await store_in_rag(title, result["summary"], url)
                                processed_count += 1
                        except Exception as e:
                            print(f"❌ Error processing {url}: {str(e)}")
            
            print(f"✅ Processed {processed_count} URLs from search")
                
        except json.JSONDecodeError as e:
            print(f"⚠️ Could not parse JSON from agent response: {str(e)}")
            print(f"Raw response (first 500 chars): {str(search_results)[:500]}...")
            
            # DON'T fail here - continue with empty search results
            print("⚠️ Continuing with empty search results")
        
        # Step 2: AFTER search, query RAG for ADDITIONAL relevant information
        print(f"🔍 Querying RAG for information about: {query.event_name}")
        rag_query = f"{query.event_name} disaster"
        if start_date != "past year":
            rag_query += f" {start_date}"
        if end_date != "present":
            rag_query += f" {end_date}"
            
        rag_data = await query_rag(rag_query)
        
        # Check RAG results for relevance using similarity threshold
        rag_summaries = []
        rag_sources = []
        
        if rag_data and "results" in rag_data:
            # Include all RAG results that aren't duplicates of search results
            for item in rag_data["results"]:
                # Check for duplicate URLs
                url = item["metadata"].get("url", "")
                if url and url not in search_urls and url not in rag_sources:
                    rag_summaries.append({
                        "content": item["content"],
                        "url": url,
                        "title": item["metadata"].get("title", "Unknown title"),
                        "source": "rag"
                    })
                    rag_sources.append(url)
            
            print(f"✅ Found {len(rag_summaries)} additional items in RAG")
        
        # Step 3: Combine search and RAG results
        all_summaries = search_summaries + rag_summaries
        all_sources = search_sources + rag_sources
        
        # If we don't have enough data after combining both sources, SEARCH AGAIN with broadened query
        if not all_summaries:
            print("⚠️ No results found, broadening search...")
            broader_search_query = f"{query.event_name} disaster OR wildfire OR emergency"
            
            try:
                broader_results = call_agent(broader_search_query)
                
                # Try to parse the broader results (simplified parsing)
                try:
                    if "```json" in broader_results:
                        json_str = broader_results.split("```json")[1].split("```")[0].strip()
                        broader_data = json.loads(json_str)
                    else:
                        broader_data = json.loads(broader_results)
                    
                    # Process each URL from broader search
                    if "sources" in broader_data:
                        for source in broader_data["sources"]:
                            if "url" in source:
                                url = source["url"]
                                if url not in search_urls and url not in rag_sources:
                                    try:
                                        result = process_url(url)
                                        if "summary" in result:
                                            title = source.get("title", "Article summary")
                                            search_summaries.append({
                                                "content": result["summary"],
                                                "url": url,
                                                "title": title,
                                                "source": "broader_search" 
                                            })
                                            search_sources.append(url)
                                            await store_in_rag(title, result["summary"], url)
                                    except Exception as e:
                                        print(f"❌ Error processing broader search URL {url}: {str(e)}")
                
                except Exception as e:
                    print(f"❌ Error parsing broader search results: {str(e)}")
            
            except Exception as e:
                print(f"❌ Broader search failed: {str(e)}")
        
        # Update combined results after broader search
        all_summaries = search_summaries + rag_summaries
        all_sources = search_sources + rag_sources
        
        # If we still don't have anything, return a clear error
        if not all_summaries:
            return ReportResponse(
                report=f"# No Information Found\n\nAfter extensive searching, no relevant information could be found about **{query.event_name}**. Please try a different search query or check spelling.",
                sources=[],
                event_name=query.event_name,
                start_date=start_date,
                end_date=end_date
            )
        
        # Step 4: Generate report using combined data
        client = genai_client(api_key=os.environ.get("GEMINI_API_KEY"))
        
        # Format the summaries for the prompt
        summaries_text = ""
        for i, summary in enumerate(all_summaries):
            source_indicator = "[SEARCH]" if summary["source"] == "search" else "[RAG]"
            summaries_text += f"\nSOURCE {i+1} {source_indicator}:\n"
            summaries_text += f"TITLE: {summary.get('title', 'Unknown')}\n"
            summaries_text += f"URL: {summary.get('url', 'No URL')}\n"
            summaries_text += f"CONTENT: {summary['content']}\n\n"
        
        prompt = f"""
        Generate a comprehensive disaster report about the {query.event_name} that occurred between 
        {start_date} and {end_date}.
        
        Use the following information from various sources:
        
        {summaries_text}
        
        Your report should include:
        
        ## Disaster Event: {query.event_name}
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
        
        IMPORTANT: IF THE SOURCES DON'T ACTUALLY CONTAIN INFORMATION ABOUT {query.event_name},
        STATE CLEARLY THAT "NO RELEVANT INFORMATION ABOUT {query.event_name} WAS FOUND" INSTEAD OF
        GENERATING A FICTIONAL REPORT.
        
        Format the report in Markdown with clear headings and structure.
        """
        
        try:
            response = client.models.generate_content(
                contents=prompt,
                model="gemini-2.0-flash"
            )
            report = response.text
        except Exception as e:
            print(f"❌ Error generating report: {str(e)}")
            report = f"# Error Generating Report\n\nThe system encountered an error while generating the report: {str(e)}\n\nPlease try again later."
        
        # Save the report
        report_filename = f"disaster_report_{query.event_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_filename, 'w') as f:
            f.write(report)
        
        return ReportResponse(
            report=report,
            sources=all_sources,
            event_name=query.event_name,
            start_date=start_date,
            end_date=end_date
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

@app.get("/health")
async def health_check():
    """Check the health of the API and RAG connection"""
    try:
        # Check RAG connection
        rag_status = "unknown"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{RAG_SERVER_URL}/health")
                if response.status_code == 200:
                    rag_status = "connected"
                    rag_data = response.json()
                else:
                    rag_status = f"error: {response.status_code}"
                    rag_data = {"error": response.text}
        except Exception as e:
            rag_status = f"disconnected: {str(e)}"
            rag_data = {"error": str(e)}
            
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "rag_connection": rag_status,
            "rag_details": rag_data,
            "environment": {
                "gemini_api": "configured" if os.getenv("GEMINI_API_KEY") else "missing"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    print("🚀 Starting Disaster Reporter API with RAG Integration...")
    print("Visit http://localhost:8080/docs for API documentation")
    uvicorn.run("bravo-backend:app", host="0.0.0.0", port=8080, reload=True)