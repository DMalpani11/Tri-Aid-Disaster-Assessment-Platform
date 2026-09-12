from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import uvicorn

# Import functions from social_media_agent.py
from social_media_agent import (
    fetch_and_store_social_media,
    generate_social_report,
    generate_complete_report,
    DisasterQuery,
    ReportResponse
)

# Initialize FastAPI
app = FastAPI(title="Social Media Disaster Reporting API")

@app.post("/fetch_social_data")
async def fetch_social_data(query: DisasterQuery):
    """
    Fetch social media data for a disaster and store in RAG
    """
    try:
        result = fetch_and_store_social_media(query.dict())
        return {
            "status": "success",
            "stored_count": result["stored_count"],
            "total_posts_found": result["posts"],
            "event": query.event_name
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/social_report")
async def social_report(query: DisasterQuery):
    """
    Generate a report based on social media posts only
    """
    try:
        report = generate_social_report(query.dict())
        return ReportResponse(
            report=report["report"],
            sources=report["sources"],
            event_name=query.event_name,
            start_date=query.start_date,
            end_date=query.end_date
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/combined_report")
async def combined_report(query: DisasterQuery):
    """
    Generate a comprehensive report combining news and social media
    """
    try:
        report = generate_complete_report(query.dict())
        return ReportResponse(
            report=report["report"],
            sources=report["sources"],
            event_name=query.event_name,
            start_date=query.start_date,
            end_date=query.end_date
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Check API health"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "social_agent": "running",
            "apify_integration": "configured"
        }
    }

if __name__ == "__main__":
    print("🚀 Starting Social Media Disaster Reporting API...")
    print("Visit http://localhost:8081/docs for API documentation")
    uvicorn.run("social_agent_api:app", host="0.0.0.0", port=8081, reload=True)