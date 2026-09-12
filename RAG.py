import os
import datetime
import httpx
import json
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
import chromadb

# Initialize FastAPI
app = FastAPI(title="RAG Service with Multiple Collections")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(
    path="./chroma_db"  # Use path instead of persist_directory
)

# Create collections for different data types
news_collection = chroma_client.get_or_create_collection("news")
social_collection = chroma_client.get_or_create_collection("socials")
user_collection = chroma_client.get_or_create_collection("user_inputs")

# Define models for each data type
class NewsItem(BaseModel):
    title: str
    content: str
    source: Optional[str] = None
    url: Optional[str] = None
    
class SocialPost(BaseModel):
    content: str
    platform: str
    username: Optional[str] = None
    tags: Optional[List[str]] = None
    
class UserInput(BaseModel):
    content: str
    user_id: Optional[str] = None
    context: Optional[str] = None
    
class QueryRequest(BaseModel):
    query: str
    collections: Optional[List[str]] = ["news", "socials", "user_inputs"]
    limit: int = 5

# Function to get embeddings from Ollama
async def get_embeddings(text: str) -> List[float]:
    """Get embeddings from Ollama's nomic-embed-text model"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:11434/api/embeddings",
                json={
                    "model": "nomic-embed-text",
                    "prompt": text
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail="Failed to generate embeddings from Ollama")
                
            return response.json()["embedding"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

# Helper function to add document to a collection
async def add_to_collection(collection, content: str, metadata: Dict[str, Any]):
    """Add document to a ChromaDB collection with timestamp"""
    try:
        # Add timestamp to metadata
        metadata["timestamp"] = datetime.now().isoformat()
        
        # Generate embedding
        embedding = await get_embeddings(content)
        
        # Generate a unique ID based on content and timestamp
        doc_id = f"{hash(content)}-{metadata['timestamp']}"
        
        # Add to collection
        collection.add(
            documents=[content],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[doc_id]
        )
        
        return {"id": doc_id, "status": "added"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding document to collection: {str(e)}")

# Endpoints for adding content to collections
@app.post("/add/news")
async def add_news(item: NewsItem):
    metadata = {
        "title": item.title,
        "source": item.source,
        "url": item.url,
        "type": "news"
    }
    return await add_to_collection(news_collection, item.content, metadata)

@app.post("/add/social")
async def add_social(post: SocialPost):
    metadata = {
        "platform": post.platform,
        "username": post.username,
        "tags": json.dumps(post.tags) if post.tags else "",
        "type": "social"
    }
    return await add_to_collection(social_collection, post.content, metadata)

@app.post("/add/user")
async def add_user_input(input_data: UserInput):
    metadata = {
        "user_id": input_data.user_id,
        "context": input_data.context,
        "type": "user_input"
    }
    return await add_to_collection(user_collection, input_data.content, metadata)

# Query endpoint to search across collections
@app.post("/query")
async def query_rag(query_req: QueryRequest):
    try:
        # Generate embedding for the query
        query_embedding = await get_embeddings(query_req.query)
        
        results = []
        
        # Query each requested collection
        for coll_name in query_req.collections:
            if coll_name == "news":
                collection = news_collection
            elif coll_name == "socials":
                collection = social_collection
            elif coll_name == "user_inputs":
                collection = user_collection
            else:
                continue
                
            # Query the collection
            response = collection.query(
                query_embeddings=[query_embedding],
                n_results=query_req.limit
            )
            
            # Format results
            for i, (doc, metadata, distance) in enumerate(zip(
                response["documents"][0], 
                response["metadatas"][0],
                response["distances"][0]
            )):
                results.append({
                    "content": doc,
                    "metadata": metadata,
                    "relevance_score": 1 - distance,  # Convert distance to similarity score
                    "collection": coll_name
                })
        
        # Sort by relevance score (highest first)
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Return only the top K results
        return {"results": results[:query_req.limit]}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying collections: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "collections": {
            "news": news_collection.count(),
            "socials": social_collection.count(),
            "user_inputs": user_collection.count()
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)