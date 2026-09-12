# Tri-Aid: Autonomous Multi-Agent Disaster Intelligence & Rapid Assessment Platform

An autonomous, multi-agent crisis response and intelligence system developed for the **MIT Global AI Hackathon (MIT GAH)**.

Tri-Aid ingests, cross-verifies, and synthesizes real-time multimodal data across **three crisis intelligence streams**:
1. **News & Web Intelligence:** Automated discovery, classification (live vs. static blogs), and web scraping of breaking disaster coverage.
2. **Social Media & Eyewitness Feeds:** Ingestion of eyewitness tweets, sentiment analysis, and LLM-driven misinformation detection.
3. **Geospatial & Satellite Vision:** OpenStreetMap infrastructure risk assessment and multimodal satellite imagery inspection for damage detection and evacuation corridors.

---

## 🏛️ Architecture Overview

```mermaid
flowchart TD
    User([User / Emergency Coordinator]) -->|Disaster Query| Gateway[FastAPI Layer]

    subgraph News_Stream[Stream 1: News Intelligence - bravo.py & bravo-backend.py]
        Gateway --> Discovery[Discovery Agent: Google Search]
        Discovery --> Classifier[Classifier: Static vs Live Updates]
        Classifier --> Scraper[Scraper: newspaper3k / BS4]
        Scraper --> NewsSummarizer[Gemini 2.0 Flash Summarizer]
    end

    subgraph Social_Stream[Stream 2: Social Media Intelligence - social_media_agent.py]
        Gateway --> ApifyTwitter[Apify Twitter/X Scraper]
        ApifyTwitter --> SocialParser[Eyewitness & Sentiment Extractor]
    end

    subgraph Central_Memory[Central RAG Service - RAG.py]
        NewsSummarizer -->|Embed & Store| ChromaNews[(ChromaDB: news)]
        SocialParser -->|Embed & Store| ChromaSocial[(ChromaDB: socials)]
        Gateway -->|Semantic Query| ChromaNews
        Gateway -->|Semantic Query| ChromaSocial
    end

    subgraph Geo_Stream[Stream 3: Geospatial & Vision - mapper.py]
        Gateway -.-> Overpass[Overpass API: OSM Infrastructure]
        Gateway -.-> SatVision[Gemini 1.5 Pro Vision: Satellite Imagery]
    end

    subgraph Synthesis_Engine[Report Synthesis]
        ChromaNews --> MasterSynthesizer[Gemini 2.0 Flash Report Generator]
        ChromaSocial --> MasterSynthesizer
        MasterSynthesizer --> FinalReport[Cited Markdown Disaster Report]
    end

    FinalReport --> User
```

---

## 🧩 Core Modules

| Module | Description | 
| :--- | :--- | :--- |
| **`RAG.py`** | Central vector database service managing ChromaDB collections (`news`, `socials`, `user_inputs`) with Ollama `nomic-embed-text` embeddings. | 
| **`bravo.py`** | Multi-agent news workflow built with Google ADK (`SequentialAgent`, `Runner`) orchestrating Discovery, Page Classification, and Scraping. |  
| **`bravo-backend.py`** | Search-First, RAG-Second hybrid API that dynamically searches, summarizes, stores into ChromaDB, and compiles structured situation reports. | 
| **`social_media_agent.py`** | Apify-powered Twitter/X scraper extracting eyewitness reports, public sentiment, and urgent relief needs, with combined report synthesis. | 
| **`social_agent_api.py`** | FastAPI endpoints for social data ingestion and report generation. | 
| **`disaster_api.py`** | Standalone disaster reporting API service with synchronous RAG fact augmentation. | 
| **`mapper.py`** | Geospatial analytics module using OpenStreetMap Overpass API, OSMnx, and Gemini Multimodal Vision for damage and evacuation corridor analysis. |  

---


```

