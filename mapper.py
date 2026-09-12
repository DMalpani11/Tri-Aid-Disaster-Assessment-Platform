import overpy
import osmnx as ox 
import networkx as nx
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd 
import rasterio 
import rasterstats as rs 
from typing import List, Dict, Tuple, TypedDict
from langgraph.tools import Tool
from google import genai 
import os

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
api = overpy.Overpass()

@Tool
def get_structures (area, structure):
    """
    Get the structures from the area using Overpass API
    """
    # Define the query
    query = f"""
    [out:json];
    (
      node["building"="{structure}"]({area});
      way["building"="{structure}"]({area});
      relation["building"="{structure}"]({area});
    );
    out body;
    >;
    out skel qt;
    """
    
    # Execute the query
    result = api.query(query)
    # Extract the nodes and ways
    nodes = []
    ways = []
    for node in result.nodes:
        nodes.append({
            "id": node.id,
            "lat": node.lat,
            "lon": node.lon,
            "tags": node.tags
        })
    for way in result.ways:
        ways.append({
            "id": way.id,
            "nodes": way.nodes,
            "tags": way.tags
        })
    return nodes, ways

@Tool
def analyze_disaster_situation(area: str, structures: List[Dict], population_data: Dict):
    """
    Analyze disaster situation using Gemini model to provide insights and recommendations.
    
    Args:
        area: String representing the geographical area coordinates
        structures: List of structures in the area
        population_data: Dictionary with population density information
        
    Returns:
        Dict: Analysis results including risk assessment and recommendations
    """
    # Prepare the prompt with the available data
    prompt = f"""
    Analyze the following disaster area data and provide insights:
    
    Area coordinates: {area}
    
    Available structures: {len(structures)} buildings found
    Structure types: {set(s.get('tags', {}).get('building', 'unknown') for s in structures if isinstance(s, dict) and 'tags' in s)}
    
    Population data: {population_data}
    
    Provide:
    1. Initial risk assessment
    2. Key infrastructure to focus on
    3. Recommended response priorities
    4. Potential hazards to be aware of
    """
    
    # Generate content using Gemini model
    
    response = client.models.generate_content(
        contents = prompt, 
        model = "gemini-2.0-flash"
    )
    
    # Structure and return the analysis
    analysis = {
        "risk_assessment": response.text.split("Initial risk assessment:")[1].split("Key infrastructure")[0].strip() if "Initial risk assessment:" in response.text else "Not available",
        "key_infrastructure": response.text.split("Key infrastructure:")[1].split("Recommended response")[0].strip() if "Key infrastructure:" in response.text else "Not available",
        "response_priorities": response.text.split("Recommended response priorities:")[1].split("Potential hazards")[0].strip() if "Recommended response priorities:" in response.text else "Not available",
        "potential_hazards": response.text.split("Potential hazards:")[1].strip() if "Potential hazards:" in response.text else "Not available",
        "full_response": response.text
    }
    
    return analysis

@Tool
def analyze_satellite_imagery(image_path: str):
    """
    Analyze satellite imagery using Gemini multimodal capabilities to identify damage and structures.
    
    Args:
        image_path: Path to the satellite image file
        
    Returns:
        Dict: Analysis of the satellite imagery
    """
    # Load the image
    with open(image_path, "rb") as f:
        image_data = f.read()
    
    model = client.models.get("gemini-1.5-pro-vision")
    
    prompt = """
    Please analyze this satellite imagery of a potential disaster area.
    Identify:
    1. Visible damage to infrastructure
    2. Key structures (hospitals, schools, government buildings)
    3. Road accessibility issues
    4. Flooded or damaged areas
    5. Safe zones for potential evacuation centers
    
    Provide a structured analysis that emergency responders could use.
    """
    
    response = model.generate_content([prompt, image_data])
    
    # Return the structured analysis
    return {
        "analysis": response.text,
        "image_path": image_path
    }

@Tool
def assess_population_risk(area: str, structures: List[Dict], population_data: Dict, disaster_type: str):
    """
    Assess population risk based on structures, population data and disaster type.
    
    Args:
        area: String representing the geographical area coordinates
        structures: List of structures in the area
        population_data: Dictionary with population density information
        disaster_type: Type of disaster (earthquake, flood, wildfire, etc.)
        
    Returns:
        Dict: Population risk assessment
    """
    # Convert structures and population data to a summary format
    structure_summary = {}
    for s in structures:
        if isinstance(s, dict) and 'tags' in s:
            building_type = s['tags'].get('building', 'unknown')
            structure_summary[building_type] = structure_summary.get(building_type, 0) + 1
    
    # Prepare the prompt with the available data
    prompt = f"""
    Assess population risk for a {disaster_type} in the following area:
    
    Area coordinates: {area}
    
    Structure summary: {structure_summary}
    
    Population data: {population_data}
    
    Provide:
    1. Overall risk level (Low, Medium, High, Critical)
    2. Vulnerable population groups
    3. Evacuation difficulty assessment
    4. Resource requirements for emergency response
    5. Timeline recommendations for evacuation if needed
    """
    
    # Generate content using Gemini model
    response = client.models.generate_content(
        contents = prompt, 
        model = "gemini-2.0-flash"
    )
    
    # Parse the response to extract structured data
    # This is a simplified parsing; you might need more robust extraction
    risk_assessment = {}
    if "Overall risk level:" in response.text:
        risk_assessment["overall_risk"] = response.text.split("Overall risk level:")[1].split("\n")[0].strip()
    if "Vulnerable population groups:" in response.text:
        risk_assessment["vulnerable_groups"] = response.text.split("Vulnerable population groups:")[1].split("\n")[0].strip()
    if "Evacuation difficulty assessment:" in response.text:    
        risk_assessment["evacuation_difficulty"] = response.text.split("Evacuation difficulty assessment:")[1].split("\n")[0].strip()
    if "Resource requirements:" in response.text:
        risk_assessment["resource_requirements"] = response.text.split("Resource requirements:")[1].split("\n")[0].strip()
    if "Timeline recommendations:" in response.text:
        risk_assessment["timeline_recommendations"] = response.text.split("Timeline recommendations:")[1].split("\n")[0].strip()
    if "Evacuation difficulty assessment:" in response.text:
        risk_assessment["evacuation_difficulty"] = response.text.split("Evacuation difficulty assessment:")[1].split("\n")[0].strip()
    
    # Extract risk level
    if "Overall risk level:" in response.text:
        risk_text = response.text.split("Overall risk level:")[1].split("\n")[0].strip()
        for level in ["Low", "Medium", "High", "Critical"]:
            if level.lower() in risk_text.lower():
                risk_assessment["overall_risk"] = level
                break
    
    return risk_assessment
