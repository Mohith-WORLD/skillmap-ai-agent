"""
Skill Demand Search Tool using Tavily Search API.
Preserves the notebook implementation with clean error handling.
"""

from typing import Optional
from langchain_core.tools import BaseTool, tool
from langchain_tavily import TavilySearch


def create_skill_demand_tool(tavily_api_key: Optional[str] = None) -> BaseTool:
    """
    Create the skill demand research tool using Tavily Search.
    
    If the API key is absent, creates a graceful stub tool informing the agent.
    """
    if tavily_api_key and tavily_api_key.strip():
        return TavilySearch(
            name="skill_demand_tool",
            description="Search for industry demand, salary insights, and career trends for technologies and skills.",
            max_results=5,
            search_depth="advanced",
            tavily_api_key=tavily_api_key.strip(),
        )

    @tool("skill_demand_tool")
    def missing_tavily_tool(query: str) -> str:
        """Search for industry demand, salary insights, and career trends for technologies and skills."""
        return (
            "Error: TAVILY_API_KEY is not configured in the environment. "
            "Please provide a valid Tavily API key in .env to perform live industry skill searches."
        )

    return missing_tavily_tool
