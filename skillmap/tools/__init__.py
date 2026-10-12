"""
Tools module initialization.
"""

from typing import List
from langchain_core.tools import BaseTool
from skillmap.config import Config
from skillmap.tools.skill_demand import create_skill_demand_tool
from skillmap.tools.job_search import create_job_search_tool


def get_tools(config: Config) -> List[BaseTool]:
    """Instantiate and return the suite of agent tools given configuration."""
    skill_tool = create_skill_demand_tool(config.tavily_api_key)
    job_tool = create_job_search_tool(config.rapidapi_key)
    return [skill_tool, job_tool]


__all__ = ["get_tools", "create_skill_demand_tool", "create_job_search_tool"]
