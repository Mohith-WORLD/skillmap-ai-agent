"""
Job Search Tool using RapidAPI JSearch.
Preserves the notebook implementation with robust error handling and API key encapsulation.
"""

from typing import Any, Dict, List, Optional
import requests
from langchain_core.tools import BaseTool, tool


def create_job_search_tool(rapidapi_key: Optional[str] = None) -> BaseTool:
    """
    Factory function to create the search_jobs LangChain tool with configured RapidAPI credentials.
    """
    api_key = (rapidapi_key or "").strip()

    @tool("search_jobs")
    def search_jobs(skill: str, location: str) -> List[Dict[str, Any]]:
        """Search for jobs requiring a specific skill using the JSearch API."""
        print(f"\n[Tool: search_jobs] Searching jobs for: '{skill}' in '{location}'")

        if not api_key:
            print("[Tool: search_jobs] Warning: RAPIDAPI_KEY is not configured.")
            return [
                {
                    "error": (
                        "RAPIDAPI_KEY is not configured in environment. "
                        "Please provide your RapidAPI key in .env to fetch live job listings."
                    )
                }
            ]

        url = "https://jsearch.p.rapidapi.com/search"
        headers = {
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": "jsearch.p.rapidapi.com",
        }
        params = {
            "query": f"{skill} in {location}",
            "page": "1",
            "num_pages": "1",
            "country": "in",
            "employment_types": "INTERN,FULLTIME",
            "job_requirements": "no_experience,under_3_years_experience",
        }

        try:
            response = requests.get(url, headers=headers, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            jobs = data.get("data", [])
            print(f"[Tool: search_jobs] Found {len(jobs)} jobs\n")

            return [
                {
                    "title": job.get("job_title"),
                    "company": job.get("employer_name"),
                    "location": job.get("job_city"),
                    "apply_link": job.get("job_apply_link"),
                }
                for job in jobs
            ]
        except requests.exceptions.RequestException as e:
            print(f"[Tool: search_jobs] API request error: {e}")
            return [
                {
                    "error": f"Failed to retrieve job listings from JSearch: {str(e)}"
                }
            ]

    return search_jobs
