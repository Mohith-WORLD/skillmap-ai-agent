"""
Comprehensive Pytest Test Suite for SkillMap AI Agent.
"""

import os
import tempfile
import sqlite3
import pytest

from skillmap.config import Config, load_config
from skillmap.prompts import DEFAULT_SYSTEM_PROMPT
from skillmap.tools import get_tools, create_skill_demand_tool, create_job_search_tool
from skillmap.memory import get_sqlite_saver, list_saved_threads
from skillmap.agent import (
    build_agent,
    invoke_agent,
    get_conversation_history,
    extract_response_text,
)
from test_memory import MockToolChatModel


def test_config_defaults():
    config = Config()
    assert config.model_name == "gemini-2.5-flash"
    assert config.sqlite_db_path == "data/checkpoints.db"
    assert not config.has_google_key
    assert "GOOGLE_API_KEY" in config.missing_keys()
    assert "TAVILY_API_KEY" in config.missing_keys()
    assert "RAPIDAPI_KEY" in config.missing_keys()


def test_config_status_report_masks_keys():
    config = Config(
        google_api_key="mock_google_api_key_for_testing_12345",
        tavily_api_key="mock_tavily_key_for_testing_67890",
        rapidapi_key="mock_rapidapi_key_for_testing_99999",
    )
    report = config.get_status_report()
    assert report["is_ready"] is True
    # Ensure sensitive credentials are never leaked in full
    assert "mock_google_api_key_for_testing_12345" not in report["GOOGLE_API_KEY"]
    assert "..." in report["GOOGLE_API_KEY"]


def test_tools_graceful_missing_keys():
    tools = get_tools(Config())
    assert len(tools) == 2
    names = [t.name for t in tools]
    assert "skill_demand_tool" in names
    assert "search_jobs" in names

    # Test job tool with missing key returns error response rather than crashing
    job_tool = tools[1]
    res = job_tool.invoke({"skill": "Python", "location": "India"})
    assert isinstance(res, list)
    assert "error" in res[0]
    assert "RAPIDAPI_KEY is not configured" in res[0]["error"]


def test_sqlite_saver_creation_and_thread_isolation():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_isolation.db")

    saver = get_sqlite_saver(db_path)
    assert os.path.exists(db_path)

    model = MockToolChatModel(responses=["Hello Thread 1", "Hello Thread 2"])
    agent = build_agent(
        config=Config(sqlite_db_path=db_path),
        checkpointer=saver,
        model=model,
    )

    invoke_agent(agent, "Hi from User 1", "thread_alpha")
    invoke_agent(agent, "Hi from User 2", "thread_beta")

    saved_threads = list_saved_threads(db_path)
    assert "thread_alpha" in saved_threads
    assert "thread_beta" in saved_threads

    history_alpha = get_conversation_history(agent, "thread_alpha")
    history_beta = get_conversation_history(agent, "thread_beta")

    assert len(history_alpha) == 2
    assert len(history_beta) == 2
    assert "Hi from User 1" in extract_response_text(history_alpha[0].content)
    assert "Hi from User 2" in extract_response_text(history_beta[0].content)

    saver.conn.close()
    try:
        os.remove(db_path)
        os.rmdir(temp_dir)
    except Exception:
        pass


def test_extract_response_text_formats():
    assert extract_response_text("Simple string") == "Simple string"
    assert extract_response_text([{"text": "Block 1"}, {"text": "Block 2"}]) == "Block 1\nBlock 2"
    assert extract_response_text(["Part A", "Part B"]) == "Part A\nPart B"
