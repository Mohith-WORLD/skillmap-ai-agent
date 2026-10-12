"""
Automated Memory Verification Test for SkillMap AI Agent.

Verifies Task 6 requirements:
1. Ask the agent to find jobs.
2. Ask a follow-up question about the second job.
3. Simulate application restart (close DB, delete agent in-memory instance).
4. Verify conversation retrieval from SQLite using original thread_id.
"""

import os
import sys
import tempfile
import sqlite3
from typing import List

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage


class MockToolChatModel(FakeListChatModel):
    """Deterministic test chat model that supports tool binding for LangGraph agents."""
    def bind_tools(self, tools, **kwargs):
        return self
from skillmap.config import Config, load_config
from skillmap.memory import get_sqlite_saver, list_saved_threads
from skillmap.agent import (
    build_agent,
    invoke_agent,
    get_conversation_history,
    extract_response_text,
)


def run_memory_verification(live: bool = False):
    print("=" * 70)
    print(" SKILLMAP AI AGENT: CONVERSATIONAL MEMORY & CHECKPOINT TEST")
    print("=" * 70)

    config = load_config()
    test_thread_id = "portfolio_verification_thread_001"

    # Use a dedicated test database
    temp_dir = tempfile.mkdtemp()
    test_db_path = os.path.join(temp_dir, "test_checkpoints.db")
    print(f"\n[Environment] Using isolated SQLite database at: {test_db_path}")

    # Determine execution mode (live API or deterministic test model)
    use_live = live and config.has_google_key
    if use_live:
        print("[Mode] LIVE API MODE: Testing with Google Gemini and live tools.")
    else:
        print("[Mode] SIMULATION MODE: Testing LangGraph state graph and SQLite checkpointing mechanics.")
        if live and not config.has_google_key:
            print("  (Note: Live mode requested but GOOGLE_API_KEY was not found. Using deterministic test model.)")

    # Responses designed to mirror notebook turns
    mock_responses = [
        # Turn 1 response: Skill demand + Job listings
        (
            "Generative AI demand in 2025 is growing exponentially across India, "
            "especially for LLM fine-tuning, RAG pipelines, and agent orchestration.\n\n"
            "Here are current opportunities in India:\n"
            "1. AI Engineer Intern - Bengaluru, India (Infosys) - Apply: https://example.com/job1\n"
            "2. Generative AI Solutions Developer - Hyderabad, India (TechVision Labs) - Apply: https://example.com/job2\n"
            "3. Machine Learning Associate - Pune, India (Innovate AI) - Apply: https://example.com/job3"
        ),
        # Turn 2 response: Follow-up on Job 2
        (
            "Regarding the second job: Generative AI Solutions Developer at TechVision Labs in Hyderabad, India.\n"
            "This position focuses on developing enterprise LLM applications, integrating vector databases, "
            "and building autonomous agents. Experience with Python, LangChain, and prompt engineering is prioritized. "
            "You can apply here: https://example.com/job2"
        ),
        # Turn 3 response: Post-restart continuation
        (
            "Yes, following up on your application for TechVision Labs in Hyderabad: "
            "Preparing a portfolio showcasing LangGraph agents with persistent memory will make you stand out!"
        ),
    ]

    # --------------------------------------------------------------------------
    # PHASE 1: First Application Session
    # --------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("PHASE 1: APPLICATION SESSION 1 - Initial Queries")
    print("-" * 70)

    # Initialize checkpointer & agent for Session 1
    saver_session1 = get_sqlite_saver(test_db_path)
    model_session1 = None if use_live else MockToolChatModel(responses=mock_responses[:2])

    agent_session1 = build_agent(
        config=Config(sqlite_db_path=test_db_path, google_api_key=config.google_api_key),
        checkpointer=saver_session1,
        model=model_session1,
    )

    # Turn 1: Find jobs
    query_1 = "What's the demand for generative AI in the industry and show me related job openings in India"
    print(f"\n[Turn 1] User Prompt: '{query_1}'")
    resp_1 = invoke_agent(agent_session1, query_1, test_thread_id)
    print(f"\n[Turn 1] Agent Response:\n{resp_1}")

    # Turn 2: Follow-up question about the second job
    query_2 = "Tell me more about the second job you showed"
    print(f"\n[Turn 2] User Prompt: '{query_2}'")
    resp_2 = invoke_agent(agent_session1, query_2, test_thread_id)
    print(f"\n[Turn 2] Agent Response:\n{resp_2}")

    # Inspect Session 1 message count
    history_s1 = get_conversation_history(agent_session1, test_thread_id)
    print(f"\n[Session 1 Status] Messages recorded in memory: {len(history_s1)}")
    assert len(history_s1) >= 4, f"Expected at least 4 messages (2 human, 2 AI), got {len(history_s1)}"

    # --------------------------------------------------------------------------
    # PHASE 2: Application Restart Simulation
    # --------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("PHASE 2: SIMULATING APPLICATION RESTART & PROCESS TERMINATION")
    print("-" * 70)
    print("Destroying Session 1 in-memory agent instances and closing DB connection...")

    # Close connection and destroy references
    saver_session1.conn.close()
    del agent_session1
    del saver_session1

    print("Session 1 completely terminated. Simulating new process launch...")

    # --------------------------------------------------------------------------
    # PHASE 3: Application Session 2 (Post-Restart)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("PHASE 3: APPLICATION SESSION 2 - Reconnecting to SQLite Database")
    print("-" * 70)

    # Open completely new connection to the existing SQLite DB file
    saver_session2 = get_sqlite_saver(test_db_path)
    model_session2 = None if use_live else MockToolChatModel(responses=[mock_responses[2]])

    agent_session2 = build_agent(
        config=Config(sqlite_db_path=test_db_path, google_api_key=config.google_api_key),
        checkpointer=saver_session2,
        model=model_session2,
    )

    # Verify conversation retrieval using original thread_id
    saved_threads = list_saved_threads(test_db_path)
    print(f"[Verification] Threads discovered in SQLite file: {saved_threads}")
    assert test_thread_id in saved_threads, f"Thread ID '{test_thread_id}' was not found in SQLite checkpoints!"

    history_s2 = get_conversation_history(agent_session2, test_thread_id)
    print(f"[Verification] Messages recovered for '{test_thread_id}': {len(history_s2)}")
    assert len(history_s2) == len(history_s1), (
        f"Message count mismatch after restart! Expected {len(history_s1)}, got {len(history_s2)}"
    )

    # Verify message content integrity
    first_user_msg = extract_response_text(history_s2[0].content)
    first_ai_msg = extract_response_text(history_s2[1].content)
    second_user_msg = extract_response_text(history_s2[2].content)
    second_ai_msg = extract_response_text(history_s2[3].content)

    print("\n[Integrity Check]")
    print(f"  Turn 1 User Query Recovered : {first_user_msg[:60]}...")
    print(f"  Turn 1 AI Response Recovered: {first_ai_msg[:60]}...")
    print(f"  Turn 2 User Query Recovered : {second_user_msg[:60]}...")
    print(f"  Turn 2 AI Response Recovered: {second_ai_msg[:60]}...")

    assert query_1 in first_user_msg
    assert query_2 in second_user_msg

    # Turn 3: Seamless continuation on existing thread
    query_3 = "What should I prepare for the interview for that role?"
    print(f"\n[Turn 3 - Post Restart] User Prompt: '{query_3}'")
    resp_3 = invoke_agent(agent_session2, query_3, test_thread_id)
    print(f"\n[Turn 3 - Post Restart] Agent Response:\n{resp_3}")

    final_history = get_conversation_history(agent_session2, test_thread_id)
    print(f"\n[Final Status] Total messages in thread '{test_thread_id}': {len(final_history)}")
    assert len(final_history) >= 6, "Expected 6 messages after Turn 3"

    saver_session2.conn.close()

    print("\n" + "=" * 70)
    print(" TEST RESULT: ALL CONVERSATIONAL MEMORY CHECKS PASSED SUCCESSFULLY!")
    print(" Thread persistence survived application restart with 100% fidelity.")
    print("=" * 70 + "\n")

    # Cleanup temp db
    try:
        os.remove(test_db_path)
        os.rmdir(temp_dir)
    except Exception:
        pass


if __name__ == "__main__":
    is_live = "--live" in sys.argv
    run_memory_verification(live=is_live)
