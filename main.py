"""
SkillMap AI Agent CLI Entry Point.

Provides an interactive terminal interface, persistent thread management,
and configuration diagnostics.
"""

import argparse
import sys
import uuid
from skillmap.config import load_config
from skillmap.agent import build_agent, invoke_agent, get_conversation_history, extract_response_text
from skillmap.memory import get_sqlite_saver, list_saved_threads, get_thread_messages


def print_banner():
    banner = r"""
======================================================================
   ____  _     _ _ _ __  __              _     ___ 
  / ___|| |__ (_) | |  \/  | __ _ _ __  / \   |_ _|
  \___ \| |/ /| | | | |\/| |/ _` | '_ \ / _ \   | | 
   ___) |   < | | | | |  | | (_| | |_) / ___ \  | | 
  |____/|_|\_\|_|_|_|_|  |_|\__,_| .__/_/   \_\___|
                                 |_|               
   Skill-to-Career Mapping AI Agent with LangGraph SQLite Memory
======================================================================
"""
    print(banner)


def show_status(config):
    report = config.get_status_report()
    print("\n--- Environment Configuration Status ---")
    for key, value in report.items():
        print(f"  {key:<18}: {value}")
    print("----------------------------------------\n")
    if not report["is_ready"]:
        missing = config.missing_keys()
        print(f"Notice: Missing credentials for: {', '.join(missing)}")
        print("Copy .env.example to .env and insert your API keys to enable live LLM and search tools.\n")


def display_history(saver, thread_id: str):
    history = get_thread_messages(saver, thread_id)
    if not history:
        print(f"\n[No message history found for thread: {thread_id}]\n")
        return

    print(f"\n=== Conversation History for Thread: {thread_id} ({len(history)} messages) ===")
    for idx, msg in enumerate(history, 1):
        role = getattr(msg, "type", "message").upper()
        content = extract_response_text(getattr(msg, "content", msg))
        print(f"\n[{idx}] {role}:")
        print(content)
    print("=========================================================================\n")


def run_interactive(agent, saver, config, thread_id: str):
    print(f"Active Thread ID: '{thread_id}'")
    existing_history = get_thread_messages(saver, thread_id)
    if existing_history:
        print(f"[Thread Resumed] Recovered {len(existing_history)} existing message(s) from SQLite. Type 'history' to review.")
    print("Type your message below. Commands:")
    print("  'exit' or 'quit' : End current session")
    print("  'history'        : View full conversation history for this thread")
    print("  'threads'        : List all saved thread IDs in SQLite")
    print("  'new'            : Start a new thread session")
    print("-" * 70)

    current_thread = thread_id

    while True:
        try:
            user_input = input(f"\n[{current_thread}] You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting session. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Session terminated. Conversation checkpoints remain safely stored in SQLite.")
            break

        if user_input.lower() == "history":
            display_history(saver, current_thread)
            continue

        if user_input.lower() == "threads":
            saved = list_saved_threads(config.sqlite_db_path)
            print(f"\nSaved Threads in SQLite ({len(saved)}): {saved}\n")
            continue

        if user_input.lower() == "new":
            current_thread = f"session_{uuid.uuid4().hex[:6]}"
            print(f"\nSwitched to new Thread ID: '{current_thread}'")
            continue

        print("\nAgent is thinking...")
        try:
            response = invoke_agent(agent, user_input, current_thread)
            print(f"\n[SkillMap Agent]:\n{response}\n")
        except Exception as e:
            print(f"\n[Error executing query]: {e}")
            if not config.has_google_key:
                print("Hint: Please check your GOOGLE_API_KEY in .env file.")


def main():
    parser = argparse.ArgumentParser(description="SkillMap AI Agent with SQLite Conversational Memory")
    parser.add_argument("--thread-id", type=str, default="user_default_thread", help="Conversation thread ID")
    parser.add_argument("--query", type=str, help="Single query to execute on the specified thread")
    parser.add_argument("--history", action="store_true", help="Print conversation history for thread and exit")
    parser.add_argument("--status", action="store_true", help="Display environment configuration status and exit")
    parser.add_argument("--list-threads", action="store_true", help="List all thread IDs saved in SQLite")

    args = parser.parse_args()

    print_banner()
    config = load_config()

    if args.status:
        show_status(config)
        return

    # Checkpoint saver can be initialized without requiring external LLM API keys
    try:
        saver = get_sqlite_saver(config.sqlite_db_path)
    except Exception as e:
        print(f"Failed to initialize SQLite checkpointer: {e}")
        sys.exit(1)

    if args.list_threads:
        threads = list_saved_threads(config.sqlite_db_path)
        print(f"Saved Thread IDs in '{config.sqlite_db_path}':")
        for t in threads:
            print(f"  - {t}")
        return

    if args.history:
        display_history(saver, args.thread_id)
        return

    # Check credentials if executing query or running interactive
    if not config.has_google_key:
        print("[Warning] GOOGLE_API_KEY is not configured in your environment or .env file.")
        print("To run live conversations with Google Gemini, populate your .env file.")
        print("To run the automated verification test suite without live API keys, execute:")
        print("    python test_memory.py\n")
        sys.exit(1)

    try:
        agent = build_agent(config=config, checkpointer=saver)
    except Exception as e:
        print(f"Failed to initialize agent: {e}")
        sys.exit(1)

    if args.query:
        print(f"Querying thread '{args.thread_id}': {args.query}\n")
        resp = invoke_agent(agent, args.query, args.thread_id)
        print(f"[SkillMap Agent]:\n{resp}")
        return

    # Interactive loop
    run_interactive(agent, saver, config, args.thread_id)


if __name__ == "__main__":
    main()
