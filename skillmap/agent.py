"""
Agent builder and orchestrator using LangChain and LangGraph with SQLite persistence.
"""

from typing import Any, Dict, List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

from skillmap.config import Config, load_config
from skillmap.prompts import DEFAULT_SYSTEM_PROMPT
from skillmap.tools import get_tools
from skillmap.memory import get_sqlite_saver


def init_gemini_model(config: Config) -> BaseChatModel:
    """Initialize the Google Gemini chat model."""
    if not config.has_google_key:
        raise ValueError(
            "Missing GOOGLE_API_KEY. Please set GOOGLE_API_KEY (or GEMINI_API_KEY) in .env "
            "or pass a valid key to Config."
        )

    return ChatGoogleGenerativeAI(
        model=config.model_name,
        google_api_key=config.google_api_key,
        temperature=0.2,
    )


def build_agent(
    config: Optional[Config] = None,
    checkpointer: Optional[SqliteSaver] = None,
    model: Optional[BaseChatModel] = None,
    system_prompt: Optional[str] = None,
    debug: bool = False,
) -> CompiledStateGraph:
    """
    Construct and compile the SkillMap agent graph with persistent memory.
    """
    if config is None:
        config = load_config()

    if checkpointer is None:
        checkpointer = get_sqlite_saver(config.sqlite_db_path)

    if model is None:
        model = init_gemini_model(config)

    tools = get_tools(config)
    prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=prompt,
        checkpointer=checkpointer,
        debug=debug,
    )
    return agent


def extract_response_text(message_content: Any) -> str:
    """
    Extract readable text from varied message content structures
    (string, list of dict blocks, or text parts).
    """
    if isinstance(message_content, str):
        return message_content
    elif isinstance(message_content, list):
        extracted = []
        for part in message_content:
            if isinstance(part, dict) and "text" in part:
                extracted.append(part["text"])
            elif isinstance(part, str):
                extracted.append(part)
            elif hasattr(part, "text"):
                extracted.append(getattr(part, "text"))
            else:
                extracted.append(str(part))
        return "\n".join(extracted)
    return str(message_content)


def invoke_agent(
    agent: CompiledStateGraph,
    prompt: str,
    thread_id: str,
) -> str:
    """
    Execute a conversational turn on a specific thread_id and return the agent response.
    """
    config = {"configurable": {"thread_id": thread_id}}
    response = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config=config,
    )
    last_message = response["messages"][-1]
    return extract_response_text(last_message.content)


def get_conversation_history(
    agent: CompiledStateGraph,
    thread_id: str,
) -> List[BaseMessage]:
    """Retrieve message history for a given thread_id."""
    state = agent.get_state({"configurable": {"thread_id": thread_id}})
    if not state or not state.values or "messages" not in state.values:
        return []
    return state.values["messages"]
