"""
LLM Service for Voice Agent - Streaming LLM with LangGraph

Based on Z-agent-llm app/services/sandwich/llm_service.py

Features:
- Streaming LLM responses (Gemini)
- LangGraph with tool calling
- Interruption support for barge-in
- Usage metadata tracking
- Tool calling with filler phrases
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
from dataclasses import dataclass
from functools import lru_cache

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("[LLM] ⚠️ langchain-google-genai not available")

# Import voice agent components
try:
    from voice_agent_graph import create_voice_agent_graph, VoiceAgentState
except ImportError:
    print("[LLM] ⚠️ voice_agent_graph not available - graph creation will fail")

try:
    from voice_agent_tools import (
        retriever_tool,
        db_executor_tool,
        appointment_availability_check_tool,
        booking_appointment,
        end_call,
        human_handoff,
    )
except ImportError:
    print("[LLM] ⚠️ voice_agent_tools not available")

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """LLM configuration."""
    provider: str = "GeminiAI"
    model: str = "gemini-2.5-flash"
    api_key: str = ""
    temperature: float = 0.7


@dataclass
class VoiceRequestParams:
    """Voice agent request parameters."""
    llm: LLMConfig
    chatbot_id: str = "default"
    thread_id: str = ""
    custom_prompt: str = ""
    time_zone: str = "UTC"
    user_details: Dict[str, Any] = None
    static_tools: List[Dict[str, Any]] = None
    kb_config: Dict[str, Any] = None
    tts_config: Dict[str, Any] = None
    supported_languages: List[str] = None
    tools: List[Dict[str, Any]] = None
    vector_db: Dict[str, Any] = None
    embedding: Dict[str, Any] = None

    def __post_init__(self):
        if self.user_details is None:
            self.user_details = {}
        if self.static_tools is None:
            self.static_tools = []
        if self.supported_languages is None:
            self.supported_languages = ["english"]
        if self.tools is None:
            self.tools = []


@lru_cache(maxsize=50)
def _get_graph_cache_key(params_hash: str) -> str:
    """Cache key for graph instances."""
    return params_hash


def build_graph_from_config(config: Dict[str, Any]) -> tuple:
    """
    Build voice agent LangGraph from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (graph, graph_params)
    """
    # Extract LLM config
    llm_cfg = config.get("llm", {})
    llm_config = LLMConfig(
        provider=llm_cfg.get("provider", "GeminiAI"),
        model=llm_cfg.get("model", "gemini-2.5-flash"),
        api_key=llm_cfg.get("api_key", ""),
        temperature=llm_cfg.get("temperature", 0.7),
    )

    # Build VoiceRequestParams
    params = VoiceRequestParams(
        llm=llm_config,
        chatbot_id=config.get("chatbot_id", "default"),
        thread_id=config.get("thread_id", ""),
        custom_prompt=config.get("custom_prompt", ""),
        time_zone=config.get("time_zone", "UTC"),
        user_details=config.get("user_details", {}),
        static_tools=config.get("static_tools", []),
        kb_config=config.get("kb_config", {}),
        tts_config=config.get("tts_config", {}),
        supported_languages=config.get("supported_languages", ["english"]),
        tools=config.get("tools", []),
        vector_db=config.get("vector_db", {}),
        embedding=config.get("embedding", {}),
    )

    # Create graph
    graph = create_voice_agent_graph(params)

    logger.info(f"[LLM] Graph built for chatbot_id={params.chatbot_id}")
    return graph, params


async def stream_response(
    user_text: str,
    conversation_history: List,
    graph: StateGraph,
    graph_params: VoiceRequestParams,
    interruption_event: asyncio.Event,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Stream LLM response with tool calling support.

    Args:
        user_text: User's transcribed text
        conversation_history: Conversation history
        graph: LangGraph instance
        graph_params: Voice agent parameters
        interruption_event: Event for barge-in detection

    Yields:
        Dict with type and data:
        - {"type": "llm_token", "text": "...", "standard_gap": ms}
        - {"type": "tool_call", "tool": "...", "query": "..."}
        - {"type": "tool_response", "tool": "...", "citations": [...]}
        - {"type": "end_call"}
        - {"type": "human_handoff"}
        - {"type": "llm_done", "full_text": "...", "timing": {...}, "usage": {...}}
        - {"type": "error", "error": "..."}
    """
    start_time = time.time()
    input_tokens = 0
    output_tokens = 0
    tools_called = []
    full_response = ""
    llm_ttft = None

    print(f"[LLM_STREAM] 🚀 Starting LLM stream for: '{user_text[:100]}...'")
    print(f"[LLM_STREAM] Conversation history: {len(conversation_history)} messages")

    try:
        # Prepare messages
        messages = []

        # Add conversation history
        for msg in conversation_history:
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("role") == "assistant":
                messages.append(AIMessage(content=msg.get("content", "")))

        # Add current user message
        messages.append(HumanMessage(content=user_text))

        print(f"[LLM_STREAM] Prepared {len(messages)} messages for graph")

        # Prepare config with interruption event
        config = RunnableConfig(
            configurable={
                "interruption_event": interruption_event,
                "thread_id": graph_params.thread_id,
                "chatbot_id": graph_params.chatbot_id,
            }
        )

        # Stream response from graph
        event_count = 0
        async for event in graph.astream(
            {"messages": messages},
            config=config,
            stream_mode="updates",
        ):
            event_count += 1
            print(f"[LLM_STREAM] Event #{event_count}: {list(event.keys())}")
            # Check for interruption
            if interruption_event.is_set():
                logger.info("[LLM] Interruption detected during streaming")
                return

            # Process event
            for node_name, node_output in event.items():
                print(f"[LLM_STREAM] Node '{node_name}' output keys: {list(node_output.keys()) if isinstance(node_output, dict) else type(node_output)}")

                if node_name == "llm":
                    # LLM output
                    if "messages" in node_output:
                        print(f"[LLM_STREAM] LLM node has {len(node_output['messages'])} messages")
                        for msg_idx, msg in enumerate(node_output["messages"]):
                            print(f"[LLM_STREAM] Message #{msg_idx}: type={type(msg).__name__}")

                            if isinstance(msg, AIMessage):
                                # Log raw content for debugging
                                print(f"[LLM_STREAM] AIMessage content type: {type(msg.content)}")
                                print(f"[LLM_STREAM] AIMessage content value: {repr(msg.content)[:200]}")
                                print(f"[LLM_STREAM] AIMessage has tool_calls: {hasattr(msg, 'tool_calls')}")
                                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                    print(f"[LLM_STREAM] Tool calls: {msg.tool_calls}")

                                # Track TTFT
                                if llm_ttft is None:
                                    llm_ttft = (time.time() - start_time) * 1000
                                    print(f"[LLM_STREAM] ⏱️ TTFT: {llm_ttft:.0f}ms")

                                # Stream text content
                                if hasattr(msg, 'content') and isinstance(msg.content, str):
                                    print(f"[LLM_STREAM] Content length: {len(msg.content)} chars")

                                    for char in msg.content:
                                        if interruption_event.is_set():
                                            return

                                        full_response += char
                                        output_tokens += 1

                                        yield {
                                            "type": "llm_token",
                                            "text": char,
                                            "standard_gap": 0,
                                        }
                                elif hasattr(msg, 'content') and isinstance(msg.content, list):
                                    # 🔥 FIX: Handle Gemini 2.5 structured content format
                                    print(f"[LLM_STREAM] 🔧 Handling structured content (list) from Gemini 2.5")
                                    for block in msg.content:
                                        if isinstance(block, dict) and block.get('type') == 'text':
                                            text_content = block.get('text', '')
                                            print(f"[LLM_STREAM] 🔧 Extracted text from block: '{text_content[:100]}...'")

                                            for char in text_content:
                                                if interruption_event.is_set():
                                                    return

                                                full_response += char
                                                output_tokens += 1

                                                yield {
                                                    "type": "llm_token",
                                                    "text": char,
                                                    "standard_gap": 0,
                                                }
                                        else:
                                            print(f"[LLM_STREAM] ⚠️ Unknown block type: {type(block)} = {repr(block)[:100]}")
                                else:
                                    print(f"[LLM_STREAM] ⚠️ Content is not a string: {type(msg.content)}")
                                    if msg.content is None:
                                        print(f"[LLM_STREAM] Content is None!")
                                    elif isinstance(msg.content, list):
                                        print(f"[LLM_STREAM] Content is list with {len(msg.content)} items")
                                        for i, item in enumerate(msg.content):
                                            print(f"[LLM_STREAM]   Item #{i}: {type(item).__name__} = {repr(item)[:100]}")

                                # Handle tool calls
                                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                    for tool_call in msg.tool_calls:
                                        tool_name = tool_call.get("name", "unknown")
                                        if tool_name not in ["end_call", "human_handoff"]:
                                            if tool_name not in tools_called:
                                                tools_called.append(tool_name)

                                            yield {
                                                "type": "tool_call",
                                                "tool": tool_name,
                                                "query": str(tool_call.get("args", {})),
                                            }

                elif node_name == "tools":
                    # Tool output
                    if "messages" in node_output:
                        for msg in node_output["messages"]:
                            if isinstance(msg, ToolMessage):
                                tool_name = msg.name if hasattr(msg, 'name') else "unknown"

                                yield {
                                    "type": "tool_response",
                                    "tool": tool_name,
                                    "content": msg.content,
                                }

                                # Check for special tools
                                if tool_name == "end_call":
                                    yield {"type": "end_call"}
                                elif tool_name == "human_handoff":
                                    yield {"type": "human_handoff"}

        # Calculate timing
        total_time = (time.time() - start_time) * 1000

        # Estimate input tokens (rough estimate)
        input_tokens = sum(len(m.content) // 4 for m in messages if hasattr(m, 'content'))

        # 🔍 EXPLICIT: Final summary
        print(f"[LLM_STREAM] 📊 STREAM SUMMARY:")
        print(f"[LLM_STREAM]   Total events: {event_count}")
        print(f"[LLM_STREAM]   Full response length: {len(full_response)} chars")
        print(f"[LLM_STREAM]   Full response: '{full_response}'")
        print(f"[LLM_STREAM]   Output tokens: {output_tokens}")
        print(f"[LLM_STREAM]   Tools called: {tools_called}")
        print(f"[LLM_STREAM]   Total time: {total_time:.0f}ms")

        yield {
            "type": "llm_done",
            "full_text": full_response,
            "timing": {
                "llm_ttft_ms": llm_ttft,
                "llm_streaming_ms": total_time - (llm_ttft or 0),
                "graph_total_ms": total_time,
            },
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        }

        logger.info(f"[LLM] Streaming complete: {full_response[:100]}...")
        logger.info(f"[LLM] Timing: TTFT={llm_ttft}ms, Total={total_time}ms")

    except asyncio.CancelledError:
        logger.info("[LLM] Cancelled due to barge-in")
        yield {
            "type": "error",
            "error": "Cancelled due to barge-in",
        }
    except Exception as e:
        logger.error(f"[LLM] Error: {e}", exc_info=True)
        yield {
            "type": "error",
            "error": str(e),
        }


def get_voice_system_prompt(
    custom_prompt: str = "",
    chatbot_id: str = "default",
    time_zone: str = "UTC",
    user_details: Dict[str, Any] = None,
    static_tools: List = None,
    kb_config: Dict = None,
    tts_config: Dict = None,
    supported_languages: List[str] = None,
) -> str:
    """
    Get voice agent system prompt.

    Args:
        custom_prompt: Custom system prompt
        chatbot_id: Chatbot ID
        time_zone: Time zone
        user_details: User details
        static_tools: Static tools enabled
        kb_config: Knowledge base config
        tts_config: TTS config
        supported_languages: Supported languages

    Returns:
        System prompt string
    """
    base_prompt = """You are a helpful voice assistant. Your responses should be:

1. **Concise and Natural**: Keep responses brief (1-2 sentences typically) for voice interactions.
2. **Conversational**: Use natural language appropriate for speech, not formal writing.
3. **No Special Formatting**: Avoid using markdown, bullets, or other text formatting that won't be spoken well.
4. **Action-Oriented**: When appropriate, guide users toward clear next steps.

When using tools, provide natural transitions like "Let me check that for you" or "I'll help you with that."

If a user's request is unclear, ask a brief clarifying question rather than making assumptions.
"""

    if custom_prompt:
        return f"{base_prompt}\n\nAdditional instructions:\n{custom_prompt}"

    return base_prompt
