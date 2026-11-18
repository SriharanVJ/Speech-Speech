"""
Voice Agent LangGraph - Complete Voice Agent with Tools

Based on Z-agent-llm app/agents/voice_agent/graph.py

Features:
- LangGraph with LLM and tool nodes
- Voice-specific system prompt
- Tool calling support
- Interruption support for barge-in
- State management
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("[VoiceGraph] ⚠️ langchain-google-genai not available")

logger = logging.getLogger(__name__)


# =============================================================================
# STATE
# =============================================================================


@dataclass
class VoiceAgentState:
    """Voice agent state for LangGraph."""
    messages: List = field(default_factory=list)
    retrieved_documents: List = field(default_factory=list)
    time_zone: str = "UTC"
    chatbot_id: str = "default"
    re_ranker: bool = False
    hybrid_search: bool = False
    compress_now: bool = False
    compress_only: bool = False


# =============================================================================
# IMPORT TOOLS
# =============================================================================


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
    print("[VoiceGraph] ⚠️ voice_agent_tools not available")
    # Create dummy tools for now
    from langchain_core.tools import tool

    @tool
    def retriever_tool(query: str) -> str:
        return f"Retrieved: {query}"

    @tool
    def db_executor_tool(query: str) -> str:
        return f"DB result: {query}"

    @tool
    def appointment_availability_check_tool(date: str, time_zone: str = "UTC") -> str:
        return f"Available slots on {date}: 9:00 AM, 2:00 PM, 4:00 PM"

    @tool
    def booking_appointment(date: str, time: str, reason: str = "", time_zone: str = "UTC") -> str:
        return f"Appointment booked for {date} at {time}"

    @tool
    def end_call() -> str:
        return "Thank you for calling. Have a great day!"

    @tool
    def human_handoff() -> str:
        return "I'm transferring you to a human agent."


# =============================================================================
# PROMPTS
# =============================================================================


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


# =============================================================================
# GRAPH
# =============================================================================


def create_voice_agent_graph(params) -> StateGraph:
    """
    Build the voice agent LangGraph.

    Args:
        params: VoiceRequestParams with configuration

    Returns:
        Compiled StateGraph
    """
    print(f"[VoiceGraph] 🏗️ Building voice agent graph")
    logger.info("[VoiceGraph] Building voice agent graph")

    # Initialize LLM
    try:
        print(f"[VoiceGraph] 🔧 GEMINI_AVAILABLE: {GEMINI_AVAILABLE}")
        if not GEMINI_AVAILABLE:
            raise ImportError("langchain-google-genai not installed")

        print(f"[VoiceGraph] 🔑 API Key length: {len(params.llm.api_key)}")
        print(f"[VoiceGraph] 📡 Model: {params.llm.model}")
        print(f"[VoiceGraph] 🌡️ Temperature: {params.llm.temperature}")

        print(f"[VoiceGraph] 🚀 Initializing ChatGoogleGenerativeAI...")
        llm = ChatGoogleGenerativeAI(
            model=params.llm.model,
            api_key=params.llm.api_key,
            temperature=params.llm.temperature,
        )
        print(f"[VoiceGraph] ✅ LLM loaded: {params.llm.model}")
        logger.info(f"[VoiceGraph] LLM loaded: {params.llm.model}")
    except Exception as e:
        print(f"[VoiceGraph] ❌ LLM init failed: {e}")
        import traceback
        traceback.print_exc()
        logger.error(f"[VoiceGraph] LLM init failed: {e}")
        raise ValueError(f"LLM init failed: {e}")

    # Build tool list
    tools = [
        retriever_tool,
        db_executor_tool,
        appointment_availability_check_tool,
        booking_appointment,
        end_call,
        human_handoff,
    ]

    tools_by_name = {t.name: t for t in tools}
    logger.info(f"[VoiceGraph] Tools: {list(tools_by_name.keys())}")

    # Bind tools to LLM
    model = llm.bind_tools(tools)

    # Define graph nodes

    async def call_model(state: VoiceAgentState, config: RunnableConfig) -> Dict[str, Any]:
        """LLM node — generates response or tool calls."""
        try:
            interruption_event = config.get("configurable", {}).get("interruption_event")
            if interruption_event and interruption_event.is_set():
                logger.warning("[VoiceGraph] LLM node aborted due to barge-in")
                raise asyncio.CancelledError("Barge-in")

            messages = state.messages
            print(f"[VoiceGraph] 🔔 LLM Invoked with {len(messages)} messages")
            for i, msg in enumerate(messages):
                print(f"[VoiceGraph]   Message #{i}: {type(msg).__name__} - {str(msg)[:100]}...")

            # Build system prompt
            system_prompt = get_voice_system_prompt(
                custom_prompt=params.custom_prompt,
                chatbot_id=params.chatbot_id,
                time_zone=params.time_zone,
                user_details=params.user_details,
                static_tools=params.static_tools,
                kb_config=params.kb_config,
                tts_config=params.tts_config,
                supported_languages=params.supported_languages,
            )
            messages_with_prompt = [SystemMessage(content=system_prompt)] + list(messages)
            print(f"[VoiceGraph] 📝 System prompt length: {len(system_prompt)} chars")

            # Invoke LLM
            start_time = asyncio.get_event_loop().time()
            print(f"[VoiceGraph] 🚀 Calling LLM (model={params.llm.model})...")
            response = await model.ainvoke(messages_with_prompt, config)
            elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000

            # 🔍 EXPLICIT: Log response details
            print(f"[VoiceGraph] ✅ LLM response received in {elapsed_ms:.0f}ms")
            print(f"[VoiceGraph] 📦 Response type: {type(response).__name__}")
            print(f"[VoiceGraph] 📦 Response content type: {type(response.content)}")
            print(f"[VoiceGraph] 📦 Response.content value: {repr(response.content)[:300]}")
            print(f"[VoiceGraph] 🔧 Has tool_calls: {hasattr(response, 'tool_calls')}")
            if hasattr(response, 'tool_calls') and response.tool_calls:
                print(f"[VoiceGraph] 🔧 Tool calls: {response.tool_calls}")
            print(f"[VoiceGraph] 📏 Content length: {len(response.content) if isinstance(response.content, str) else 'N/A'}")

            logger.info(f"[VoiceGraph] LLM response: {response.content[:100] if response.content else 'tool_calls'}...")
            logger.info(f"[VoiceGraph] LLM time: {elapsed_ms:.0f}ms")

            return {
                "messages": [response],
                "retrieved_documents": state.retrieved_documents,
            }

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"[VoiceGraph] call_model error: {e}")
            return {
                "messages": [AIMessage(content="I'm sorry, something went wrong. Please try again.")],
                "retrieved_documents": state.retrieved_documents,
            }

    async def call_tool(state: VoiceAgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Tool execution node — runs tool calls from LLM."""
        try:
            interruption_event = config.get("configurable", {}).get("interruption_event")
            if interruption_event and interruption_event.is_set():
                logger.warning("[VoiceGraph] Tool node aborted due to barge-in")
                raise asyncio.CancelledError("Barge-in")

            outputs = []
            for tool_call in state.messages[-1].tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                logger.info(f"[VoiceGraph] Tool call: {tool_name} with args: {tool_args}")

                # Inject time_zone for appointment tools
                if tool_name in ["appointment_availability_check_tool", "booking_appointment"]:
                    tool_args["time_zone"] = state.time_zone

                # Execute tool
                tool_result = await tools_by_name[tool_name].ainvoke(tool_args, config)

                logger.info(f"[VoiceGraph] Tool {tool_name} returned: {str(tool_result)[:100]}...")

                outputs.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"],
                    )
                )

            return {
                "messages": outputs,
                "retrieved_documents": state.retrieved_documents,
            }

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"[VoiceGraph] call_tool error: {e}")
            return {
                "messages": [
                    ToolMessage(
                        content=f"Error: {str(e)}",
                        name=tool_name if 'tool_name' in locals() else "unknown",
                        tool_call_id=state.messages[-1].tool_calls[0]["id"] if state.messages[-1].tool_calls else "unknown",
                    )
                ],
                "retrieved_documents": state.retrieved_documents,
            }

    # Conditional edge

    def should_continue(state: VoiceAgentState) -> str:
        messages = state.messages
        last = messages[-1]
        if not last.tool_calls:
            return "end"
        first_tc = last.tool_calls[0]
        tc_name = first_tc.get("name") if isinstance(first_tc, dict) else getattr(first_tc, "name", None)
        if tc_name in ("end_call", "human_handoff"):
            return "end"
        return "continue"

    # Compression node (placeholder for future implementation)

    async def compress_history(state: VoiceAgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Compress conversation history (placeholder)."""
        return {"compress_now": False}

    def route_entry(state: VoiceAgentState) -> str:
        """Entry router — go through `compress` node when flag is set, else straight to `llm`."""
        return "compress" if state.compress_now else "llm"

    def route_post_compress(state: VoiceAgentState) -> str:
        """After compress runs: exit if compress_only=True, else fall through to llm."""
        return "end" if state.compress_only else "llm"

    # Build the graph

    try:
        workflow = StateGraph(VoiceAgentState)

        workflow.add_node("llm", call_model)
        workflow.add_node("tools", call_tool)
        workflow.add_node("compress", compress_history)

        # Entry routing
        workflow.add_conditional_edges(
            START,
            route_entry,
            {"compress": "compress", "llm": "llm"},
        )

        # Post-compress routing
        workflow.add_conditional_edges(
            "compress",
            route_post_compress,
            {"end": END, "llm": "llm"},
        )

        # LLM routing
        workflow.add_conditional_edges(
            "llm",
            should_continue,
            {"continue": "tools", "end": END},
        )

        workflow.add_edge("tools", "llm")

        # Compile graph
        checkpointer = None  # Could add MemorySaver() for state persistence
        compiled = workflow.compile(checkpointer=checkpointer)
        logger.info("[VoiceGraph] Graph compiled successfully")
        return compiled

    except Exception as e:
        logger.error(f"[VoiceGraph] Graph compilation failed: {e}")
        raise ValueError(f"Graph compilation failed: {e}")
