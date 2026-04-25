import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env", override=True)

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError

from config import (
    DEFAULT_RECURSION_LIMIT,
    FINAL_REPORT_MAX_TOKENS,
    FINAL_REPORT_MODEL_NAME,
)
from prompts import final_report_generation_prompt
from scoping import clarify_with_user, human_scope_review, write_research_brief
from state import AgentState
from supervisor import supervisor_agent
from utils import format_messages, get_today_str

try:
    from IPython import get_ipython

    ipython = get_ipython()
    if ipython is not None:
        ipython.run_line_magic("load_ext", "autoreload")
        ipython.run_line_magic("autoreload", "2")
except Exception:
    pass

write_model = init_chat_model(
    model=FINAL_REPORT_MODEL_NAME,
    max_tokens=FINAL_REPORT_MAX_TOKENS,
)

@traceable(run_type="chain", name="final_report_generation")
async def final_report_generation(state:AgentState):
    #写最终报告node
    notes = state.get("notes",[])

    findings = "\n".join(notes)

    final_report_prompt = final_report_generation_prompt.format(
        research_brief=state.get("research_brief", ""),
        findings=findings,
        date=get_today_str()
    )

    final_report = await write_model.ainvoke([HumanMessage(content=final_report_prompt)])

    return {
        "final_report": final_report.content, 
        "messages": [AIMessage(content="Here is the final report: " + final_report.content)],
    }

deep_researcher_builder = StateGraph(AgentState)

deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("human_scope_review", human_scope_review)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)
deep_researcher_builder.add_node("supervisor_subgraph", supervisor_agent)
deep_researcher_builder.add_node("final_report_generation", final_report_generation)

deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("write_research_brief", "supervisor_subgraph")
deep_researcher_builder.add_edge("supervisor_subgraph", "final_report_generation")
deep_researcher_builder.add_edge("final_report_generation", END)

_mem = os.environ.get("CHECKPOINT_BACKEND", "redis").strip().lower() == "memory"
_redis_url = os.environ.get("REDIS_URL")
_checkpoint_ttl_minutes = int(os.environ.get("CHECKPOINT_TTL_MINUTES", "1440"))
_checkpoint_ttl = (
    {"default_ttl": _checkpoint_ttl_minutes, "refresh_on_read": True}
    if _checkpoint_ttl_minutes > 0
    else None
)
_cp_ready = False

checkpointer = (
    InMemorySaver()
    if _mem
    else AsyncRedisSaver(
        redis_url=_redis_url,
        connection_args={"decode_responses": False},
        ttl=_checkpoint_ttl,
    )
)


async def ensure_redis_checkpoint_initialized() -> None:
    """首次 ainvoke/astream 前在同一事件循环里 await setup"""
    global _cp_ready
    if _mem or _cp_ready:
        return
    try:
        await checkpointer.setup()
    except RedisConnectionError as e:
        raise RuntimeError(
            f"Redis 连不上 {_redis_url!r}: {e}。起 Stack: docker compose up -d。或 CHECKPOINT_BACKEND=memory"
        ) from e
    except ResponseError as e:
        msg = str(e).lower()
        if "ft." not in msg and "unknown command" not in msg:
            raise
        raise RuntimeError(
            f"{_redis_url!r} 需要 Redis Stack/RediSearch。docker compose up -d 或 CHECKPOINT_BACKEND=memory"
        ) from e
    _cp_ready = True

agent = deep_researcher_builder.compile(checkpointer=checkpointer)

thread = {
    "configurable": {"thread_id": "1", "recursion_limit": DEFAULT_RECURSION_LIMIT}
    } #默认25step终止，把上限拉高

async def main():
    async def run_and_print(user_input: str):
        await ensure_redis_checkpoint_initialized()
        final_state = None
        visited_nodes = []

        async for chunk in agent.astream(
            {"messages": [HumanMessage(content=user_input)]},
            config=thread,
            stream_mode="updates",
        ):
            final_state = chunk
            for node_name, update in chunk.items():
                visited_nodes.append(node_name)
                print(f"\n=== {node_name} ===")

                if "messages" in update:
                    format_messages(update["messages"])

                if "supervisor_messages" in update:
                    format_messages(update["supervisor_messages"])

        print("\n=== nodes visited ===")
        print(" -> ".join(visited_nodes))

        return final_state

    await run_and_print("I want to research the best shops in Los angeles.")
    await run_and_print("Let's examine coffee quality to assess the best coffee shops in LA")


if __name__ == "__main__":
    asyncio.run(main())