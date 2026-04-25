import asyncio
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.types import Command

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import DEFAULT_RECURSION_LIMIT  # noqa: E402
from full_agent import agent, ensure_redis_checkpoint_initialized  # noqa: E402
from utils import format_messages  # noqa: E402


async def run_agent(query: str, thread_id: str = "1"):
    await ensure_redis_checkpoint_initialized()
    config = {
        "configurable": {
            "thread_id": thread_id,
            "recursion_limit": DEFAULT_RECURSION_LIMIT,
        }
    }
    return await agent.ainvoke(
        {"messages": [HumanMessage(content=query)]},
        config=config,
    )


async def resume_agent(action: str, thread_id: str = "1"):
    await ensure_redis_checkpoint_initialized()
    config = {
        "configurable": {
            "thread_id": thread_id,
            "recursion_limit": DEFAULT_RECURSION_LIMIT,
        }
    }
    payload = {"action": action.strip().lower()}
    return await agent.ainvoke(Command(resume=payload), config=config)


async def main():
    query = "I want to research the best coffee shops in Los Angeles."
    query2 = "I want to find a cafe to study, so based on environment, parking acessibility, seats..."
    result = await run_agent(query)
    result2 = await run_agent(query2)
    format_messages(result2["messages"])


if __name__ == "__main__":
    asyncio.run(main())
