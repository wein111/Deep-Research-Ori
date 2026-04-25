import asyncio

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage, filter_messages
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from langsmith import traceable
from pydantic import BaseModel, Field

from config import (
    MAX_CONCURRENT_RESEARCHERS,
    MAX_RESEARCHER_ITERATIONS,
    SUPERVISOR_MODEL_NAME,
)
from prompts import lead_researcher_prompt
from research import researcher_agent, think_tool
from state import SupervisorState
from utils import format_messages, get_today_str

load_dotenv()

try:
    from IPython import get_ipython

    ipython = get_ipython()
    if ipython is not None:
        ipython.run_line_magic("load_ext", "autoreload")
        ipython.run_line_magic("autoreload", "2")
except Exception:
    pass

@tool
class ConductResearch(BaseModel):
    """Tool for delegating a research task to a specialized sub-agent."""
    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )

@tool
class ResearchComplete(BaseModel):
    """Tool for indicating that the research process is complete."""
    pass


def get_notes_from_tool_calls(messages:list[BaseMessage])->list[str]:
    result = [tool_msg.content for tool_msg in filter_messages(messages,include_types="tool")]
    return result


supervisor_tools = [ConductResearch, ResearchComplete, think_tool]
supervisor_model = init_chat_model(model=SUPERVISOR_MODEL_NAME)
supervisor_model_with_tools = supervisor_model.bind_tools(supervisor_tools)

@traceable(run_type="chain", name="supervisor")
async def supervisor(state:SupervisorState):
    #监督者基于已有信息，决定下一步该做什么
    supervisor_messages = state.get("supervisor_messages", [])

    system_message = lead_researcher_prompt.format(
        date=get_today_str(), 
        max_concurrent_research_units=MAX_CONCURRENT_RESEARCHERS,
        max_researcher_iterations=MAX_RESEARCHER_ITERATIONS
    )

    input_messages = [SystemMessage(content=system_message)] + supervisor_messages

    response = await supervisor_model_with_tools.ainvoke(input_messages)

    return Command(
        #这里把supervisor node作为一个大监督agent，判断是结束or继续or思考
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )

@traceable(run_type="chain", name="supervisor_tools")
async def supervisor_tools(state:SupervisorState):
    #调用researcher
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)

    most_recent_message = supervisor_messages[-1]

    tool_messages = []
    all_raw_notes = []
    next_node = "supervisor"  # Default next step
    should_end = False

    #判断是否进入终止
    if_exceeded_iterations = research_iterations >= MAX_RESEARCHER_ITERATIONS
    if_no_tool_calls = not most_recent_message.tool_calls
    if_research_complete = any(tool_call["name"]=="ResearchComplete" for tool_call in most_recent_message.tool_calls)
    #三种情况都会终止
    if if_exceeded_iterations or if_no_tool_calls or if_research_complete:
        should_end = True
        next_node = END
    else:
        try:
            think_tool_calls = [
                tool_call for tool_call in most_recent_message.tool_calls 
                if tool_call["name"] == "think_tool"
            ]
            
            conduct_research_calls = [
                tool_call for tool_call in most_recent_message.tool_calls 
                if tool_call["name"] == "ConductResearch"
            ]

            #执行think_tools
            for tool_call in think_tool_calls:
                observation = think_tool.invoke(tool_call["args"])
                tool_messages.append(
                    ToolMessage(
                        content=observation,
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    )
                )
            #执行research_tools
            if conduct_research_calls: 
                coros= [
                    researcher_agent.ainvoke({
                        "researcher_messages":[HumanMessage(content=tool_call["args"]["research_topic"])],
                        "research_topic":tool_call["args"]["research_topic"]
                    })
                    for tool_call in conduct_research_calls
                ]

                researcher_tool_results = await asyncio.gather(*coros)

                researcher_tool_messages = [
                    ToolMessage(
                        content=result.get("compressed_research", "Error synthesizing research report"),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    ) for result, tool_call in zip(researcher_tool_results, conduct_research_calls)
                ]

                tool_messages.extend(researcher_tool_messages)

                all_raw_notes = [
                    "\n".join(result.get("raw_notes", [])) 
                    for result in researcher_tool_results
                ]
        
        except Exception as e:
            print(f"Error in supervisor tools: {e}")
            should_end = True
            next_node = END
        
    if should_end==True:
        return Command(
            goto=next_node,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )
    else:
        return Command(
            goto=next_node,
            update={
                "supervisor_messages": tool_messages,
                "raw_notes": all_raw_notes
            }
        )

supervisor_builder = StateGraph(SupervisorState)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_edge(START,"supervisor")

supervisor_agent = supervisor_builder.compile()


async def main():
    research_brief = """I want to identify and evaluate the coffee shops in San Francisco that are considered the best based specifically  
    on coffee quality. My research should focus on analyzing and comparing coffee shops within the San Francisco area, 
    using coffee quality as the primary criterion. I am open regarding methods of assessing coffee quality (e.g.,      
    expert reviews, customer ratings, specialty coffee certifications), and there are no constraints on ambiance,      
    location, wifi, or food options unless they directly impact perceived coffee quality. Please prioritize primary    
    sources such as the official websites of coffee shops, reputable third-party coffee review organizations (like     
    Coffee Review or Specialty Coffee Association), and prominent review aggregators like Google or Yelp where direct  
    customer feedback about coffee quality can be found. The study should result in a well-supported list or ranking of
    the top coffee shops in San Francisco, emphasizing their coffee quality according to the latest available data as  
    of July 2025."""

    result = await supervisor_agent.ainvoke({"supervisor_messages": [HumanMessage(content=f"{research_brief}.")]})
    format_messages(result['supervisor_messages'])


if __name__ =="__main__":
    asyncio.run(main())