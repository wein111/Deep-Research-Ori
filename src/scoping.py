from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from langsmith import traceable
from rich.markdown import Markdown

from config import SCOPING_MODEL_NAME, SCOPING_TEMPERATURE
from prompts import (
    clarify_with_user_instructions,
    transform_messages_into_research_topic_prompt,
)
from state import AgentState, ClarifyWithUser, ResearchQuestion
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


model = init_chat_model(model=SCOPING_MODEL_NAME, temperature=SCOPING_TEMPERATURE)


@traceable(run_type="chain", name="clarify_with_user")
def clarify_with_user(state:AgentState):
    # 判断是否要user 去clarify问题
    messages = state["messages"]

    Structured_output_model = model.with_structured_output(ClarifyWithUser)

    user_messages = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages=messages),
        date = get_today_str()
    )

    response = Structured_output_model.invoke([
        HumanMessage(content = user_messages)
    ])

    if response.need_clarification:
        return Command(
            goto=END,
            update={"messages":[AIMessage(content=response.question)]}
        )

    # 无需澄清时先返回确认消息，再进入独立的 HITL 节点。
    return Command(
        goto="human_scope_review",
        update={"messages": [AIMessage(content=response.verification)]},
    )


@traceable(run_type="chain", name="human_scope_review")
def human_scope_review(_state: AgentState):
    # 在独立节点里做 interrupt，避免把 clarify 与 HITL 逻辑耦合到一起
    review = interrupt("请使用 Approve / Revise 按钮继续。")
    action = ""
    if isinstance(review, str):
        action = review.strip().lower()
    elif isinstance(review, dict):
        action = str(review.get("action", "")).strip().lower()

    if action == "approve":
        return Command(
            goto="write_research_brief",
            update={
                "messages": [
                    AIMessage(
                        content="Scope approved. I will now turn it into a research brief and begin the research process."
                    )
                ]
            },
        )
    
    # revise: no feedback field accepted. Ask user to send revised scope next.
    return Command(
        goto=END,
        update={
            "messages": [
                AIMessage(
                    content=(
                        "Got it. Please provide your revised scope in the next message, "
                        "and I will re-clarify before research."
                    )
                )
            ]
        },
    )
    
@traceable(run_type="chain", name="write_research_brief")
def write_research_brief(state:AgentState):
    # 写加长版问题
    structured_output_model = model.with_structured_output(ResearchQuestion)

    input_messages = transform_messages_into_research_topic_prompt.format(
        messages = get_buffer_string(state.get('messages',[])),
        date = get_today_str(),
    )

    response = structured_output_model.invoke([
        HumanMessage(content = input_messages)
    ])

    return {
        "research_brief":response.research_brief,
        "supervisor_messages":[HumanMessage(content=f"{response.research_brief}.")]
    }

if __name__=="__main__":
    graph = StateGraph(AgentState)

    # Add workflow nodes
    graph.add_node("clarify_with_user", clarify_with_user)
    graph.add_node("human_scope_review", human_scope_review)
    graph.add_node("write_research_brief", write_research_brief)

    # Add workflow edges
    graph.add_edge(START, "clarify_with_user")
    graph.add_edge("write_research_brief", END)

    checkpointer = InMemorySaver()
    scoping = graph.compile(checkpointer=checkpointer)

    def print_step(title: str, result: dict):
        print(f"\n=== {title} ===")
        format_messages(result.get("messages", []))
        if result.get("research_brief"):
            print("research_brief:", result.get("research_brief"))

    # Flow A: approve path
    cfg_a = {"configurable": {"thread_id": "demo-approve"}}
    a1 = scoping.invoke(
        {"messages": [HumanMessage(content="Research the best coffee shops in San Francisco by coffee quality.")]},
        config=cfg_a,
    )
    print_step("A1 first invoke (should pause for review)", a1)

    a2 = scoping.invoke(Command(resume={"action": "approve"}), config=cfg_a)
    print_step("A2 resume approve (should produce research_brief)", a2)

    # Flow B: revise path
    cfg_b = {"configurable": {"thread_id": "demo-revise"}}
    b1 = scoping.invoke(
        {"messages": [HumanMessage(content="Research top coffee shops in SF.")]},
        config=cfg_b,
    )
    print_step("B1 first invoke (should pause for review)", b1)

    b2 = scoping.invoke(
        Command(resume={"action": "revise"}),
        config=cfg_b,
    )
    print_step("B2 resume revise (should end this turn without assistant output)", b2)

    b3 = scoping.invoke(
        {"messages": [HumanMessage(content="Based on coffee beam.")]},
        config=cfg_b,
    )
    a2 = scoping.invoke(Command(resume={"action": "approve"}), config=cfg_b)
    print_step("A2 resume approve (should produce research_brief)", a2)
    print_step("B3 follow-up invoke (should re-enter clarify and ask for approval again)", a2)