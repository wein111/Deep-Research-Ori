"""State Definitions and Pydantic Schemas for Research Scoping.
"""

import operator
from typing_extensions import Optional, Annotated, List, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# ===== STATE DEFINITIONS =====

#整个DR输入的state
class AgentInputState(MessagesState):
    """Input state for the full agent - only contains messages from user input."""
    pass

class AgentState(MessagesState):
    """
    Main state for the full multi-agent research system.
    Extends MessagesState with additional fields for research coordination.
    """
    # Research brief generated from user conversation history
    research_brief: Optional[str]
    # Messages exchanged with the supervisor agent for coordination
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    # Raw unprocessed research notes collected during the research phase
    raw_notes: Annotated[list[str], operator.add] = []
    # Processed and structured notes ready for report generation
    notes: Annotated[list[str], operator.add] = []
    # Final formatted research report
    final_report: str

class ResearcherState(TypedDict):
    """
    State for the research agent containing message history and research metadata.
    """
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_call_iterations: int
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[List[str], operator.add]

class ResearcherOutputState(TypedDict):
    """
    Output state for the research agent containing final research results.
    """
    compressed_research: str
    raw_notes: Annotated[List[str], operator.add]
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]

class SupervisorState(TypedDict):
    """
    State for the multi-agent research supervisor.
    """
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    research_brief: str
    notes: Annotated[list[str], operator.add] = []
    raw_notes: Annotated[list[str], operator.add] = []
    research_iterations: int = 0


# ===== STRUCTURED OUTPUT SCHEMAS =====

class ClarifyWithUser(BaseModel):
    """Schema for user clarification decision and questions."""

    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    verification: str = Field(
        description="When no clarification is needed: brief scope summary plus one line pointing to the Approve/Revise controls (do not ask the user to type approve/revise). When ready to research: acknowledgement that research will start.",
    )
    ready_to_research: bool = Field(
        description="Whether the user has already confirmed the scope and the system should proceed to research now.",
    )

class ResearchQuestion(BaseModel):
    """Schema for structured research brief generation."""

    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )


class Summary_of_One_Research(BaseModel):
    """Schema for webpage content summarization."""
    summary: str = Field(description="Concise summary of the webpage content")
    key_excerpts: str = Field(description="Important quotes and excerpts from the content")

