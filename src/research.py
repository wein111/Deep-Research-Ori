import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, filter_messages
from langchain_core.tools import InjectedToolArg, tool
from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from tavily import TavilyClient
from typing_extensions import Annotated, List, Literal

from config import (
    CREDIBILITY_MIN_SCORE,
    MAX_SEARCH_ITERATIONS,
    RESEARCH_AGENT_MODEL_NAME,
    RESEARCH_COMPRESS_MAX_TOKENS,
    RESEARCH_COMPRESS_MODEL_NAME,
    RESEARCH_SUMMARIZATION_MODEL_NAME,
    TAVILY_MAX_RESULTS,
)
from credibility import CredibilityScorer
from prompts import (
    compress_research_human_message,
    compress_research_system_prompt,
    research_agent_prompt,
    summarize_webpage_prompt,
)
from state import (
    ResearcherOutputState,
    ResearcherState,
    Summary_of_One_Research as Summary,
)
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

# 定义模型
research_agent_model = init_chat_model(model=RESEARCH_AGENT_MODEL_NAME)
summarization_model = init_chat_model(model=RESEARCH_SUMMARIZATION_MODEL_NAME)
compress_model = init_chat_model(
    model=RESEARCH_COMPRESS_MODEL_NAME,
    max_tokens=RESEARCH_COMPRESS_MAX_TOKENS,
)

tavily_client = TavilyClient()
_credibility_scorer = CredibilityScorer()


@traceable(run_type="retriever", name="talivy_search_multiple")
def talivy_search_multiple(
    search_queries:List[str],
    max_results: int = TAVILY_MAX_RESULTS,
    topic:Literal["general","news","finance"]="general",
    include_raw_content: bool = True, 
)->List[dict]:
    # 利用query，进行tavily搜索
    search_results=[]
    for q in search_queries:
        result = tavily_client.search(
            q,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic    
        )
        search_results.append(result)

    return search_results


@traceable(run_type="chain", name="deduplicate_search_results")
def deduplicate_search_results(search_results:List[dict])->List[dict]:
    #去除重复的搜索结果
    unique_results={}

    for result in search_results:
        for response in result['results']:
            url=response['url']
            if url not in unique_results:
                unique_results[url]=response

    return unique_results


@traceable(run_type="chain", name="filter_results_by_credibility")
def filter_results_by_credibility(
    results: dict,
    *,
    min_score: int | None = None,
    ensure_at_least_one: bool = True,
) -> dict:
    """
    In/out same shape as ``deduplicate_search_results``: ``{url: tavily_row, ...}``.

    Converts to rows, drops low-credibility rows, then rebuilds the url-keyed
    dict (skips rows missing ``url``). If none pass the threshold and
    ``ensure_at_least_one`` is True, keeps the single highest-scoring row.
    """
    if not results:
        return {}
    rows = list(results.values())
    thr = (
        min_score
        if min_score is not None
        else int(os.environ.get("CREDIBILITY_MIN_SCORE", str(CREDIBILITY_MIN_SCORE)))
    )
    filtered = _credibility_scorer.filter_by_credibility(
        rows,
        min_score=thr,
        ensure_at_least_one=ensure_at_least_one,
    )
    return {
        r["url"]: r
        for r in filtered
        if isinstance(r, dict) and r.get("url")
    }


@traceable(run_type="llm", name="summarize_raw_content")
def summarize_RawContent_to_content(raw_content:str):
    #把搜索结果的raw_content转成content
    try:
        structured_output_model = summarization_model.with_structured_output(Summary)
        
        # Generate summary
        summary = structured_output_model.invoke([
            HumanMessage(content=summarize_webpage_prompt.format(
                webpage_content=raw_content, 
                date=get_today_str()
            ))
        ])
        
        # Format summary with clear structure
        formatted_summary = (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )
        
        return formatted_summary
        
    except Exception as e:
        print(f"Failed to summarize webpage: {str(e)}")
        return raw_content[:1000] + "..." if len(raw_content) > 1000 else raw_content


@traceable(run_type="chain", name="process_search_results")
def process_search_results(search_results):
    #预处理搜索结果
    processed_results={}
    
    for url,result in search_results.items():
        #优先把raw_content总结后输出
        if result.get("raw_content"):
            content = summarize_RawContent_to_content(result['raw_content'])
        else:
            content = result['content']

        processed_results[url]= {
            'content':content,
            'title':result['title']
        }
    
    return processed_results


@traceable(run_type="chain", name="format_search_output")
def format_search_output(processed_results:dict):
    #给搜索结果排好格式
    if not processed_results:
        return "No valid search results found!!!"

    formatted_results="Search Results: \n\n"

    for i,(url,result) in enumerate(processed_results.items(),1):
        formatted_results += f"\n\n--- SOURCE {i}: {result['title']} ---\n"
        formatted_results += f"URL: {url}\n\n"
        formatted_results += f"SUMMARY:\n{result['content']}\n\n"
        formatted_results += "-" * 80 + "\n"
        
    return formatted_results

#define tools
@tool(parse_docstring=True)
@traceable(run_type="tool", name="tavily_search")
def tavily_search(
    query:str,
    max_results: Annotated[int, InjectedToolArg] = TAVILY_MAX_RESULTS,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
)->str:
    """
    Get results from Tavily search API with content summarization.

    Args:
        query: A single search query to execute
        max_results: Maximum number of results to return
        topic: Topic to filter results by ('general', 'news', 'finance')
    """
    search_results = talivy_search_multiple(
        [query],
        max_results=max_results,
        topic = topic,
        include_raw_content=True,
    )

    unique_results = deduplicate_search_results(search_results)
    credibility_results = filter_results_by_credibility(unique_results)
    processed_results = process_search_results(credibility_results)
    formatted_results = format_search_output(processed_results)

    return formatted_results

@tool(parse_docstring=True)
@traceable(run_type="tool", name="think_tool")
def think_tool(reflection:str)->str:
    """Tool for strategic reflection on research progress and decision-making.
    
    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.
    
    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?
    
    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?
    
    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps
    """
    return f"Reflection recorded:{reflection}"


#构建图
tools = [tavily_search,think_tool]
tools_by_name= {tool.name:tool for tool in tools}

# Initialize models
model_with_tools = research_agent_model.bind_tools(tools)

@traceable(run_type="chain", name="research_llm_call")
def llm_call(state:ResearcherState):
    # 负责决定query以及要不要继续搜索
    messages_list_input = [SystemMessage(content=research_agent_prompt)]+state["researcher_messages"]
    response = model_with_tools.invoke(messages_list_input)

    return {
        "researcher_messages":[response]
    }

@traceable(run_type="chain", name="research_tool_node")
def tool_node(state:ResearcherState):
    # 调用工具
    search_iterations = state.get("tool_call_iterations", 0) + 1

    tool_calls = state["researcher_messages"][-1].tool_calls

    observations=[]
    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        tool_response = tool.invoke(tool_call["args"])
        observations.append(tool_response)

    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"],
        ) for observation, tool_call in zip(observations, tool_calls)
    ]

    return {
        "researcher_messages": tool_outputs,
        "tool_call_iterations": search_iterations,
    }

@traceable(run_type="chain", name="compress_research")
def compress_research(state:ResearcherState):
    # 总结搜索结果
    system_message = compress_research_system_prompt.format(date=get_today_str())
    messages = [SystemMessage(content=system_message)] + state.get("researcher_messages", []) + [HumanMessage(content=compress_research_human_message)]
    response = compress_model.invoke(messages)

    raw_notes=[]
    for m in filter_messages(state["researcher_messages"],include_types=["tool","ai"]):
        raw_messages = str(m.content)
        raw_notes.append(raw_messages)
    
    return {
        "compressed_research": str(response.content),
        "raw_notes": ["\n".join(raw_notes)]
    }

def should_continue(state: ResearcherState):
    last_message = state["researcher_messages"][-1]

    if last_message.tool_calls:
        return "tool_node"
    else:    
        return "compress_research"


def check_iterations(state: ResearcherState):
    search_iterations = state.get("tool_call_iterations", 0)

    if search_iterations >= MAX_SEARCH_ITERATIONS:
        return "compress_research"

    return "llm_call"


# Build the agent workflow
agent_builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)

# Add nodes to the graph
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_node("compress_research", compress_research)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_node":"tool_node",
        "compress_research":"compress_research",
    }
)
agent_builder.add_conditional_edges(
    "tool_node",
    check_iterations,
    {
        "llm_call": "llm_call",
        "compress_research": "compress_research",
    }
)
agent_builder.add_edge("compress_research", END)

researcher_agent = agent_builder.compile()

if __name__ == "__main__":
    # --- quick credibility filter smoke test (no Tavily call) ---
    _fake_tavily_pages = [
        {
            "results": [
                {
                    "url": "https://www.cdc.gov/travel/index.html",
                    "title": "CDC travel",
                    "content": "trusted gov",
                },
                {
                    "url": "https://spammy-news.ml/clickbait",
                    "title": "Suspicious",
                    "content": "low trust TLD",
                },
                {
                    "url": "https://www.reuters.com/world/example",
                    "title": "Reuters",
                    "content": "trusted news",
                },
            ]
        }
    ]
    _uniq = deduplicate_search_results(_fake_tavily_pages)
    print("[credibility test] before filter:", len(_uniq), "urls:", list(_uniq.keys()))
    for _u, _row in _uniq.items():
        _s = _credibility_scorer.score_url(_u)
        print(f"  score={_s['score']:<3} level={_s['level']:<6} {_u}")
    _filtered = filter_results_by_credibility(_uniq)
    print("[credibility test] after filter:", len(_filtered), "urls:", list(_filtered.keys()))
    print()

    # Example brief
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

    result = researcher_agent.invoke({"researcher_messages": [HumanMessage(content=f"{research_brief}.")]})
    format_messages(result['researcher_messages'])