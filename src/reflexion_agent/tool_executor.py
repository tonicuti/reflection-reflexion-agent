from dotenv import load_dotenv

load_dotenv()

from langchain_tavily import TavilySearch
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode

from schemas import AnswerQuestion, ReviseAnswer

tavily_tool = TavilySearch(max_results=3)


def run_queries(search_queries: list[str], **kwargs):
    """Run search queries and return the results."""
    return tavily_tool.batch([{"query": q} for q in search_queries])

execute_tool = ToolNode(
    [
        StructuredTool.from_function(
            run_queries,
            name="AnswerQuestion",
        ),
        StructuredTool.from_function(
            run_queries,
            name="ReviseAnswer",
        ),
    ]
)