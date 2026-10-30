from dotenv import load_dotenv

load_dotenv()

from typing import Literal

from langchain_core.messages import AIMessage, ToolMessage 
from langgraph.graph import END, START, StateGraph, MessagesState

from chains import revisor, first_responder
from tool_executor import execute_tool

MAX_ITERATIONS = 2

def draft_node(state: MessagesState):
    """
    Draft the initial response.
    """
    result = first_responder.invoke({"messages": state["messages"]})
    return {
        "messages": [
            AIMessage(
                content=result.model_dump_json(),
                name="AnswerQuestion",
            )
        ]
    }

def revise_node(state: MessagesState):
    """
    Revise the response based on the new information.
    """
    result = revisor.invoke({"messages": state["messages"]})
    return {
        "messages": [
            AIMessage(
                content=result.model_dump_json(),
                name="ReviseAnswer",
            )
        ]
    }

def event_loop(state: MessagesState) -> Literal["execute_tool", END]:
    """
    Event loop to handle the conversation flow.
    """
    count_tool_visited = sum(
        isinstance(msg, ToolMessage) for msg in state["messages"]
    ) 
    
    num_iterations = count_tool_visited 
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tool"

builder = StateGraph(MessagesState)
builder.add_node("draft", draft_node)
builder.add_node("revise", revise_node)
builder.add_node("execute_tool", execute_tool)
builder.add_edge(START, "draft")
builder.add_edge("draft", "execute_tool")
builder.add_edge("execute_tool", "revise")
builder.add_conditional_edges("revise", event_loop, ["execute_tool", END])

graph = builder.compile()

print(graph.get_graph().draw_mermaid())

res = graph.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Write about AI-Powered SOC / Autonomous SOC problem domain, list startups that do that and raised capital."
            }
        ]
    }
)

last_message = res["messages"][-1]

if isinstance(last_message, AIMessage) and last_message.tool_calls:
    print(last_message.tool_calls[0]["args"]["answer"])
    
print(res)