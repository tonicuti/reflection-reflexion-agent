from dotenv import load_dotenv

load_dotenv()  

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph, MessagesState

from chains import generate_chain, reflection_chain

REFLECT = "reflect"
GENERATE = "generate"

def generation_node(state: MessagesState) -> dict:
    """Generates a tweet based on the user's request."""
    response = generate_chain.invoke({"messages": state["messages"]})
    return {"messages": [response]}

def reflection_node(state: MessagesState) -> dict:
    """Reflects on the generated tweet and provides critique and recommendations."""
    response = reflection_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=response.content)]}

builder = StateGraph(MessagesState)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)

builder.set_entry_point(GENERATE)

def should_continue(state: MessagesState):
    if len(state["messages"]) > 6:
        return END
    return REFLECT 

builder.add_conditional_edges(GENERATE, should_continue, {
    END: END, 
    REFLECT: REFLECT
})
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()

print(graph.get_graph().draw_mermaid())

if __name__ == "__main__":
    inputs = HumanMessage(content = """Make this tweet better:
                                        @LangChainAI
            - newly Tool Calling feature is seriously undderated. 
            After a long wait, it's here- making the implementation of Agents across different models with function calling-super easy.
            Made a video covering their newest blog post.
                          """)
    
    response = graph.invoke({"messages": [inputs]})
