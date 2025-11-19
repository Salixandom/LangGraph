from typing import TypedDict, Annotated, List
from langgraph.graph import add_messages, StateGraph, END, START
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_community.tools import TavilySearchResults
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()
memory = MemorySaver()

class BasicState(TypedDict):
    messages: Annotated[list, add_messages]
    
search_tool = TavilySearchResults(max_results=2)
tools = [search_tool]

llm = ChatGroq(
    model = "openai/gpt-oss-120b",
    temperature=0,
)

llm_with_tools = llm.bind_tools(tools=tools)

def model(state: BasicState):
    return {
        "messages": [llm_with_tools.invoke(state["messages"])]
    }
    
def tools_router(state: BasicState):
    last_message = state["messages"][-1]
    
    if (hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0):
        return "tool_node"
    else:
        return END
    
tool_node = ToolNode(tools=tools, messages_key="messages")

graph = StateGraph(BasicState)
graph.add_node("model", model)
graph.add_node("tool_node", tool_node)

graph.set_entry_point("model")
graph.add_conditional_edges("model", tools_router)
graph.add_edge("tool_node", "model")

app = graph.compile(checkpointer=memory, interrupt_before=["tool_node"])

config = {"configurable": {"thread_id": 1}}

events = app.stream({
    "messages": [HumanMessage(content="What is the weather in Narsingdi Bd?")]
}, config=config, stream_mode="values")

for event in events:
    event["messages"][-1].pretty_print()
    
events = app.stream(None, config, stream_mode="values")

for event in events:
    event["messages"][-1].pretty_print()
    
