from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.tools import TavilySearchResults
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()

class BasicChatBotState(TypedDict):
    messages: Annotated[list, add_messages]
    
search_tool = TavilySearchResults(max_results=2)
tools = [search_tool]

llm = ChatGroq(
    model = "openai/gpt-oss-120b",
    temperature=0,
)

llm_with_tools = llm.bind_tools(tools=tools)

def chatbot_with_tools(state: BasicChatBotState):
    return {
        "messages": [llm_with_tools.invoke(state["messages"])]
    }
    
def tools_router(state: BasicChatBotState):
    last_message = state["messages"][-1]
    
    if (hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0):
        return "tool_node"
    else:
        return END
    
tool_node = ToolNode(tools=tools, messages_key="messages")

graph = StateGraph(BasicChatBotState)
graph.add_node("chatbot_with_tools", chatbot_with_tools)
graph.add_node("tool_node", tool_node)

graph.set_entry_point("chatbot_with_tools")
graph.add_conditional_edges("chatbot_with_tools", tools_router)
graph.add_edge("tool_node", "chatbot_with_tools")

app = graph.compile()

while True:
    user_input = input("User: ")
    if(user_input in ["exit", "quit", "end"]):
        break
    else:
        result = app.invoke({
            "messages": [HumanMessage(content=user_input)]
        })
        print("AI:", result["messages"][-1].content)