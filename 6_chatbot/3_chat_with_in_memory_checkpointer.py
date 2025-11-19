from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

memory = MemorySaver()

llm = ChatGroq(
    model = "openai/gpt-oss-120b",
    temperature=0,
)

class BasicChatState(TypedDict):
    messages: Annotated[list, add_messages]
    
def ChatBot(state: BasicChatState):
    return {
        "messages": [llm.invoke(state["messages"])]
    }
    
graph = StateGraph(BasicChatState)

graph.add_node("chatbot", ChatBot)
graph.set_entry_point("chatbot")
graph.add_edge("chatbot", END)

app = graph.compile(checkpointer=memory)

config = {"configurable":{
    "thread_id": 1
}}

while True:
    user_input = input("User: ")
    if(user_input in ["exit", "quit", "end"]):
        break
    else:
        result = app.invoke({
            "messages": [HumanMessage(content=user_input)]
        }, config=config)
        print("AI:", result["messages"][-1].content)