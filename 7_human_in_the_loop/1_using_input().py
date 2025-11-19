from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage
from langgraph.graph import add_messages, StateGraph, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]
    
llm = ChatGroq(
    model = "openai/gpt-oss-120b",
    temperature=0,
)

GENERATE_POST = "generate_post"
GET_REVIEW_DECISION = "get_review_decision"
PUBLISH_POST = "publish_post"
COLLECT_FEEDBACK = "collect_feedback"

def generate_post(state: State):
    return {
        "messages": [llm.invoke(state["messages"])]
    }
    
def get_review_decision(state: State):
    post_content = state["messages"][-1].content
    
    print("\nCurrent LinkedIn Post:\n")
    print(post_content)
    print("\n")
    
    decision = input("Post to LinkedIn? (yes/no): ").strip().lower()
    
    if decision == "yes":
        return PUBLISH_POST
    else:
        return COLLECT_FEEDBACK
    
def publish_post(state: State):
    final_post = state["messages"][-1].content
    print("\nPublishing the following post to LinkedIn:\n")
    print(final_post)
    print("\nPost published successfully!\n")
    
def collect_feedback(state: State):
    feedback = input("Please provide your feedback on the post: ")
    return {
        "messages": [HumanMessage(content=feedback)]
    }
    
graph = StateGraph(State)

graph.add_node(GENERATE_POST, generate_post)
graph.add_node(PUBLISH_POST, publish_post)
graph.add_node(COLLECT_FEEDBACK, collect_feedback)

graph.set_entry_point(GENERATE_POST)
graph.add_conditional_edges(GENERATE_POST, get_review_decision)
graph.add_edge(PUBLISH_POST, END)
graph.add_edge(COLLECT_FEEDBACK, GENERATE_POST)

app = graph.compile()

response = app.invoke({
    "messages": [HumanMessage(content="Create a LinkedIn post about the benefits of using AI in everyday work.")]
})