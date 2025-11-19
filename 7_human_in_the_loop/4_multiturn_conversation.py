from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.types import Command, interrupt
from typing import TypedDict, Annotated, List
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import uuid
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model = "openai/gpt-oss-120b",
    temperature=0,
)

class State(TypedDict):
    linkedIn_topic: str
    generated_post: Annotated[List[str], add_messages]
    human_feedback: Annotated[List[str], add_messages]
    
def model(state: State):
    """Here we are using the LLM to generate a linkedIn post with human feedback incorporated"""
    
    print("[model] Generating Content")
    linkedIn_topic = state["linkedIn_topic"]
    feedback = state["human_feedback"] if "human_feedback" in state else ["No Feedback yet"]
    

    prompt = f"""
        linkedIn Topic: {linkedIn_topic}
        Human Feedback: {feedback[-1] if feedback else "No feedback yet"}
        
        Generate a structured and well-written linkedIn post based on the given topic
        Consider previous human feedback to refine the response.
        Don't give any extra text rather than the generated post
    """
    
    response = llm.invoke([
        SystemMessage(content="You are an expert linkedIn content writer"),
        HumanMessage(content=prompt)
    ])
    
    generated_linkedIn_post = response.content
    
    print(f"[model_node] Generated post: \n{generated_linkedIn_post}\n")
    
    return {
        "generated_post": [AIMessage(content=generated_linkedIn_post)],
        "human_feedback": feedback
    }
    
def human_node(state: State):
    """Human Intervention node - loops back to model unless input is done"""
    
    print("\n[human_node] awaiting human feedback...")
    
    generated_post = state["generated_post"]
    user_feedback = interrupt({
            "generated_post": generated_post,
            "message": "Provide feedback or type 'done' to finish"
    })
    
    print(f"[human_node] Received human feedback: {user_feedback}")
    
    if user_feedback.lower().strip() == "done":
        return Command(update={"human_feedback": state["human_feedback"] + ["Finalised"]}, goto="end_node")
    
    return Command(update={"human_feedback": state["human_feedback"] + [user_feedback]}, goto="model")

def end_node(state: State):
    print("\n[end_node] Process finished")
    print("Final Generated Post: \n", state["generated_post"][-1].content)
    print("Final Human Feedback: \n", state["human_feedback"][-1].content)
    
    return {
        "generated_post": state["generated_post"],
        "human_feedback": state["human_feedback"]
    }

graph = StateGraph(State)

graph.add_node("model", model)
graph.add_node("human_node", human_node)
graph.add_node("end_node", end_node)

graph.add_edge(START, "model")
graph.add_edge("model", "human_node")

graph.set_finish_point("end_node")

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

thread_config = {"configurable": {"thread_id": uuid.uuid4()}}

linkedIn_topic = input("Enter your linkedIn topic: ")
initial_state = {
    "linkedIn_topic": linkedIn_topic,
    "generated_post": [],
    "human_feedback": []
}

for chunk in app.stream(initial_state, config=thread_config):
    for node_id, value in chunk.items():
        if node_id == "__interrupt__":
            while True:
                user_feedback = input("Provide feedback (or type 'done' to finish): ")
                
                app.invoke(Command(resume=user_feedback), config=thread_config)
                
                if user_feedback.lower().strip() == "done":
                    break