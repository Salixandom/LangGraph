from typing import List

from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, MessageGraph

from chain import revisor_chain, first_responder_chain
from execute_tools import execute_tools

graph = MessageGraph()

RESPONDER = "responder"
REVISOR = "revisor"
EXECUTOR = "executor"
MAX_ITERATIONS = 1

graph.add_node(RESPONDER, first_responder_chain)
graph.add_node(REVISOR, revisor_chain)
graph.add_node(EXECUTOR, execute_tools)

def event_loop(state: List[BaseMessage]) -> str:
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    num_iterations = count_tool_visits
    if num_iterations > MAX_ITERATIONS:
        return END
    return EXECUTOR

graph.add_edge(RESPONDER, EXECUTOR)
graph.add_edge(EXECUTOR, REVISOR)
graph.add_conditional_edges(REVISOR, event_loop)

graph.set_entry_point(RESPONDER)

app = graph.compile()

response = app.invoke("Write about how small businesses can leverage social media marketing to grow their customer base.")
print(response[-1])