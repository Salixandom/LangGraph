import json
from typing import List
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_community.tools import TavilySearchResults

tavily_tool = TavilySearchResults(max_results=5)

def execute_tools(state: List[BaseMessage]) -> List[BaseMessage]:
    last_ai_msg = state[-1]

    # Ensure it's an AIMessage
    if not isinstance(last_ai_msg, AIMessage):
        return []

    # The structured result may be stored as a list/dict or as a JSON string.
    data = None

    # If it's already a list (e.g. content=[{...}])
    if last_ai_msg.content and isinstance(last_ai_msg.content, list):
        first = last_ai_msg.content[0]
        if isinstance(first, dict):
            data = first

    # If it's a dict directly
    if data is None and last_ai_msg.content and isinstance(last_ai_msg.content, dict):
        data = last_ai_msg.content

    # If it's a JSON string, try to parse
    if data is None and last_ai_msg.content and isinstance(last_ai_msg.content, str):
        try:
            import json
            parsed = json.loads(last_ai_msg.content)
            if isinstance(parsed, dict):
                data = parsed
            elif isinstance(parsed, list) and parsed:
                # maybe the JSON was a list containing the dict
                if isinstance(parsed[0], dict):
                    data = parsed[0]
        except Exception:
            return []

    if not data or not isinstance(data, dict):
        return []

    # Extract search queries
    search_queries = data.get("search_queries", [])
    if not search_queries:
        return []

    query_results = {}
    for query in search_queries:
        result = tavily_tool.invoke(query)
        query_results[query] = result

    # Return a ToolMessage with results
    return [
        ToolMessage(
            content=json.dumps(query_results),
            tool_call_id="search"
        )
    ]
