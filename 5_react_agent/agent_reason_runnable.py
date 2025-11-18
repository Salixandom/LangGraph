from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
from langchain.agents import tool, create_react_agent
from langchain import hub
import datetime

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

searchTool = TavilySearchResults(
    search_depth="basic"
)

@tool
def get_sys_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """ Returns the current system time in the specified format. """
    
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

tools = [searchTool, get_sys_time]

react_prompt = hub.pull("hwchase17/react")

react_agent_runnable = create_react_agent(
    tools=tools, 
    llm=llm, 
    prompt=react_prompt
)