from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

generation_system_prompt = """You are a twitter techie influencer assistant tasked with writing excellent twitter posts.
                            Generate the best twitter post possible for the user's request. Don't generate extra words. Generate the tweet only.
                            If the user provides critique, respond with a revised version of your previous attempts."""

generation_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(generation_system_prompt),
    MessagesPlaceholder(variable_name="history"),
])

reflection_system_prompt = """You are a viral twitter influencer tasked with grading a tweet. Generate critique and recommendations for the user's tweet.
                            Always provide detailed recommendations, including requests for length, virality, style and
                            specific changes to wording, hashtags, and structure to improve engagement."""

reflection_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(reflection_system_prompt),
    MessagesPlaceholder(variable_name="history"),
])

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm