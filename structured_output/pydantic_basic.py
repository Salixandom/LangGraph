from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

class Country(BaseModel):
    """Information about a country."""
    
    name: str = Field(description="The name of the country")
    capital: str = Field(description="The capital city of the country")
    population: int = Field(description="The population of the country")
    languages: list[str] = Field(description="A list of official languages spoken in the country")

structured_llm = llm.with_structured_output(Country)

response = structured_llm.invoke("Provide information about Japan.")
print(response)