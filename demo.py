from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Create an instance of ChatOpenAI. 
# It automatically uses the OPENAI_API_KEY environment variable.
chat = ChatOpenAI(model="gpt-4o-mini") # You can specify other models if needed

# Send a message to the AI model using the invoke method
response = chat.invoke("Hello, how are you?")

# Display the AI's response content
print("AI Response:")
print(response.content)
