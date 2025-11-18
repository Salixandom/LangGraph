from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import datetime

from schema import AnswerQuestion, ReviseAnswer

load_dotenv()

actor_system_template = """You are an expert AI researcher.
                        Current time: {current_time}
                        1. {first_instruction}
                        2. Reflect and critque your answer. Be severe to maximize improvement.
                        3. After the reflection, **list 1-3 search queries separately** for researching improvements. Do not include them inside the reflection."""

actor_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(actor_system_template),
    MessagesPlaceholder(variable_name="actor_prompt_messages"),
    SystemMessagePromptTemplate.from_template("Answer the user's question above using the required format."),
]).partial(current_time = lambda: datetime.datetime.now().isoformat())

def as_ai_message(output):
    # Convert Pydantic model → JSON string → AIMessage
    # Most LLM adapters expect textual message content. Returning a
    # JSON string avoids sending raw dict/list parts which some
    # transports don't recognize (causes "Unrecognized massage part format").
    try:
        json_str = output.model_dump_json()
    except Exception:
        # Fallback: dump dict then stringify
        import json
        json_str = json.dumps(output.model_dump())

    return AIMessage(content=json_str)

# --------------------------- First Responder Chain -------------------------- #

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed answer (~100 words) to the user's question."
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

structured_llm_responder = llm.with_structured_output(AnswerQuestion)

first_responder_chain = first_responder_prompt_template | structured_llm_responder | as_ai_message

# ------------------------------- Revisor Chain ------------------------------ #

revise_instruction = """Revise your previous answer using the new information.
                        - You should use the previous critique to add important information to your answer.
                            - You MUST include numerical citations in your revised answer to ensure it can be verified.
                            - Add a "Reference" section to the bottom of your answer (which does not count towards the 250 word limit) 
                            listing the sources you used to revise your answer. In the form of:
                                - [1] https://example1.com
                                - [2] https://example2.com
                        - You should use the previous critique to remove any superfluous information from your answer and 
                        make SURE it is not more than 100 words."""

structured_llm_revisor = llm.with_structured_output(ReviseAnswer)

revisor_chain = actor_prompt_template.partial(first_instruction=revise_instruction) | structured_llm_revisor | as_ai_message


# response = first_responder_chain.invoke({
#     "actor_prompt_messages": [HumanMessage(content="What are the benefits and challenges of implementing renewable energy sources on a large scale?")]
# })

# print(response)