response = first_responder_chain.invoke({
    "actor_prompt_messages": [HumanMessage(content="What are the benefits and challenges of implementing renewable energy sources on a large scale?")]
})

print(response)