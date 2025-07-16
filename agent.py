from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from tools.search_tool import search_duckduckgo
from tools.calculator_tool import calculator
from utils import get_vision_response

def main():
    # Step 1: Provide your image path and prompt for vision
    image_path = "sample.jpg"
    vision_prompt = "Describe the image and answer questions about it."

    # Step 2: Get vision output text
    vision_text = get_vision_response(image_path, vision_prompt)
    print("🖼️ Vision Output:", vision_text)

    # Step 3: Setup tools
    tools = [
        Tool(name="Search", func=search_duckduckgo, description="Useful for web searches"),
        Tool(name="Calculator", func=calculator, description="Evaluates math expressions"),
    ]

    # Step 4: Initialize local LLM pipeline
    pipe = pipeline(
        "text-generation",
        model="google/flan-t5-base",
        max_length=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15,
        eos_token_id=1,
        pad_token_id=0,
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)

    # Step 5: Initialize agent with tools and LLM
    agent = initialize_agent(
        tools, local_llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

    # Step 6: Take user input question related to the image
    user_question = input("Ask a question about the image: ")

    # Step 7: Combine vision context + user question
    prompt = f"Image context: {vision_text}\nQuestion: {user_question}"

    # Step 8: Run the agent and get the response
    response = agent.run(prompt)
    print("🤖 Agent Response:", response)

if __name__ == "__main__":
    main()
