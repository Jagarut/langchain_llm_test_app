import os
import streamlit as st
import langchain
from langchain.agents import initialize_agent, load_tools, AgentType
from langchain_community.llms.sambanova import SambaNovaCloud
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

# load_dotenv()

def generate_pet_name(animal_type, pet_color):
    # Load the SambaNova API key from the secrets.toml file
    sambanova_api_key = st.secrets["SAMBANOVA_API_KEY"]
    # sambanova_api_key = os.getenv("SAMBANOVA_API_KEY")
    # Create a SambaNovaCloud instance
    prompt_template = PromptTemplate(
        input_variables=["animal_type", "pet_color"],
        template="I have a {animal_type} pet and I want a cool name for it, it is {pet_color} color. Suggest me 5 cool names for my pet. I just want the names and nothing else.",
    )
    
    llm = SambaNovaCloud(
        sambaverse_model_name="Meta-Llama-3.1-8B-Instruct",
        streaming=False,
        model_kwargs={
            "do_sample": True,
            "max_tokens_to_generate": 1000,
            "temperature": 0.7,
            "top_p": 1.0,
            "repetition_penalty": 1,
            "top_k": 50,
        },
    )
    # Define your RunnableSequence
    name_chain = prompt_template | llm
    # Use the chain
    response = name_chain.invoke({"animal_type": animal_type, "pet_color": pet_color})
    return response


def langchain_agent():
    sambanova_api_key = st.secrets["SAMBANOVA_API_KEY"]
    llm = SambaNovaCloud(
        sambaverse_model_name="Meta-Llama-3.1-8B-Instruct",
        streaming=False,
        model_kwargs={
            "do_sample": True,
            "max_tokens_to_generate": 1000,
            "temperature": 0.5,
            "top_p": 1.0,
            "repetition_penalty": 1,
            "top_k": 50,
        },
    )
    
    tools = load_tools(["wikipedia", "llm-math"], llm=llm)
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    result = agent.run(
        "What is the average age of a dog? Multiply the age by 3. Respond only with the result as a number."
    )
    print(result)

if __name__ == "__main__":
    # print(generate_pet_name("cat", "white"))
    langchain_agent()