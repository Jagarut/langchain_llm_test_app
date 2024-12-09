import os
import streamlit as st
import langchain
from langchain.agents import initialize_agent, load_tools, AgentType
from langchain_community.llms.sambanova import SambaNovaCloud
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
# Load environment variables from a .env file (if needed)
# load_dotenv()
def generate_pet_name(animal_type, pet_color):
    """
    Generate pet names based on the animal type and color.
    Args:
    animal_type (str): The type of animal (e.g., 'cat', 'dog').
    pet_color (str): The color of the pet (e.g., 'white', 'black').
    Returns:
    str: A string containing 5 suggested pet names.
    """
    # Load the SambaNova API key from the secrets.toml file
    sambanova_api_key = st.secrets["SAMBANOVA_API_KEY"]
    # Alternatively, load from environment variables
    # sambanova_api_key = os.getenv("SAMBANOVA_API_KEY")
    # Create a prompt template for generating pet names
    prompt_template = PromptTemplate(
        input_variables=["animal_type", "pet_color"],
        template="I have a {animal_type} pet and I want a cool name for it, it is {pet_color} color. Suggest me 5 cool names for my pet. I just want the names and nothing else.",
    )

    # Initialize the SambaNovaCloud language model
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
    
    # Define the RunnableSequence for generating pet names
    name_chain = prompt_template | llm
    # Use the chain to generate pet names
    response = name_chain.invoke({"animal_type": animal_type, "pet_color": pet_color})
    return response
 
 
def langchain_agent():
    """
    Initialize and run a LangChain agent to perform a specific task.
    This function sets up a LangChain agent with tools like Wikipedia and LLM-math,
    and runs a query to find the average age of a dog, multiply it by 3, and return the result.
    """
    # Load the SambaNova API key from the secrets.toml file
    sambanova_api_key = st.secrets["SAMBANOVA_API_KEY"]
    
         # Initialize the SambaNovaCloud language model
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
    
    # Load tools for the agent (Wikipedia and LLM-math)
    tools = load_tools(["wikipedia", "llm-math"], llm=llm)
    
    # Initialize the LangChain agent with the specified tools and language model -DEPRECATED-
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    
    
    # Run the agent with a specific query
    result = agent.invoke(
        "What is the average age of a dog? Multiply the age by 3. Respond only with the result as a number."
    )
    
if __name__ == "__main__":
    # Uncomment the following line to test the generate_pet_name function
    # print(generate_pet_name("cat", "white"))
    
    # Run the LangChain agent
    langchain_agent()