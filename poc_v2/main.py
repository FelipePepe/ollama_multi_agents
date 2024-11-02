# Import necessary modules
from crewai import Agent, Task, Crew
from langchain_ollama import ChatOllama
import os

# Set environment variables for API and LLM base URL
os.environ["OPENAI_API_KEY"] = ""  # Insert your OpenAI API key here
os.environ["LLM_BASE_URL"] = "http://localhost:11434"  # Local LLM endpoint

# Initialize the language model from OpenAI with the specified base URL and model
model = ChatOllama(
    model = "crewai-llama3",  # Example model; replace with your desired Ollama model
    temperature=0,
    base_url=os.environ["LLM_BASE_URL"]
)

# Define the agent's role, goal, backstory, and settings
general_agent = Agent(
    role="Math Professor",
    goal="""Provide solutions to students asking mathematical questions, 
            explaining answers clearly and understandably.""",
    backstory="""You are a skilled math professor who enjoys explaining math 
                 problems in a way that students can easily follow.""",
    allow_delegation=False,
    verbose=False,
    llm=model
)

# Define a task for the agent to solve
task = Task(
    description="What is 83 * 32 * 45?",
    agent=general_agent,
    expected_output="A numerical answer."
)

# Initialize the crew with the agent and task, setting verbosity
crew = Crew(
    agents=[general_agent],
    tasks=[task],
    verbose=True
)

# Execute the crew's tasks and print the result
result = crew.kickoff()
print(result)