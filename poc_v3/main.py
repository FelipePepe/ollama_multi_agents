from crewai import Agent, Task, Crew
from langchain_ollama import ChatOllama
import os

# Set environment variables for API and LLM base URL
os.environ["OPENAI_API_KEY"] = ""  # Insert your OpenAI API key here
os.environ["LLM_BASE_URL"] = "http://localhost:11434"  # Local LLM endpoint

llmLlava = ChatOllama(
    model="llava:7b",
    temperature=0,
    base_url=os.environ["LLM_BASE_URL"]
)

llmLlama = ChatOllama(
    model="crewai-llama3",
    temperature=0,
    base_url=os.environ["LLM_BASE_URL"]
)

# 1. Image Classifier Agent (to check if the image is an animal)
classifier_agent = Agent(
    role="Image Classifier Agent",
    goal="Determine if the image is of an animal or not",
    backstory="You have an eye for animals! Your job is to identify whether the input image is of an animal or something else.",
    llm=llmLlava  # Model for image-related tasks
)

# 2. Animal Description Agent (to describe the animal in the image)
description_agent = Agent(
    role="Animal Description Agent",
    goal="Describe the animal in the image",
    backstory="You love nature and animals. Your task is to describe any animal based on an image.",
    llm=llmLlava  # Model for image-related tasks
)

# 3. Information Retrieval Agent (to fetch additional info about the animal)
info_agent = Agent(
    role="Information Agent",
    goal="Give compelling information about a certain animal",
    backstory="You are very good at telling interesting facts. You don't give any wrong information if you don't know it.",
    llm=llmLlama  # Model for general knowledge retrieval
)

# 4. Translate the information to Spanish.
translate_agent = Agent(
    role="Translate Agent",
    goal="Translate all information to Spanish",
    backstory="You are the best translator to Spanish.",
    llm=llmLlama
)

# Task 1: Check if the image is an animal
task1 = Task(
    description="Classify the image ({image_path}) and tell me if it's an animal.",
    expected_output="If it's an animal, say 'animal'; otherwise, say 'not an animal'.",
    agent=classifier_agent
)

# Task 2: If it's an animal, describe it
task2 = Task(
    description="Describe the animal in the image ({image_path}).",
    expected_output="Give a detailed description of the animal.",
    agent=description_agent
)

# Task 3: Provide more information about the animal
task3 = Task(
    description="Give additional information about the described animal.",
    expected_output="Provide at least 5 interesting facts or information about the animal.",
    agent=info_agent
)

# Task 4: Translate to Spanish
task4 = Task(
    description="Give all information about the described animal and translate to Spanish.",
    expected_output="All information translated to Spanish.",
    agent=translate_agent
)

# Crew to manage the agents and tasks
crew = Crew(
    agents=[classifier_agent, description_agent, info_agent, translate_agent],
    tasks=[task1, task2, task3, task4],
    verbose=True
)

# Prepare inputs for the tasks
inputs = {
    'image_path': 'racoon.jpeg',  # Ensure this path is correct and the image exists
}

# Execute the tasks with the provided image path
result = crew.kickoff(inputs=inputs)
print(result)
