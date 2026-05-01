import os
from typing import Type
from pydantic import BaseModel, Field
from crewai import Agent, Task
from crewai.tools import BaseTool
from config import LOCAL_LLM, TARGET_PDF, SEARCH_QUERY, TOPIC, STATE_AG01

class FileSearchInput(BaseModel):
    search_query: str = Field(..., description="The keyword to search for in filenames.")

class FileDiscoveryTool(BaseTool):
    name: str = "local_file_discovery_tool"
    description: str = "Searches the './study_materials' folder for .pdf files."
    args_schema: Type[BaseModel] = FileSearchInput

    def _run(self, search_query: str) -> str:
        base_path = "./study_materials"
        if not os.path.exists(base_path): return "Error: Folder not found."
        files = [f for f in os.listdir(base_path) if search_query.lower() in f.lower()]
        return f"./study_materials/{files[0]}" if files else "No files found."

agent = Agent(
    role="Librarian",
    goal=f"Locate the research paper for {TOPIC} and STOP.",
    backstory="You are a precise librarian. Once you find a path, provide it as your Final Answer.",
    tools=[FileDiscoveryTool()],
    llm=LOCAL_LLM,
    max_iter=2,
    verbose=True
)

task = Task(
    description=f"Find the '{TARGET_PDF}' file using the search query '{SEARCH_QUERY}'.",
    expected_output="The exact file path string.",
    agent=agent
)

if __name__ == "__main__":
    print(f" AGENT 1: SEARCHING FOR {TARGET_PDF} ")
    result = agent.execute_task(task)
    with open(STATE_AG01, "w", encoding="utf-8") as f:
        f.write(str(result))
    print(f"\nSTATE SAVED TO: {STATE_AG01}")