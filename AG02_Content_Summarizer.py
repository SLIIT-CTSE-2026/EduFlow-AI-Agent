import os
from crewai import Agent, Task
from crewai.tools import BaseTool
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from config import LOCAL_LLM, TOPIC, STATE_AG01, STATE_AG02

class ReadFileTool(BaseTool):
    name: str = "read_file_tool"
    description: str = "Extracts text from a provided local file path."

    def _run(self, file_path: str) -> str:
        file_path = file_path.strip().replace("'", "").replace('"', "")
        try:
            loader = PyPDFLoader(file_path) if file_path.endswith('.pdf') else TextLoader(file_path)
            docs = loader.load()
            return "\n".join([doc.page_content for doc in docs])[:3500] 
        except Exception as e:
            return f"Error reading file: {str(e)}"

agent = Agent(
    role="Teaching Assistant",
    goal=f"Summarize the {TOPIC} paper into 5 points. NO JSON.",
    backstory=f"You simplify complex concepts in {TOPIC}. Provide plain text only.",
    tools=[ReadFileTool()],
    max_iter=2,
    llm=LOCAL_LLM,
    verbose=True
)

if __name__ == "__main__":
    with open(STATE_AG01, "r") as f:
        path = f.read().strip()
    
    task = Task(
        description=f"Read the file at {path} and create a 5-point summary about {TOPIC}.",
        expected_output="A 5-point bulleted summary.",
        agent=agent
    )
    
    result = agent.execute_task(task)
    with open(STATE_AG02, "w", encoding="utf-8") as f:
        f.write(str(result))
    print(f"\nSTATE SAVED TO: {STATE_AG02}")