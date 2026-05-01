from crewai import Agent, Task
from crewai.tools import BaseTool
from config import LOCAL_LLM, STATE_AG02, STATE_AG03

class QuizSaverTool(BaseTool):
    name: str = "quiz_saver_tool"
    description: str = f"Saves quiz text to {STATE_AG03}."

    def _run(self, content: str) -> str:
        with open(STATE_AG03, "w", encoding="utf-8") as f:
            f.write(content)
        return "Quiz saved successfully."

agent = Agent(
    role="Professor",
    goal="Generate a 5-question MCQ quiz from the summary provided.",
    backstory="You create MCQs with options A-C and an answer key.",
    tools=[QuizSaverTool()],
    llm=LOCAL_LLM,
    verbose=True
)

if __name__ == "__main__":
    with open(STATE_AG02, "r") as f:
        summary = f.read().strip()
    
    task = Task(
        description=f"Create 5 MCQs based on this summary: {summary}. Save it to file.",
        expected_output="The full text of the quiz.",
        agent=agent
    )
    
    result = agent.execute_task(task)

    with open(STATE_AG03, "w", encoding="utf-8") as f:
        f.write(str(result))
    
    print(f"\nSTATE SAVED TO: {STATE_AG03}")