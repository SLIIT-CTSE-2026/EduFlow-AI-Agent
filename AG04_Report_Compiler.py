from crewai import Agent, Task
from crewai.tools import BaseTool
from config import LOCAL_LLM, TOPIC, STATE_AG02, STATE_AG03, FINAL_REPORT

class ReportCompilerTool(BaseTool):
    name: str = "report_compiler_tool"
    description: str = f"Writes the final report to {FINAL_REPORT}."

    def _run(self, report_content: str) -> str:
        with open(FINAL_REPORT, "w", encoding="utf-8") as f:
            f.write(report_content)
        return f"Report created: {FINAL_REPORT}"

agent = Agent(
    role="Coordinator",
    goal="Compile the summary and quiz into a clean Markdown document.",
    backstory="You format final academic reports for students.",
    tools=[ReportCompilerTool()],
    llm=LOCAL_LLM,
    verbose=True
)

if __name__ == "__main__":
    with open(STATE_AG02, "r") as f:
        summary = f.read().strip()
    with open(STATE_AG03, "r") as f:
        quiz = f.read().strip()

    combined_state = f"# {TOPIC} Study Report\n\n## Summary\n{summary}\n\n## Quiz\n{quiz}"

    task = Task(
        description="Format the combined summary and quiz into a Markdown report.",
        expected_output="Confirmation of report creation.",
        agent=agent
    )
    
    result = agent.execute_task(task, context=combined_state)
    print(f"\nSUCCESS: {FINAL_REPORT} generated.")