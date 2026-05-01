from crewai import Crew, Process
from AG01_Document_Searcher import agent as curator, task as search_task
from AG02_Content_Summarizer import agent as analyst
from AG03_Quiz_Generator import agent as examiner
from AG04_Report_Compiler import agent as organizer
import subprocess

# Defining the Orchestrated Crew
edu_flow_crew = Crew(
    agents=[curator, analyst, examiner, organizer],
    tasks=[search_task],
    process=Process.sequential,
    verbose=True
)

if __name__ == "__main__":
    print(" STARTING DYNAMIC ORCHESTRATION ")
    subprocess.run(["python", "AG01_Document_Searcher.py"], check=True)
    subprocess.run(["python", "AG02_Content_Summarizer.py"], check=True)
    subprocess.run(["python", "AG03_Quiz_Generator.py"], check=True)
    subprocess.run(["python", "AG04_Report_Compiler.py"], check=True)
    print(" ORCHESTRATION COMPLETE ")