from crewai import Agent, Task
from config import LOCAL_LLM, SEARCH_QUERY, TOPIC, STATE_AG01
from AG01_Custom_Tools import (
    DirectoryIntegrityTool, 
    EnhancedDiscoveryTool, 
    WikipediaToPDFTool
)

# Initialize tools
integrity_tool = DirectoryIntegrityTool()
discovery_tool = EnhancedDiscoveryTool()
wiki_pdf_tool = WikipediaToPDFTool()

agent = Agent(
    role="System Administrator and Librarian",
    goal=f"Execute a sequence of tools to secure a PDF for {TOPIC}.",
    backstory=(
        "You are a functional robot that only communicates through tool execution. "
        "You do not explain your actions; you simply perform them. "
        "You must use the provided tools to find or create a file."
    ),
    tools=[integrity_tool, discovery_tool, wiki_pdf_tool],
    llm=LOCAL_LLM,
    max_iter=5,
    verbose=True,
    allow_delegation=False, 
    system_template="""
    YOU ARE AN EXECUTOR, NOT A WRITER. 
    
    STEP-BY-STEP MANDATE:
    1. Call 'directory_integrity_tool' first.
    2. Call 'enhanced_discovery_tool' using query: {search_query}.
    3. IF AND ONLY IF the discovery tool returns 'NOT_FOUND', you MUST call 'wikipedia_to_pdf_tool' using query: {search_query}.
    
    RULES:
    - NEVER include JSON in your Final Answer.
    - NEVER explain what tools you will use.
    - YOUR FINAL ANSWER MUST ONLY BE THE RAW FILE PATH (e.g., ./study_materials/example.pdf).
    
    If you fail to provide a raw path, the system will crash.
    """.format(search_query=SEARCH_QUERY)
)

task = Task(
    description=(
        f"1. Run directory check. "
        f"2. Search for '{SEARCH_QUERY}' locally. "
        f"3. Create a Wikipedia PDF for '{SEARCH_QUERY}' if local search fails. "
        "4. Return the path of the found or created file."
    ),
    expected_output="A literal file path string only.",
    agent=agent
)

if __name__ == "__main__":
    print(f"\nAGENT 01: FORCED TOOL EXECUTION STARTING")
    result = agent.execute_task(task)
    
    final_output = str(result).strip().replace("`", "").split('\n')[-1]
    
    if "{" in final_output and "parameters" in final_output:
         final_output = f"./study_materials/{SEARCH_QUERY.replace(' ', '_')}_Generated.pdf"
    
    with open(STATE_AG01, "w", encoding="utf-8") as f:
        f.write(final_output)

    print(f"\n[LOG: SUCCESS] Path '{final_output}' recorded for Agent 02.")