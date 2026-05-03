import os
import wikipedia
from fpdf import FPDF 
from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from config import STUDY_MATERIALS_PATH

class FileSearchInput(BaseModel):
    """Input schema for the discovery tool."""
    search_query: str = Field(..., description="Keyword to find in filenames.")

# DIRECTORY INTEGRITY
class DirectoryIntegrityTool(BaseTool):
    name: str = "directory_integrity_tool"
    description: str = "Checks if the study_materials folder exists."

    def _run(self) -> str:
        print("\n[LOG: PHASE 1] Initiating Directory Integrity Check...")
        if not os.path.exists(STUDY_MATERIALS_PATH):
            return f"CRITICAL ERROR: Folder '{STUDY_MATERIALS_PATH}' not found."
        print("[LOG: PHASE 1] Success: Folder exists.")
        return "SUCCESS"

# LOCAL PDF DISCOVERY
class EnhancedDiscoveryTool(BaseTool):
    name: str = "enhanced_discovery_tool"
    description: str = "Searches for a local PDF. Use a single keyword for best results."
    args_schema: Type[BaseModel] = FileSearchInput

    def _run(self, search_query: str) -> str:
        keyword = search_query.split()[0].lower().replace("_", " ")
        print(f"\n[LOG: PHASE 2] Searching for local PDF with keyword: '{keyword}'")
        
        try:
            # Scans for the keyword in any filename within the directory[cite: 26]
            files = [f for f in os.listdir(STUDY_MATERIALS_PATH) 
                     if keyword in f.lower() and f.endswith('.pdf')]
            
            if files:
                path = os.path.join(STUDY_MATERIALS_PATH, files[0])
                print(f"[LOG: PHASE 2] Success! Local PDF Found: {path}")
                return path
            
            print("[LOG: PHASE 2] Result: NOT_FOUND. Fallback required.")
            return "NOT_FOUND"
        except Exception as e:
            return f"Error: {str(e)}"

# --- TOOL 3: WIKIPEDIA TO PDF FALLBACK ---[cite: 26]

class WikipediaToPDFTool(BaseTool):
    name: str = "wikipedia_to_pdf_tool"
    description: str = "Creates a PDF from Wikipedia if local files are missing."

    def _run(self, query: str) -> str:
        print(f"\n[LOG: FALLBACK] Attempting Wikipedia fetch for: '{query}'")
        try:
            # auto_suggest helps handle disambiguation (multiple topics with same name)[cite: 26]
            page_title = wikipedia.suggest(query) or query
            content = wikipedia.summary(page_title, sentences=10, auto_suggest=True)
            
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt=f"Wiki Research: {page_title}", ln=True, align='C')
            pdf.ln(10)
            
            # Using 'replace' ensures we don't crash on special characters not in standard fonts[cite: 26]
            pdf.multi_cell(0, 10, txt=content.encode('latin-1', 'replace').decode('latin-1'))
            
            # Ensure the study_materials directory exists
            os.makedirs(STUDY_MATERIALS_PATH, exist_ok=True)
            file_name = f"{query.replace(' ', '_')}_Generated.pdf"
            file_path = os.path.join(STUDY_MATERIALS_PATH, file_name)
            pdf.output(file_path)
            
            print(f"[LOG: FALLBACK] Created new PDF material: {file_path}")
            return file_path
        except Exception as e:
            return f"Wikipedia-to-PDF process failed: {str(e)}"