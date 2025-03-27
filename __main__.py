from pathlib import Path
import sys
import os

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

import random
from utils import read_pdfs
from Multi_Agents import CV_Analyzer


if __name__ == '__main__':
    # 1) Create your CV_Analyzer
    cv_multi_agent = CV_Analyzer()
    
    # 2) Load all PDFs from knowledge_base into a dictionary
    pdf_texts = read_pdfs('data/CV/knowledge_base')
    if not pdf_texts:
        print("No PDFs found or folder doesn't exist. Exiting.")
        exit(1)

    # 3) Main loop: present menu for random or specific PDF
    while True:
        print("\n=== PDF Analyzer Menu ===")
        print("1) Analyze a random CV")
        print("2) Analyze a specific CV by filename")
        print("3) Ctrl+C to exit.")
        
        try:
            choice = input("\nEnter your choice [1/2]: ").strip()
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        
        if choice not in ["1", "2"]:
            print("Invalid selection. Please try again.\n")
            continue
        
        # --- Identify which PDF we'll analyze ---
        if choice == "1":
            # Randomly pick from all keys in the dictionary
            selected_pdf = random.choice(list(pdf_texts.keys()))
            print(f"\nRandomly selected CV: {selected_pdf}")
        else:
            # Prompt the user for the exact filename
            selected_pdf = input("\nEnter exact PDF filename: ").strip()
            if selected_pdf not in pdf_texts:
                print("Invalid filename! Must be a PDF in the dictionary.")
                continue

        # 4) Get the text from the dictionary
        pdf_content = pdf_texts[selected_pdf]
        if not pdf_content.strip():
            print(f"Failed to extract content from {selected_pdf} (it may be empty).")
            continue

        # 5) Analyze and show results
        analysis_result = cv_multi_agent.analyze(pdf_content)  # adapt to your method
        print("\n=== Analysis Report ===")
        print(analysis_result)
        print("\n" + "=" * 40 + "\n")
