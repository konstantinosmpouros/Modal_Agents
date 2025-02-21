# sudo apt install poppler-utils
from pdf2image import convert_from_path
from pathlib import Path
import pdfplumber
import os
import random

def extract_img_from_pdf(folder='knowledge_base'):
    pdf_folder = Path(folder)
    pdf_images = {}

    if not pdf_folder.exists():
        print(f'Error: The folder {folder} does not exist.')
        return {}

    for pdf_file in pdf_folder.glob("*.pdf"):
        # Convert PDF to a list of images (one per page)
        pages = convert_from_path(pdf_file, dpi=300)

        img_pages = []
        # Save each page as an image
        for _, page in enumerate(pages):
            img_pages.append(page)
        
        pdf_images[pdf_file.name] = img_pages
        
    return pdf_images


def read_pdfs(folder='knowledge_base'):
    pdf_folder = Path(folder)
    pdf_text = {}

    if not pdf_folder.exists():
        print(f'Error: The folder {folder} does not exist.')
        return {}

    for pdf_file in pdf_folder.glob("*.pdf"):
        with pdfplumber.open(pdf_file) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            pdf_text[pdf_file.name] = text

    return pdf_text


def get_random_pdf_as_string(knowledge_base_dir='knowledge_base'):
    # Check if the directory exists
    if not os.path.isdir(knowledge_base_dir):
        raise ValueError(f"The directory '{knowledge_base_dir}' does not exist.")
    
    # List all PDF files in the directory (case-insensitive)
    pdf_files = [file for file in os.listdir(knowledge_base_dir) if file.lower().endswith('.pdf')]
    
    if not pdf_files:
        raise ValueError("No PDF files found in the knowledge base directory.")
    
    # Randomly select a PDF file
    selected_pdf = random.choice(pdf_files)
    pdf_path = os.path.join(knowledge_base_dir, selected_pdf)
    
    # Extract text from the selected PDF file using pdfplumber
    extracted_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:  # Ensure that text was extracted successfully
                extracted_text += page_text + "\n"
    
    return extracted_text
