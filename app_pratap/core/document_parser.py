"""
document_parser.py

This module contains the DocumentParser class, responsible for ingesting a single
PDF file. It uses the PyMuPDF (fitz) library for its superior speed and features.

The core logic follows the "Structure-First" approach:
1.  It first attempts to extract the document's Table of Contents (ToC) to get an
    author-intended list of sections, their titles, and page numbers.
2.  If no ToC is available, it falls back to a page-by-page sectioning model,
    ensuring the system can handle any PDF.
3.  It then extracts the full text of the document and maps it to the
    identified sections.
"""

import fitz  # PyMuPDF
import os
from typing import List
from data_structures import Document, Section

class DocumentParser:
    """
    Parses a single PDF document to extract its structure and text content.
    """
    def __init__(self, pdf_path: str):
        """
        Initializes the parser with the path to a PDF file.

        Args:
            pdf_path (str): The full path to the PDF file.
        
        Raises:
            FileNotFoundError: If the pdf_path does not exist.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"No file found at {pdf_path}")
        self.pdf_path = pdf_path
        self.filename = os.path.basename(pdf_path)

    def parse(self) -> Document:
        """
        Executes the full parsing pipeline for the PDF.

        Returns:
            Document: A Document object populated with a list of Section objects.
        """
        print(f"[{self.filename}] Opening document...")
        doc = fitz.open(self.pdf_path)
        
        sections = self._extract_sections_from_toc(doc)
        if not sections:
            print(f"[{self.filename}] No ToC found. Falling back to page-based sectioning.")
            sections = self._extract_sections_by_page(doc)

        self._populate_sections_with_text(doc, sections)
        
        doc.close()
        print(f"[{self.filename}] Finished parsing. Found {len(sections)} sections.")
        return Document(filename=self.filename, sections=sections)

    def _extract_sections_from_toc(self, doc: fitz.Document) -> List[Section]:
        """
        Extracts section information using the PDF's Table of Contents.
        """
        toc = doc.get_toc()
        if not toc:
            return []

        sections = []
        for level, title, page_num in toc:
            # PyMuPDF page numbers are 1-based in get_toc(), which matches our needs.
            sections.append(
                Section(
                    title=title.strip(),
                    page_number=page_num,
                    source_doc_name=self.filename,
                    raw_text="" # Text will be populated in a later step
                )
            )
        return sections

    def _extract_sections_by_page(self, doc: fitz.Document) -> List[Section]:
        """
        Fallback method: creates a section for each page if no ToC is found.
        """
        sections = []
        for i in range(len(doc)):
            page_num = i + 1
            sections.append(
                Section(
                    title=f"Page {page_num}",
                    page_number=page_num,
                    source_doc_name=self.filename,
                    raw_text=""
                )
            )
        return sections

    def _populate_sections_with_text(self, doc: fitz.Document, sections: List[Section]):
        """
        Populates the raw_text field of each Section object by extracting text
        between the start of one section and the start of the next.
        """
        if not sections:
            return

        num_pages = len(doc)
        for i, section in enumerate(sections):
            # Page numbers from ToC are 1-based, fitz pages are 0-based
            start_page = section.page_number - 1
            
            # Determine the end page for the current section's text
            if i + 1 < len(sections):
                # The section ends right before the next section begins
                end_page = sections[i+1].page_number - 1
            else:
                # This is the last section, so it goes to the end of the document
                end_page = num_pages
            
            # Ensure page numbers are within the valid range
            start_page = max(0, start_page)
            end_page = min(end_page, num_pages)

            # Extract text from the relevant page range
            text_parts = []
            for page_num in range(start_page, end_page):
                page = doc.load_page(page_num)
                text_parts.append(page.get_text("text"))
            
            section.raw_text = "\n".join(text_parts).strip()

