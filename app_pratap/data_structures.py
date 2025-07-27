"""
data_structures.py

This module defines the core data classes used throughout the NLP system.
These structures ensure a consistent and typed representation of documents,
sections, and chunks as they are processed by the pipeline. Using dataclasses
provides type hints, readability, and basic object-oriented features without
the boilerplate of full class definitions.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

@dataclass
class Chunk:
    """
    Represents a single text chunk, which is the smallest unit for semantic analysis.
    
    Attributes:
        text (str): The raw text content of the chunk.
        embedding (np.ndarray): The dense vector embedding of the text. This is excluded
                                from the default representation for readability.
        source_doc_name (str): The filename of the document this chunk originated from.
        source_section_title (str): The title of the section this chunk belongs to.
        similarity_score (Optional[float]): The cosine similarity score of this chunk
                                            against the user query. This is populated
                                            during the ranking phase.
    """
    text: str
    source_doc_name: str
    source_section_title: str
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    similarity_score: Optional[float] = None

@dataclass
class Section:
    """
    Represents a logical section of a document, as identified by the Table of Contents
    or other parsing logic. It contains a list of its constituent text chunks.
    
    Attributes:
        title (str): The title of the section (e.g., "Introduction", "Methodology").
        page_number (int): The page number where the section begins.
        source_doc_name (str): The filename of the document this section belongs to.
        raw_text (str): The full, unprocessed text content of the entire section.
        chunks (List[Chunk]): A list of Chunk objects derived from this section's text.
        aggregated_score (Optional[float]): The final relevance score for the section,
                                            calculated by aggregating the scores of its
                                            most relevant chunks.
        rank (Optional[int]): The final importance rank of this section across all documents.
    """
    title: str
    page_number: int
    source_doc_name: str
    raw_text: str
    chunks: List[Chunk] = field(default_factory=list)
    aggregated_score: Optional[float] = None
    rank: Optional[int] = None

@dataclass
class Document:
    """
    Represents a single input PDF document.
    
    Attributes:
        filename (str): The original filename of the PDF.
        sections (List[Section]): A list of all sections identified within the document.
    """
    filename: str
    sections: List[Section] = field(default_factory=list)

@dataclass
class SystemInput:
    """
    A container for the initial inputs to the system, providing a clean
    way to pass around the core query parameters.
    
    Attributes:
        pdf_paths (List[str]): A list of file paths to the input PDFs.
        persona (str): The user persona string.
        job_to_be_done (str): The job-to-be-done string.
    """
    pdf_paths: List[str]
    persona: str
    job_to_be_done: str

@dataclass
class FinalOutput:
    """
    Represents the final, structured output that will be serialized to JSON.
    This class mirrors the required JSON format exactly.
    """
    metadata: dict
    extracted_sections: List[dict]
    sub_section_analysis: List[dict]

