"""
embedding_engine.py

This module defines the EmbeddingEngine, a singleton class responsible for all
semantic operations. Its key responsibilities are:
1.  Loading the pre-trained sentence-transformer model ('thenlper/gte-small')
    from a local, offline directory.
2.  Chunking section text into semantically meaningful paragraphs with a
    sentence overlap to maintain context.
3.  Encoding text (both queries and document chunks) into dense vector embeddings.

The singleton pattern ensures that the heavyweight model is loaded into memory
only once per application run, saving significant time and resources.
"""
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
from data_structures import Section, Chunk

# NLTK is used for robust sentence splitting. It's a lightweight dependency.
# It's assumed the 'punkt' tokenizer is pre-downloaded.
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("NLTK 'punkt' tokenizer not found. Please download it first by running: \nimport nltk\nnltk.download('punkt')")
    # In a real offline environment, this would be pre-packaged.

class EmbeddingEngine:
    """
    A singleton class to manage the sentence embedding model and processes.
    """
    _instance = None
    
    # --- Singleton Pattern ---
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(EmbeddingEngine, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_path: str = "models/thenlper/gte-small"):
        """
        Initializes the engine. The constructor will only run the first time
        the object is created due to the singleton pattern.
        
        Args:
            model_path (str): The local file path to the pre-downloaded model.
        """
        # The check prevents re-initialization on subsequent calls
        if not hasattr(self, 'model'):
            print("Initializing Embedding Engine...")
            if not os.path.isdir(model_path):
                raise FileNotFoundError(
                    f"Model directory not found at '{model_path}'. "
                    "Please ensure the 'thenlper/gte-small' model is downloaded there."
                )
            
            # Load the model from the local directory for offline use
            self.model = SentenceTransformer(model_path)
            print("Embedding model loaded successfully.")

    def embed_query(self, query_text: str) -> np.ndarray:
        """
        Creates a single vector embedding for the user query.
        
        Args:
            query_text (str): The combined persona and job-to-be-done string.
            
        Returns:
            np.ndarray: A 1D numpy array representing the query embedding.
        """
        print("Embedding user query...")
        # normalize_embeddings=True is recommended for cosine similarity
        embedding = self.model.encode(query_text, normalize_embeddings=True)
        return embedding

    def process_section(self, section: Section):
        """
        Chunks and embeds the text within a single Section object.
        This method updates the section's `chunks` list in-place.
        """
        print(f"[{section.source_doc_name}] Chunking section: '{section.title}'")
        chunks_text = self._chunk_text_with_overlap(section.raw_text)
        
        if not chunks_text:
            section.chunks = []
            return

        print(f"[{section.source_doc_name}] Found {len(chunks_text)} chunks. Generating embeddings...")
        embeddings = self.model.encode(chunks_text, normalize_embeddings=True, show_progress_bar=False)
        
        chunks = []
        for text, embedding in zip(chunks_text, embeddings):
            chunks.append(
                Chunk(
                    text=text,
                    embedding=embedding,
                    source_doc_name=section.source_doc_name,
                    source_section_title=section.title
                )
            )
        section.chunks = chunks

    def _chunk_text_with_overlap(self, text: str) -> List[str]:
        """
        Implements the paragraph-based chunking with single-sentence overlap.
        
        Args:
            text (str): The full raw text of a section.
            
        Returns:
            List[str]: A list of text chunks ready for embedding.
        """
        if not text:
            return []

        # Split the text into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if not paragraphs:
            return []

        chunks = []
        for i, para in enumerate(paragraphs):
            chunk = para
            # If this is not the last paragraph, find the first sentence of the next one
            if i < len(paragraphs) - 1:
                next_para = paragraphs[i+1]
                # Use NLTK for robust sentence splitting
                next_para_sentences = nltk.sent_tokenize(next_para)
                if next_para_sentences:
                    # Append the first sentence of the next paragraph as overlap
                    overlap_sentence = next_para_sentences[0]
                    chunk += " " + overlap_sentence
            
            chunks.append(chunk)
            
        return chunks

