"""
ranking_engine.py

This module provides the RankingEngine, which is responsible for the core
analytical logic of the system. Its tasks include:
1.  Calculating the similarity between the user's query and every text chunk.
2.  Aggregating chunk-level scores to a section-level score using a
    "Mean of Top-K" strategy to identify sections with a high density of
    relevant information.
3.  Sorting all sections from all documents to create a global ranking.
4.  Identifying the most relevant snippets from the top-ranked sections for
    the detailed sub-section analysis.
"""
import numpy as np
from typing import List, Tuple
from data_structures import Document, Section, Chunk

class RankingEngine:
    """
    Performs relevance scoring, aggregation, and ranking.
    """
    def __init__(self, top_k_chunks: int = 3, top_n_sections_for_analysis: int = 5):
        """
        Initializes the ranking engine with configurable parameters.

        Args:
            top_k_chunks (int): The number of top chunks within a section to average
                                for the section's final score.
            top_n_sections_for_analysis (int): The number of top-ranked sections
                                               to include in the detailed sub-section analysis.
        """
        self.top_k_chunks = top_k_chunks
        self.top_n_sections_for_analysis = top_n_sections_for_analysis
        print(f"Ranking Engine initialized. Aggregation: Mean of Top-{self.top_k_chunks}. Analysis: Top-{self.top_n_sections_for_analysis} sections.")

    def rank_sections_globally(self, all_docs: List[Document], query_embedding: np.ndarray) -> List[Section]:
        """
        Scores and ranks all sections from all documents against the query.

        Args:
            all_docs (List[Document]): A list of all parsed and embedded Document objects.
            query_embedding (np.ndarray): The user's query vector.

        Returns:
            List[Section]: A single, flat list of all Section objects, sorted by relevance.
        """
        print("Starting relevance ranking process...")
        all_sections = self._get_all_sections(all_docs)

        # Step 1: Score every chunk in every section
        self._score_all_chunks(all_sections, query_embedding)

        # Step 2: Aggregate chunk scores to get a score for each section
        self._aggregate_section_scores(all_sections)
        
        # Step 3: Sort sections based on their aggregated score
        # We filter out sections that have no score (e.g., were empty)
        scored_sections = [s for s in all_sections if s.aggregated_score is not None]
        sorted_sections = sorted(scored_sections, key=lambda s: s.aggregated_score, reverse=True)

        # Step 4: Assign the final importance rank
        for i, section in enumerate(sorted_sections):
            section.rank = i + 1
        
        print(f"Global ranking complete. {len(sorted_sections)} sections ranked.")
        return sorted_sections

    def perform_sub_section_analysis(self, ranked_sections: List[Section]) -> List[dict]:
        """
        Identifies the most relevant text snippets from the top-ranked sections.

        Args:
            ranked_sections (List[Section]): The globally ranked list of sections.

        Returns:
            List[dict]: A list of dictionaries, each containing details of a top
                        section and its most relevant text snippet(s).
        """
        print("Performing sub-section analysis on top sections...")
        analysis_results = []
        
        # Consider only the top N sections for this analysis
        sections_for_analysis = ranked_sections[:self.top_n_sections_for_analysis]

        for section in sections_for_analysis:
            if not section.chunks:
                continue

            # Sort the chunks within this section by their individual similarity scores
            sorted_chunks = sorted(section.chunks, key=lambda c: c.similarity_score, reverse=True)
            
            # The most relevant snippet is the text of the highest-scoring chunk
            top_chunk = sorted_chunks[0]

            analysis_results.append({
                "source_document": section.source_doc_name,
                "section_title": section.title,
                "importance_rank": section.rank,
                "refined_text_snippet": top_chunk.text,
                "snippet_relevance_score": round(top_chunk.similarity_score, 4)
            })
            
        return analysis_results

    def _get_all_sections(self, all_docs: List[Document]) -> List[Section]:
        """Flattens the list of documents into a single list of sections."""
        return [section for doc in all_docs for section in doc.sections]

    def _score_all_chunks(self, all_sections: List[Section], query_embedding: np.ndarray):
        """Calculates cosine similarity for every chunk and updates it in-place."""
        for section in all_sections:
            for chunk in section.chunks:
                # Cosine similarity is the dot product of normalized vectors
                score = np.dot(chunk.embedding, query_embedding)
                chunk.similarity_score = float(score)

    def _aggregate_section_scores(self, all_sections: List[Section]):
        """Calculates the final score for each section using the Mean of Top-K strategy."""
        for section in all_sections:
            if not section.chunks:
                section.aggregated_score = 0.0
                continue
            
            chunk_scores = [chunk.similarity_score for chunk in section.chunks]
            
            # Get the top k scores, but don't fail if there are fewer than k chunks
            num_chunks_to_consider = min(self.top_k_chunks, len(chunk_scores))
            top_k_scores = sorted(chunk_scores, reverse=True)[:num_chunks_to_consider]
            
            # Calculate the mean of these top scores
            if top_k_scores:
                section.aggregated_score = np.mean(top_k_scores)
            else:
                section.aggregated_score = 0.0
