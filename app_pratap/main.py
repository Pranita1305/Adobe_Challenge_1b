"""
main.py

The main execution script for the intelligent document analyst system.
This script orchestrates the entire pipeline from input to output.

It performs the following steps:
1.  Parses command-line arguments for the PDF directory, persona, and job.
2.  Initializes the core singleton components (EmbeddingEngine).
3.  Formulates the intelligent query from the persona and job.
4.  Processes all PDFs in parallel to parse and embed their content.
5.  Uses the RankingEngine to score and rank all sections globally.
6.  Performs sub-section analysis on the top-ranked sections.
7.  Uses the OutputBuilder to generate the final JSON file.
8.  Measures and reports the total execution time.

Usage:
    python main.py \
        --pdfs_dir "path/to/your/pdfs" \
        --persona "A PhD Researcher in Computational Biology" \
        --job "Prepare a literature review on methodologies." \
        --output_path "results.json"
"""
import os
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

# Import core components and data structures
from data_structures import SystemInput, Document
from core.document_parser import DocumentParser
from core.embedding_engine import EmbeddingEngine
from core.ranking_engine import RankingEngine
from core.output_builder import OutputBuilder

def process_single_pdf(pdf_path: str, model_path: str) -> Document:
    """
    A top-level function designed to be run in a separate process.
    It parses a PDF, initializes its own embedding engine instance,
    and then chunks/embeds the content of the document.
    """
    print(f"[Process {os.getpid()}] Starting to process {os.path.basename(pdf_path)}")
    
    # 1. Parse the document structure and text
    parser = DocumentParser(pdf_path)
    document = parser.parse()

    # 2. Initialize the embedding engine within this process
    # The model is loaded once per process, which is efficient for parallelization.
    embedder = EmbeddingEngine(model_path=model_path)

    # 3. Chunk and embed each section
    for section in document.sections:
        embedder.process_section(section)
        
    print(f"[Process {os.getpid()}] Finished processing {os.path.basename(pdf_path)}")
    return document

def main():
    """Main function to run the document analysis pipeline."""
    start_time = time.time()

    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(description="Intelligent Document Analyst for Adobe Hackathon")
    parser.add_argument("--pdfs_dir", type=str, required=True, help="Directory containing input PDF files.")
    parser.add_argument("--persona", type=str, required=True, help="The user persona.")
    parser.add_argument("--job", type=str, required=True, help="The user's job-to-be-done.")
    parser.add_argument("--model_path", type=str, default="models/thenlper/gte-small", help="Local path to the embedding model.")
    parser.add_argument("--output_path", type=str, default="output.json", help="Path for the final JSON output file.")
    args = parser.parse_args()

    pdf_paths = [os.path.join(args.pdfs_dir, f) for f in os.listdir(args.pdfs_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_paths:
        print(f"Error: No PDF files found in directory '{args.pdfs_dir}'")
        return

    system_input = SystemInput(
        pdf_paths=pdf_paths,
        persona=args.persona,
        job_to_be_done=args.job
    )
    print(f"Starting analysis for {len(pdf_paths)} documents...")

    # --- 2. Initialize Engines & Formulate Query ---
    # The main process initializes the engine to create the query embedding.
    # Each child process will create its own instance for document processing.
    embedder = EmbeddingEngine(model_path=args.model_path)
    ranker = RankingEngine()
    output_builder = OutputBuilder(system_input)

    query_text = f"Persona: {system_input.persona}. Task: {system_input.job_to_be_done}"
    query_embedding = embedder.embed_query(query_text)

    # --- 3. Concurrent Document Processing ---
    all_documents: List[Document] = []
    # Use ProcessPoolExecutor to leverage multiple CPU cores
    with ProcessPoolExecutor() as executor:
        # Submit all PDF processing tasks
        future_to_pdf = {executor.submit(process_single_pdf, pdf_path, args.model_path): pdf_path for pdf_path in pdf_paths}
        
        for future in as_completed(future_to_pdf):
            pdf_path = future_to_pdf[future]
            try:
                processed_document = future.result()
                all_documents.append(processed_document)
            except Exception as exc:
                print(f"'{os.path.basename(pdf_path)}' generated an exception: {exc}")

    # --- 4. Ranking and Analysis ---
    ranked_sections = ranker.rank_sections_globally(all_documents, query_embedding)
    sub_section_analysis = ranker.perform_sub_section_analysis(ranked_sections)

    # --- 5. Output Generation ---
    final_output = output_builder.build(ranked_sections, sub_section_analysis)
    output_builder.write_to_file(final_output, args.output_path)

    # --- 6. Report Performance ---
    end_time = time.time()
    print("\n" + "="*50)
    print("PIPELINE COMPLETE")
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")
    print(f"Output written to: {args.output_path}")
    print("="*50)

if __name__ == "__main__":
    main()

# python main.py --pdfs_dir "Challenge_1b/Collection_1/PDFs" --persona "Travel Planner" --job "Plan a trip of 4 days for a group of 10 college friends."