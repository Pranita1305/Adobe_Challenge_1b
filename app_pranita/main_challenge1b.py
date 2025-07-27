import os
import json
import torch
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import multiprocessing
import re # Still needed for basic text cleaning, but not for complex keyword matching logic

# --- Import the necessary functions from ingest.py ---
try:
    from ingest import (
        extract_logical_text_blocks,
        engineer_layout_features,
        filter_header_footer_blocks,
        calculate_weighted_mean_font_size,
        filter_long_blocks,
        filter_small_fonts_by_weighted_mean,
        post_process_blocks,
        detect_sections,
        group_blocks_into_passages
    )
except ImportError:
    print("Error: Could not import from 'ingest.py'.")
    print("Please ensure 'ingest.py' is in the same directory as this script.")
    exit()

from transformers import AutoTokenizer, AutoModel

# ==============================================================================
# --- Configuration ---
# ==============================================================================
PROJECT_ROOT = Path(__file__).parent.resolve()

INPUT_JSON_PATH = PROJECT_ROOT / "input" / "Collection1" / "challenge1b_input.json"
PDF_BASE_DIR = PROJECT_ROOT / "input" / "Collection1" / "pdfs"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_PATH = PROJECT_ROOT / "transformers_models" / MODEL_NAME.replace("/", "_")

OUTPUT_JSON_DIR = PROJECT_ROOT / "output" / "json"
OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)

# Global variables for model and tokenizer to be initialized once per process
_embedding_model = None
_embedding_tokenizer = None

# ==============================================================================
# --- Constants for Output Formatting ---
# ==============================================================================
MAX_DISPLAY_CHAR_LENGTH = 700 # Max characters for refined_text in output. Longer passages will be truncated at sentence boundary.

# ==============================================================================
# --- Embedding Generation ---
# ==============================================================================

def load_embedding_model_for_process(model_path: Path):
    """Loads the tokenizer and model for generating embeddings, specifically for multiprocessing."""
    global _embedding_model, _embedding_tokenizer
    if _embedding_model is None:
        print(f"[{os.getpid()}] Loading embedding tokenizer and model from '{model_path}'...")
        _embedding_tokenizer = AutoTokenizer.from_pretrained(model_path)
        _embedding_model = AutoModel.from_pretrained(model_path)
        _embedding_model.eval() # Set model to evaluation mode
        _embedding_model.to(torch.device("cpu")) # Keep on CPU initially
        print(f"[{os.getpid()}] ✅ Embedding Model and tokenizer loaded.")
    return _embedding_model, _embedding_tokenizer

def get_sentence_embedding_in_process(text: str) -> np.ndarray:
    """Generates a semantic embedding for a single text string using mean pooling, using pre-loaded model."""
    global _embedding_model, _embedding_tokenizer
    if _embedding_model is None or _embedding_tokenizer is None:
        raise RuntimeError("Embedding model not loaded in this process. Call load_embedding_model_for_process() first.")

    if not text:
        return np.array([])

    text = str(text)
    inputs = _embedding_tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    _embedding_model.to(device) # Ensure model is on the correct device for each inference

    with torch.no_grad():
        outputs = _embedding_model(**inputs)

    last_hidden_states = outputs.last_hidden_state
    attention_mask = inputs['attention_mask']

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
    sum_embeddings = torch.sum(last_hidden_states * mask, 1)
    sum_mask = torch.clamp(mask.sum(1), min=1e-9)

    embeddings = sum_embeddings / sum_mask

    return embeddings.squeeze().cpu().numpy()

# ==============================================================================
# --- Sentence Completion Utility ---
# ==============================================================================

def truncate_to_complete_sentence(text: str, max_chars: int) -> str:
    """
    Truncates text to a maximum character length, ensuring it ends on a complete sentence.
    If the text is too short or has no sentence endings, it returns the original.
    """
    if len(text) <= max_chars:
        return text

    # Find last sentence-ending punctuation before or at max_chars
    match = re.search(r'([.!?]\s*)$', text[:max_chars]) # Look for punctuation at the end of the truncated part
    if match:
        return text[:max_chars]

    # If no punctuation at the very end, try to find the last punctuation before max_chars
    # and include the sentence ending after it.
    last_punctuation_match = None
    for m in re.finditer(r'[.!?](?:\s+|$)', text): # Find all sentence endings
        if m.start() < max_chars:
            last_punctuation_match = m
        else:
            break # No more punctuation before max_chars

    if last_punctuation_match:
        # Include the punctuation and any whitespace after it
        return text[:last_punctuation_match.end()].strip()
    else:
        # If no punctuation found, or it's just one long sentence without breaks
        # Return truncated text with "..."
        return text[:max_chars].strip() + "..."


# ==============================================================================
# --- Worker Function for Parallel Processing ---
# ==============================================================================

def process_single_pdf_task(pdf_filename: str, persona_embedding_np: np.ndarray) -> Dict[str, Any]:
    """
    Processes a single PDF for Challenge 1B: extracts sections, matches
    against a persona, and returns relevant sections.
    This function runs in a separate process.
    """
    load_embedding_model_for_process(EMBEDDING_MODEL_PATH)
    
    persona_embedding_tensor = torch.from_numpy(persona_embedding_np).float().cpu()

    pdf_path = PDF_BASE_DIR / pdf_filename
    
    print(f"[{os.getpid()}] Starting processing for PDF: {pdf_filename}")

    current_doc_sections_ranked = []
    current_doc_subsections_ranked = []

    if not pdf_path.is_file():
        print(f"[{os.getpid()}] Error: PDF file not found for {pdf_filename} at {pdf_path}. Skipping.")
        return {"extracted_sections": [], "subsection_analysis": [], "pdf_filename": pdf_filename}

    # --- PDF Text Extraction and Feature Engineering ---
    try:
        raw_blocks, page_count = extract_logical_text_blocks(str(pdf_path))
        print(f"[{os.getpid()}] Extracted {len(raw_blocks)} raw blocks from {pdf_filename}.")
        if not raw_blocks:
            print(f"[{os.getpid()}] Warning: No text blocks extracted from {pdf_filename}. Skipping.")
            return {"extracted_sections": [], "subsection_analysis": [], "pdf_filename": pdf_filename}

        non_header_footer_blocks = filter_header_footer_blocks(raw_blocks, page_count)
        print(f"[{os.getpid()}] After header/footer filter: {len(non_header_footer_blocks)} blocks.")
        
        weighted_font_threshold = calculate_weighted_mean_font_size(non_header_footer_blocks)
        
        short_blocks = filter_long_blocks(non_header_footer_blocks, weighted_font_threshold)
        print(f"[{os.getpid()}] After long block filter: {len(short_blocks)} blocks.")

        font_filtered_blocks = filter_small_fonts_by_weighted_mean(short_blocks, weighted_font_threshold)
        print(f"[{os.getpid()}] After small font filter: {len(font_filtered_blocks)} blocks.")

        final_blocks_unfeatured = post_process_blocks(font_filtered_blocks)
        print(f"[{os.getpid()}] After post-processing: {len(final_blocks_unfeatured)} blocks.")

        featured_blocks = engineer_layout_features(final_blocks_unfeatured)
        print(f"[{os.getpid()}] Engineered features for {len(featured_blocks)} blocks.")

        if not featured_blocks:
            print(f"[{os.getpid()}] Warning: No candidate blocks remained after filtering for {pdf_filename}. Skipping.")
            return {"extracted_sections": [], "subsection_analysis": [], "pdf_filename": pdf_filename}

    except Exception as e:
        print(f"[{os.getpid()}] Error during text extraction or feature engineering for {pdf_filename}: {e}")
        import traceback
        traceback.print_exc()
        return {"extracted_sections": [], "subsection_analysis": [], "pdf_filename": pdf_filename}

    # --- Detect Sections (for 'extracted_sections' output) ---
    document_outline = detect_sections(featured_blocks)
    print(f"[{os.getpid()}] Detected {len(document_outline.get('outline', []))} main outline sections for {pdf_filename}.")
    
    # --- Semantic Search and Ranking for 'extracted_sections' (High-level sections) ---
    for item in document_outline.get("outline", []):
        section_text = item.get("text", "")
        if len(section_text.strip().split()) >= 3:
            section_embedding = get_sentence_embedding_in_process(section_text)
            
            if section_embedding.size > 0:
                semantic_similarity = cosine_similarity(persona_embedding_np.reshape(1, -1), section_embedding.reshape(1, -1))[0][0]
                
                # Use semantic_similarity directly for sorting
                current_doc_sections_ranked.append({
                    "document": pdf_filename,
                    "section_title": section_text,
                    "level": item.get("level"),
                    "page_number": item.get("page"),
                    "similarity_score": float(semantic_similarity) # Only semantic score
                })
    print(f"[{os.getpid()}] Found {len(current_doc_sections_ranked)} high-level sections for ranking from {pdf_filename}.")


    # --- Group Blocks into Passages for 'subsection_analysis' ---
    passages_from_doc = group_blocks_into_passages(featured_blocks)
    print(f"[{os.getpid()}] Grouped {len(featured_blocks)} blocks into {len(passages_from_doc)} passages for {pdf_filename}.")

    # --- Semantic Search and Ranking for Passages (for 'subsection_analysis' output) ---
    min_words_for_passage = 20
    min_chars_for_passage = 100
    
    for passage in passages_from_doc:
        passage_text = passage.get("text", "")
        if passage.get('word_count', 0) >= min_words_for_passage and \
           len(passage_text.strip()) >= min_chars_for_passage:
            
            passage_embedding = get_sentence_embedding_in_process(passage_text)
            
            if passage_embedding.size > 0:
                semantic_similarity = cosine_similarity(persona_embedding_np.reshape(1, -1), passage_embedding.reshape(1, -1))[0][0]

                # Apply sentence completion logic here for the output 'refined_text'
                # The full passage_text is used for embedding, but truncated for display
                display_text = truncate_to_complete_sentence(passage_text, MAX_DISPLAY_CHAR_LENGTH)

                current_doc_subsections_ranked.append({
                    "document": pdf_filename,
                    "refined_text": display_text, # Use the truncated/cleaned text for output
                    "page_number": passage.get("page_number"),
                    "similarity_score": float(semantic_similarity) # Only semantic score
                })
    print(f"[{os.getpid()}] Found {len(current_doc_subsections_ranked)} passages for subsection analysis from {pdf_filename}.")

    print(f"[{os.getpid()}] Finished processing {pdf_filename}.")
    return {
        "pdf_filename": pdf_filename,
        "extracted_sections": current_doc_sections_ranked,
        "subsection_analysis": current_doc_subsections_ranked
    }


# ==============================================================================
# --- Main Orchestration for Parallel Processing ---
# ==============================================================================

def run_challenge_1b_parallel():
    print("--- Starting Challenge 1B Document Analysis Pipeline (Parallel) ---")
    print(f"Input JSON: {INPUT_JSON_PATH}")
    print(f"PDFs from: {PDF_BASE_DIR}")
    print(f"Output to: {OUTPUT_JSON_DIR}")
    print("-" * 50)

    # --- 1. Load Input Configuration ---
    if not INPUT_JSON_PATH.is_file():
        print(f"FATAL ERROR: Input JSON not found at '{INPUT_JSON_PATH}'.")
        return

    try:
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            challenge_input = json.load(f)
    except Exception as e:
        print(f"Error loading input JSON: {e}")
        return

    input_document_filenames = [doc["filename"] for doc in challenge_input.get("documents", [])]
    persona_role = challenge_input.get("persona", {}).get("role", "")
    job_to_be_done_task = challenge_input.get("job_to_be_done", {}).get("task", "")

    # --- ENHANCED PERSONA QUERY (for semantic embedding) ---
    # This query guides the general semantic understanding of the persona's needs.
    # We rely purely on the semantic model for relevance without explicit keywords.
    persona_query = (
        f"Role: {persona_role}. "
        f"Task: {job_to_be_done_task}. "
        "Looking for exciting activities, vibrant cities, delicious food, lively nightlife, "
        "and essential travel tips for a fun group trip. "
        "Highlight key attractions, entertainment, and culinary experiences."
    )
    print(f"\nConstructed Persona Query for Semantic Search:\n'{persona_query}'")


    # --- 2. Embed Persona Query (Main process only) ---
    if not EMBEDDING_MODEL_PATH.is_dir():
        print(f"FATAL ERROR: Embedding model not found at '{EMBEDDING_MODEL_PATH}'.")
        print("Please run 'download_models.py' first.")
        return
        
    temp_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH)
    temp_model = AutoModel.from_pretrained(EMBEDDING_MODEL_PATH)
    temp_model.eval() # Set to eval mode
    temp_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    global _embedding_model, _embedding_tokenizer
    _embedding_model = temp_model
    _embedding_tokenizer = temp_tokenizer

    persona_embedding = get_sentence_embedding_in_process(persona_query)

    _embedding_model = None
    _embedding_tokenizer = None
    del temp_model, temp_tokenizer

    if persona_embedding.size == 0:
        print("Error: Could not generate embedding for persona query. Exiting.")
        return

    # Prepare tasks for the pool (List of tuples: (pdf_filename, persona_embedding_np))
    pdf_tasks = []
    for doc_info in challenge_input.get("documents", []):
        pdf_filename = doc_info.get("filename")
        if pdf_filename:
            # No hardcoded keywords passed to worker anymore
            pdf_tasks.append((pdf_filename, persona_embedding))

    if not pdf_tasks:
        print("No PDFs found to process. Exiting.")
        return

    # --- 3. Parallel Processing ---
    print(f"\nStarting parallel processing for {len(pdf_tasks)} PDFs...")
    num_processes = min(multiprocessing.cpu_count(), len(pdf_tasks))
    print(f"Using {num_processes} processes.")

    all_extracted_sections_from_pdfs_raw = []
    all_refined_subsections_from_pdfs_raw = []
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Use starmap, as process_single_pdf_task now expects 2 arguments
        results = pool.starmap(process_single_pdf_task, pdf_tasks)
        
        for i, result in enumerate(results):
            print(f"[Main] Processed PDF {i+1}/{len(pdf_tasks)}: {result.get('pdf_filename', 'N/A')}")
            all_extracted_sections_from_pdfs_raw.extend(result["extracted_sections"])
            all_refined_subsections_from_pdfs_raw.extend(result["subsection_analysis"])

    print("\nAll parallel PDF processing complete. Consolidating results...")

    # --- 4. Final Ranking and Output Assembly (Main process) ---
    
    # --- 4a. Rank and select top 5 'extracted_sections' globally by SIMILARITY SCORE ---
    all_extracted_sections_from_pdfs_raw.sort(key=lambda x: x["similarity_score"], reverse=True) # Sort by pure semantic similarity
    
    final_extracted_sections = []
    seen_sections_keys = set()
    top_5_document_names = set() # To store document names of top 5 sections
    rank = 1
    for section in all_extracted_sections_from_pdfs_raw:
        unique_key = (section["document"], section["section_title"])
        if unique_key not in seen_sections_keys:
            final_extracted_sections.append({
                "document": section["document"],
                "section_title": section["section_title"],
                "importance_rank": rank,
                "page_number": section["page_number"]
            })
            seen_sections_keys.add(unique_key)
            top_5_document_names.add(section["document"]) # Add document name
            rank += 1
            if rank > 5:
                break
    
    # --- 4b. Filter and rank 'subsection_analysis' based on top 5 documents ---
    # First, filter to only include subsections (passages) from the top 5 documents
    filtered_subsections = [
        sub for sub in all_refined_subsections_from_pdfs_raw
        if sub["document"] in top_5_document_names
    ]

    # Then, sort these filtered subsections by SIMILARITY SCORE
    filtered_subsections.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    final_subsection_analysis = []
    seen_subsections_keys = set()
    for sub_section in filtered_subsections: # Iterate through the already filtered and sorted list
        unique_key = (sub_section["document"], sub_section["refined_text"])
        if unique_key not in seen_subsections_keys:
            final_subsection_analysis.append({
                "document": sub_section["document"],
                "refined_text": sub_section["refined_text"],
                "page_number": sub_section["page_number"]
            })
            seen_subsections_keys.add(unique_key)
            if len(final_subsection_analysis) >= 5: # Limit to top 5
                break

    output_data = {
        "metadata": {
            "input_documents": input_document_filenames,
            "persona": persona_role,
            "job_to_be_done": job_to_be_done_task,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": final_extracted_sections,
        "subsection_analysis": final_subsection_analysis
    }

    output_filename = "challenge1b_output.json"
    output_path = OUTPUT_JSON_DIR / output_filename

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        print(f"\n✅ Challenge 1B output saved to: {output_path}")
    except Exception as e:
        print(f"Error saving final output JSON: {e}")

    print("\n--- Challenge 1B Pipeline Finished Successfully! ---")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_challenge_1b_parallel()