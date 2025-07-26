import json
import os
import datetime
import numpy as np
from scipy.spatial.distance import cosine # For cosine similarity
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer
import torch

# Import the Part 1A processor
from part1a_processor import extract_structured_outline

# --- Configuration ---
# Paths for models (relative to /app in Docker, which is Collection 1)
SBERT_MODEL_PATH = "./models/sbert_minilm"
QA_MODEL_PATH = "./models/distilbert_qa"

# Input/Output directories (relative to /app in Docker)
INPUT_PDF_DIR = "./PDFs" # PDFs are now directly in./PDFs
PART1A_PROCESSED_OUTPUT_DIR = "./part1a_processed_output" # Intermediate output from Part 1A
OUTPUT_JSON_PATH = "./output/challenge1b_output.json"
CHALLENGE1B_INPUT_PATH = "./challenge1b_input.json"

# --- Model Loading ---
def load_models():
    """Loads the Sentence-BERT and DistilBERT QA models locally."""
    print("Loading Sentence-BERT model...")
    sbert_model = SentenceTransformer(SBERT_MODEL_PATH, device='cpu')
    sbert_model.eval() # Set to evaluation mode

    print("Loading DistilBERT QA model and tokenizer...")
    qa_tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_PATH)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_PATH)
    qa_model.eval() # Set to evaluation mode

    print("Models loaded successfully.")
    return sbert_model, qa_tokenizer, qa_model

# --- Data Processing Helpers ---
def load_part1a_processed_output(pdf_filename: str) -> dict:
    """
    Loads the JSON output from the Part 1A processing step for a given PDF.
    Assumes Part 1A output JSONs are named after the PDF (e.g., 'doc.pdf' -> 'doc.json').
    """
    json_filename = os.path.splitext(pdf_filename) + ".json" # Use  to get base name without extension
    json_path = os.path.join(PART1A_PROCESSED_OUTPUT_DIR, json_filename)
    if not os.path.exists(json_path):
        print(f"Warning: Part 1A processed output not found for {pdf_filename} at {json_path}. Skipping.")
        return None
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}. Skipping.")
        return None

def get_all_chunks(part1a_output: dict, pdf_filename: str) -> list:
    """
    Extracts all sections/sub-sections (chunks) with their content and metadata
    from the Part 1A structured outline.
    Assumes each outline entry has a 'content' field with the full text.
    """
    chunks =[]
    if not part1a_output or 'outline' not in part1a_output:
        return chunks

    for entry in part1a_output['outline']:
        # Ensure 'content' field exists as per assumption from Part 1A processor
        if 'content' in entry and entry['content'] and entry['text']:
            chunks.append({
                "document": pdf_filename,
                "level": entry.get('level', 'Body'), # Default to 'Body' if level not specified
                "title": entry['text'],
                "page_number": entry.get('page', 0), # Default to 0 if page not specified
                "content": entry['content']
            })
    return chunks

def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculates cosine similarity between two vectors."""
    # Ensure vectors are 1D and convert to float if not already
    vec1 = np.asarray(vec1).flatten().astype(float)
    vec2 = np.asarray(vec2).flatten().astype(float)

    # Handle zero vectors to avoid division by zero
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0

    return 1 - cosine(vec1, vec2) # scipy.spatial.distance.cosine returns cosine distance, so 1 - distance

# --- Main Processing Logic ---
def process_document_collection():
    """
    Main function to process the document collection based on persona and job-to-be-done.
    This now includes the Part 1A processing step.
    """
    # Ensure output directories exist
    os.makedirs(PART1A_PROCESSED_OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)

    # --- Step 1: Run Part 1A processing for all PDFs ---
    print("--- Starting Part 1A Processing ---")
    pdf_files_to_process =[]
    
    if not pdf_files_to_process:
        print(f"No PDF files found in {INPUT_PDF_DIR}. Please place PDFs there.")
        return

    for pdf_filename in pdf_files_to_process:
        pdf_path = os.path.join(INPUT_PDF_DIR, pdf_filename)
        print(f"Running Part 1A for: {pdf_filename}")
        part1a_output_data = extract_structured_outline(pdf_path)
        
        output_json_filename = os.path.splitext(pdf_filename) + ".json" # Use  for base name
        output_json_path = os.path.join(PART1A_PROCESSED_OUTPUT_DIR, output_json_filename)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(part1a_output_data, f, indent=2, ensure_ascii=False)
        print(f"Part 1A output saved to: {output_json_path}")
    print("--- Part 1A Processing Complete ---")

    # --- Step 2: Proceed with Part 1B logic ---
    sbert_model, qa_tokenizer, qa_model = load_models()

    # Load persona and job-to-be-done from challenge1b_input.json
    try:
        with open(CHALLENGE1B_INPUT_PATH, 'r', encoding='utf-8') as f:
            challenge_input = json.load(f)
        persona_definition = challenge_input.get("persona_definition", "")
        job_to_be_done = challenge_input.get("job_to_be_done", "")
        if not persona_definition or not job_to_be_done:
            raise ValueError("Persona definition or job-to-be-done missing in challenge1b_input.json")
    except FileNotFoundError:
        print(f"Error: {CHALLENGE1B_INPUT_PATH} not found. Please ensure it exists.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {CHALLENGE1B_INPUT_PATH}.")
        return
    except ValueError as e:
        print(f"Error in input JSON: {e}")
        return

    query_string = f"Persona: {persona_definition}. Task: {job_to_be_done}"
    print(f"\n--- Starting Part 1B Processing ---")
    print(f"Processing query: '{query_string}'")

    # Generate embedding for the combined query
    query_embedding = sbert_model.encode([query_string])

    all_document_chunks =[]
    input_documents_metadata =[]

    # Iterate through Part 1A processed JSON files
    for json_filename in os.listdir(PART1A_PROCESSED_OUTPUT_DIR):
        if json_filename.lower().endswith(".json"):
            original_pdf_filename = os.path.splitext(json_filename) + ".pdf" # Reconstruct original PDF name
            print(f"Loading Part 1A output for: {original_pdf_filename}")
            input_documents_metadata.append({"filename": original_pdf_filename, "part1a_output_path": os.path.join(PART1A_PROCESSED_OUTPUT_DIR, json_filename)})

            part1a_output = load_part1a_processed_output(original_pdf_filename)
            if part1a_output:
                chunks_from_pdf = get_all_chunks(part1a_output, original_pdf_filename)
                if chunks_from_pdf:
                    chunk_contents = [chunk['content'] for chunk in chunks_from_pdf]
                    # Filter out empty strings before encoding to avoid errors
                    valid_chunk_contents = [c for c in chunk_contents if c.strip()]
                    if valid_chunk_contents:
                        chunk_embeddings = sbert_model.encode(valid_chunk_contents, show_progress_bar=False)
                        
                        # Map embeddings back to original chunks, handling empty content
                        embedding_idx = 0
                        for i, chunk in enumerate(chunks_from_pdf):
                            if chunk['content'].strip():
                                chunk['semantic_similarity'] = calculate_cosine_similarity(query_embedding, chunk_embeddings[embedding_idx])
                                embedding_idx += 1
                            else:
                                chunk['semantic_similarity'] = 0.0 # Assign 0 similarity for empty content
                            all_document_chunks.append(chunk)
                    else:
                        print(f"No valid content chunks found in Part 1A output for {original_pdf_filename}.")
                else:
                    print(f"No valid chunks found in Part 1A output for {original_pdf_filename}.")
            else:
                print(f"Could not load Part 1A output for {original_pdf_filename}.")

    # Sort all chunks by semantic similarity to get initial ranking
    all_document_chunks.sort(key=lambda x: x['semantic_similarity'], reverse=True)

    extracted_sections =[]
    sub_section_analysis =[]

    # Process top-ranked chunks for Extractive QA
    # Adjust TOP_K_CHUNKS based on performance testing and desired granularity.
    # A higher TOP_K_CHUNKS means more QA inferences, increasing processing time.
    TOP_K_CHUNKS = 30 # Example: Process top 30 most semantically relevant chunks for QA

    processed_section_identifiers = set() # To avoid duplicate sections in extracted_sections

    for i, chunk in enumerate(all_document_chunks):
        # Add to extracted_sections if it's a unique section title (H1/H2/H3/Body)
        # Use a tuple of (document, title, page_number) to uniquely identify a section
        section_identifier = (chunk['document'], chunk['title'], chunk['page_number'])
        if section_identifier not in processed_section_identifiers:
            extracted_sections.append({
                "document": chunk['document'],
                "page_number": chunk['page_number'],
                "section_title": chunk['title'],
                "importance_rank": round(chunk['semantic_similarity'], 4) # Use semantic similarity as initial rank
            })
            processed_section_identifiers.add(section_identifier)

        # Only perform QA for a subset of top chunks to manage time, and if content is substantial
        if i < TOP_K_CHUNKS and chunk['content'].strip() and len(chunk['content'].split()) > 10: # Avoid very short chunks for QA
            try:
                inputs = qa_tokenizer(job_to_be_done, chunk['content'], return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = qa_model(**inputs)

                answer_start_scores = outputs.start_logits
                answer_end_scores = outputs.end_logits

                # Get the most likely beginning and end of the answer span
                answer_start = torch.argmax(answer_start_scores)
                answer_end = torch.argmax(answer_end_scores) + 1 # +1 because slicing is exclusive

                # Extract the answer span
                input_ids = inputs["input_ids"].tolist()
                # Ensure start and end indices are within bounds
                if answer_start < len(input_ids) and answer_end <= len(input_ids) and answer_start < answer_end:
                    refined_text = qa_tokenizer.decode(input_ids[answer_start:answer_end], skip_special_tokens=True)
                else:
                    refined_text = "" # No valid answer span found

                # Calculate confidence score (simple softmax of start/end logits)
                qa_confidence = 0.0
                if refined_text:
                    qa_confidence = (torch.max(torch.softmax(answer_start_scores, dim=-1)) +
                                     torch.max(torch.softmax(answer_end_scores, dim=-1))) / 2.0
                    qa_confidence = round(qa_confidence.item(), 4)

                # Only add to sub_section_analysis if a meaningful answer is found and confidence is high
                if refined_text and qa_confidence > 0.1 and len(refined_text.split()) > 3: # Threshold for meaningful answer
                    sub_section_analysis.append({
                        "document": chunk['document'],
                        "page_number": chunk['page_number'],
                        "refined_text": refined_text,
                        "original_section_title": chunk['title'], # Add original section title for context
                        "qa_confidence": qa_confidence # Confidence of the QA model
                    })
            except Exception as e:
                print(f"Error during QA for chunk from {chunk['document']} (page {chunk['page_number']}): {e}")
                # Continue to next chunk if QA fails for one

    # Sort extracted_sections by importance_rank (semantic similarity)
    extracted_sections.sort(key=lambda x: x['importance_rank'], reverse=True)

    # Sort sub_section_analysis by QA confidence
    sub_section_analysis.sort(key=lambda x: x.get('qa_confidence', 0), reverse=True)

    # Prepare final output JSON
    output_data = {
        "metadata": {
            "input_documents": input_documents_metadata,
            "persona": persona_definition,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds') + 'Z'
        },
        "extracted_sections": extracted_sections,
        "sub_section_analysis": sub_section_analysis
    }

    # Save output to JSON file
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n--- Part 1B Processing Complete ---")
    print(f"Final output saved to {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    process_document_collection()