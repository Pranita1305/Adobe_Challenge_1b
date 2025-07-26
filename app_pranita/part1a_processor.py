import fitz # PyMuPDF
import json
import os

def extract_structured_outline(pdf_path: str) -> dict:
    """
    Extracts a simplified structured outline (Title, H1, H2, H3) and their content
    from a PDF using PyMuPDF and basic heuristics.

    NOTE: This is a simplified heuristic-based implementation for Part 1A
    to enable the Part 1B pipeline. For a robust solution meeting the
    "ML model is a must" criteria for Part 1A [1, 1], a more sophisticated ML
    approach would be required. You should replace this function with your
    actual Part 1A ML-based solution.
    """
    document = fitz.open(pdf_path)
    outline =[]
    full_text_blocks =[]
    title = os.path.basename(pdf_path).replace(".pdf", "") # Default title

    # Heuristic for font sizes: Collect all font sizes and try to infer hierarchy
    font_sizes = {}
    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b["type"] == 0: # text block
                for line in b["lines"]:
                    for span in line["spans"]:
                        font_size = round(span["size"], 1)
                        font_flags = span["flags"] # 1=superscript, 2=italic, 4=serifed, 8=monospaced, 16=bold
                        is_bold = bool(font_flags & 16)
                        font_sizes[font_size] = font_sizes.get(font_size, 0) + 1

    # Sort unique font sizes to infer potential heading sizes
    sorted_font_sizes = sorted(font_sizes.keys(), reverse=True)
    
    # Assign arbitrary levels based on size, assuming larger means higher level
    # This is a very basic heuristic and will likely fail on complex documents
    h_levels = {}
    if len(sorted_font_sizes) >= 1: h_levels['H1'] = sorted_font_sizes
    if len(sorted_font_sizes) >= 2: h_levels['H2'] = sorted_font_sizes[1]
    if len(sorted_font_sizes) >= 3: h_levels['H3'] = sorted_font_sizes[2]

    current_section_content = ""
    last_level = None
    last_section_start_page = 0

    def finalize_current_section():
        nonlocal current_section_content
        if outline and current_section_content:
            # Append content to the last added outline entry
            if 'content' not in outline[-1]:
                outline[-1]['content'] = ""
            outline[-1]['content'] += current_section_content.strip()
        current_section_content = ""

    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]

        for b_idx, b in enumerate(blocks):
            if b["type"] == 0: # text block
                block_text = ""
                for line in b["lines"]:
                    for span in line["spans"]:
                        block_text += span["text"]
                block_text = block_text.strip()
                if not block_text:
                    continue

                is_heading = False
                level = None
                
                if b["lines"]: # Ensure there are lines to check spans
                    first_span = b["lines"]["spans"]
                    first_span_size = round(first_span["size"], 1)
                    first_span_flags = first_span["flags"]
                    is_bold = bool(first_span_flags & 16)

                    if h_levels.get('H1') == first_span_size and is_bold:
                        level = "H1"
                        is_heading = True
                    elif h_levels.get('H2') == first_span_size and is_bold:
                        level = "H2"
                        is_heading = True
                    elif h_levels.get('H3') == first_span_size and is_bold:
                        level = "H3"
                        is_heading = True

                if is_heading:
                    finalize_current_section() # Finalize content for previous heading
                    outline.append({
                        "level": level,
                        "text": block_text,
                        "page": page_num # 0-indexed page number
                    })
                    last_level = level
                    last_section_start_page = page_num
                else:
                    current_section_content += block_text + "\n"
                    
    finalize_current_section() # Finalize content for the very last section

    # If no headings were found, or if there's significant unclassified text,
    # add it as a single "Body" section.
    if not outline and full_text_blocks:
        outline.append({
            "level": "Body",
            "text": "Document Content",
            "page": 0,
            "content": "\n".join(full_text_blocks).strip()
        })
    elif full_text_blocks and outline:
        # If there's unclassified text and an outline, append to the last section
        if 'content' not in outline[-1]:
            outline[-1]['content'] = ""
        outline[-1]['content'] += "\n" + "\n".join(full_text_blocks).strip()

    # Ensure all outline entries have a 'content' field, even if empty
    for entry in outline:
        if 'content' not in entry:
            entry['content'] = ""

    document.close()
    return {
        "title": title,
        "outline": outline
    }

if __name__ == "__main__":
    # Example usage for testing part1a_processor directly
    # Make sure you have a PDF named 'South of France - Cities.pdf' in the 'PDFs' directory
    sample_pdf_path = "./PDFs/South of France - Cities.pdf" # Adjusted path for new structure
    if os.path.exists(sample_pdf_path):
        print(f"Processing {sample_pdf_path} with Part 1A processor...")
        output_data = extract_structured_outline(sample_pdf_path)
        output_filename = os.path.splitext(os.path.basename(sample_pdf_path)) + ".json"
        output_dir = "./part1a_processed_output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Part 1A output saved to {output_path}")
    else:
        print(f"Sample PDF not found at {sample_pdf_path}. Please place a PDF in the 'PDFs' directory for testing.")