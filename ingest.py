import json
import pymupdf
import pprint
import time
import os
from collections import Counter
import re
import numpy as np


def calculate_weighted_mean_font_size(blocks: list[dict]) -> float:
    """Calculates the character-weighted average font size for a list of blocks."""
    if not blocks:
        return 0.0

    total_char_length = sum(len(block['text']) for block in blocks)
    if total_char_length == 0:
        return 0.0

    weighted_font_sum = sum(block['font_size'] * len(block['text']) for block in blocks)
    return weighted_font_sum / total_char_length

def filter_long_blocks(blocks: list[dict], weighted_avg_font_size: float, max_words: int = 20) -> list[dict]:
    """
    Filters out text blocks that exceed a specified word count, unless they
    are significantly larger than the average font size (likely a title).
    """
    if not blocks:
        return []

    final_blocks = []
    for block in blocks:
        is_long = len(block['text'].split()) > max_words
        is_large_font = block['font_size'] > (weighted_avg_font_size * 1.5)

        if not is_long or is_large_font:
            final_blocks.append(block)

    return final_blocks

def filter_small_fonts_by_weighted_mean(blocks: list[dict], weighted_avg_font_size: float) -> list[dict]:
    """
    Filters out blocks with font sizes at or below a character-weighted average,
    unless the block has other heading-like features like boldness or a unique color.
    """
    if not blocks:
        return []

    colors = [block['color'] for block in blocks]
    majority_color = Counter(colors).most_common(1)[0][0] if colors else 0

    final_blocks = [
        block for block in blocks
        if block['font_size'] > weighted_avg_font_size or
           block['is_bold'] or
           block['color'] != majority_color
    ]

    return final_blocks

def find_repeating_texts(blocks: list[dict], page_count: int, min_occurrence_ratio: float = 0.5) -> set:
    """
    Finds text content that repeats across a significant number of pages.
    """
    if not blocks or page_count == 0:
        return set()

    text_page_map = {}
    for block in blocks:
        text = block['text']
        page = block['page_number']
        if text not in text_page_map:
            text_page_map[text] = set()
        text_page_map[text].add(page)

    min_pages = int(page_count * min_occurrence_ratio)
    if min_pages < 2: min_pages = 2

    repeating_texts = {text for text, pages in text_page_map.items() if len(pages) >= min_pages}

    return repeating_texts

def filter_header_footer_blocks(blocks: list[dict], page_count: int, header_margin: float = 0.12, footer_margin: float = 0.12) -> list[dict]:
    """
    Filters out blocks that are likely headers or footers based on repetition and position.
    """
    if not blocks:
        return []

    repeating_texts = find_repeating_texts(blocks, page_count)
    if not repeating_texts:
        return blocks

    page_height = max(block['bbox'][3] for block in blocks) if blocks else 792
    if page_height == 0: page_height = 792

    header_threshold = page_height * header_margin
    footer_threshold = page_height * (1 - footer_margin)

    filtered_blocks = []
    for block in blocks:
        is_repeating = block['text'] in repeating_texts
        is_in_header = block['bbox'][1] < header_threshold
        is_in_footer = block['bbox'][3] > footer_threshold

        if not (is_repeating and (is_in_header or is_in_footer)):
            filtered_blocks.append(block)

    return filtered_blocks

def post_process_blocks(blocks: list[dict]) -> list[dict]:
    """
    Cleans and finalizes the text content of each logical block.
    """
    processed_blocks = []
    for block in blocks:
        text = block["text"].strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'^\W+|\W+$', '', text)

        if text and re.search('[a-zA-Z]', text):
            block["text"] = text
            processed_blocks.append(block)

    return processed_blocks

def extract_logical_text_blocks(pdf_path: str, line_proximity_threshold: float = 4.0) -> tuple[list[dict], int]:
    """
    Extracts logically coherent text blocks by merging lines based on style and proximity.
    """
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF file: {e}")
        return [], 0

    final_blocks = []
    page_count = doc.page_count
    for page_num, page in enumerate(doc):
        flags = pymupdf.TEXT_PRESERVE_LIGATURES | pymupdf.TEXT_DEHYPHENATE
        page_dict = page.get_text("dict", flags=flags)

        spans = page_dict["blocks"]
        lines = []
        for block_in_pymupdf in spans:
            if block_in_pymupdf.get("type") == 0 and "lines" in block_in_pymupdf:
                for line in block_in_pymupdf["lines"]:
                    line_text, style_counter, line_bbox = "", Counter(), pymupdf.Rect()
                    for span in line["spans"]:
                        font_name_lower = span.get("font", "").lower()
                        is_bold_by_flag = (span.get("flags", 0) & 2**4) > 0
                        is_bold_by_name = "bold" in font_name_lower or "heavy" in font_name_lower or "black" in font_name_lower
                        is_bold = is_bold_by_flag or is_bold_by_name

                        color = span.get("color", 0)
                        span_style = (round(span["size"]), span["font"], is_bold, color)

                        span_text = span["text"]
                        style_counter[span_style] += len(span_text)
                        line_text += span_text
                        line_bbox.include_rect(span["bbox"])
                    if line_text.strip():
                        if not style_counter: continue
                        dominant_style = style_counter.most_common(1)[0][0]
                        lines.append({"text": line_text, "bbox": line_bbox, "style": dominant_style})

        if not lines: continue

        merged_blocks = []
        if lines:
            current_block = {"text": lines[0]["text"], "bbox": pymupdf.Rect(lines[0]["bbox"]), "style": lines[0]["style"]}
            for i in range(1, len(lines)):
                prev_line, current_line = lines[i-1], lines[i]
                same_style = (current_line["style"] == prev_line["style"])
                vertically_close = (current_line["bbox"].y0 - prev_line["bbox"].y1) < line_proximity_threshold
                is_list_item = re.match(r'^\s*([•-]|(\d+\.))\s+', current_line['text'])

                if same_style and vertically_close and not is_list_item:
                    current_block["text"] += " " + current_line["text"]
                    current_block["bbox"].include_rect(current_line["bbox"]) # Corrected for proper bbox expansion
                else:
                    merged_blocks.append(current_block)
                    current_block = {"text": current_line["text"], "bbox": pymupdf.Rect(current_line["bbox"]), "style": current_line["style"]}
            merged_blocks.append(current_block)

        for block in merged_blocks:
            final_blocks.append({
                "page_number": page_num + 1,
                "text": block["text"],
                "bbox": tuple(block["bbox"]),
                "font_size": block["style"][0],
                "font_name": block["style"][1],
                "is_bold": block["style"][2],
                "color": block["style"][3]
            })

    doc.close()
    return final_blocks, page_count

def engineer_layout_features(blocks: list[dict]) -> list[dict]:
    """Adds engineered layout features to each block."""
    if not blocks: return []
    font_sizes = [block['font_size'] for block in blocks if block['font_size'] > 0]
    if not font_sizes:
        for block in blocks:
            block['relative_font_size'] = 0
            block['is_bold_numeric'] = 1 if block['is_bold'] else 0
            block['char_count'] = len(block['text'])
            block['word_count'] = len(block['text'].split())
            page_height = 792
            block['vertical_position'] = block['bbox'][1] / page_height if page_height > 0 else 0
        return blocks

    modal_font_size = Counter(font_sizes).most_common(1)[0][0]

    for block in blocks:
        block['relative_font_size'] = block['font_size'] / modal_font_size if modal_font_size > 0 else 0
        block['is_bold_numeric'] = 1 if block['is_bold'] else 0
        block['char_count'] = len(block['text'])
        block['word_count'] = len(block['text'].split())
        page_height = 792
        block['vertical_position'] = block['bbox'][1] / page_height if page_height > 0 else 0
    return blocks

def identify_headings_and_content_sections(blocks: list[dict]) -> list[dict]:
    """
    Identifies actual headings and groups the subsequent content blocks into
    sections. The 'section_title' will be the heading text, and 'content_text'
    will be the combined text that follows it.
    Prioritizes headings that introduce sub-headings or significant content.
    """
    if not blocks:
        return []

    sections_with_content = []
    current_heading_candidate_block = None # Renamed to emphasize it's a candidate
    current_content_blocks = []

    blocks.sort(key=lambda b: (b['page_number'], b['bbox'][1]))

    all_doc_font_sizes = [block['font_size'] for block in blocks if block['font_size'] > 0]
    modal_doc_font_size = Counter(all_doc_font_sizes).most_common(1)[0][0] if all_doc_font_sizes else 1.0
    
    max_font_size_overall = max(block['font_size'] for block in blocks) if blocks else 0 

    for i, block in enumerate(blocks):
        # Heuristics for a potential heading:
        is_bold_feature = block['is_bold']
        is_large_font_feature = block['font_size'] > modal_doc_font_size * 1.2
        is_short_line_feature = block['word_count'] <= 10 and block['char_count'] < 120

        # Look ahead for hierarchical structure: is next block a smaller heading?
        next_block_is_subheading_candidate = False
        if i + 1 < len(blocks):
            next_block = blocks[i+1]
            if next_block['font_size'] < block['font_size'] and \
               (next_block['is_bold'] or next_block['font_size'] > modal_doc_font_size * 1.1):
                next_block_is_subheading_candidate = True
        
        # Look ahead for body text (standard heading characteristic)
        next_block_is_body_text_indicator = False
        if i + 1 < len(blocks):
            next_block = blocks[i+1]
            # Check if next block is significantly smaller and not bold (typical content)
            if not next_block['is_bold'] and next_block['font_size'] < block['font_size'] * 0.9:
                next_block_is_body_text_indicator = True

        # --- Refined Heading Identification Logic ---
        # A block is considered an 'actual heading' if it's visually prominent
        # AND it's followed by either a subheading or body text.
        is_actual_heading = (is_bold_feature or is_large_font_feature) and \
                            (is_short_line_feature or next_block_is_body_text_indicator or next_block_is_subheading_candidate)

        # Special handling for the very first block of the document:
        # If it's on page 1, is the largest/near largest font, and short (like a document title), skip it.
        # It shouldn't be considered a section heading for 'extracted_sections'.
        if i == 0 and block['page_number'] == 1 and \
           block['font_size'] >= max_font_size_overall * 0.8 and \
           block['word_count'] < 20 and \
           not is_actual_heading: # Only skip if it wasn't already identified as a strong, structured heading (e.g., a real H1 section)
             continue 
        
        if is_actual_heading:
            # If we've found a new heading, close the previous section if one exists
            if current_heading_candidate_block and current_content_blocks:
                combined_content_text = "\n\n".join([b["text"] for b in current_content_blocks])
                if combined_content_text.strip():
                    sections_with_content.append({
                        "heading_text": current_heading_candidate_block["text"],
                        "content_text": combined_content_text,
                        "page_number": current_heading_candidate_block["page_number"],
                        "original_blocks": current_content_blocks # Keep for later
                    })
            # Start a new section with the current block as its heading
            current_heading_candidate_block = block
            current_content_blocks = []
        else:
            # If this block is not a heading, add it to the current content.
            if current_heading_candidate_block: # Only add content if a heading has been established
                current_content_blocks.append(block)

    # Add the last section if any content was accumulated
    if current_heading_candidate_block and current_content_blocks:
        combined_content_text = "\n\n".join([b["text"] for b in current_content_blocks])
        if combined_content_text.strip():
            sections_with_content.append({
                "heading_text": current_heading_candidate_block["text"],
                "content_text": combined_content_text,
                "page_number": current_heading_candidate_block["page_number"],
                "original_blocks": current_content_blocks
            })
            
    return sections_with_content


def group_blocks_into_passages(blocks: list[dict], passage_proximity_threshold: float = 15.0) -> list[dict]:
    """
    Groups contiguous blocks into larger 'passages' based on vertical proximity and similar non-heading style.
    This aims to re-create multi-paragraph sections of prose.
    """
    if not blocks:
        return []

    passages = []
    current_passage = None

    all_body_text_styles = []
    for block in blocks:
        if not block['is_bold'] and block['font_size'] < 16:
             all_body_text_styles.append((block['font_size'], block['is_bold'], block['color']))
    
    most_common_body_style = Counter(all_body_text_styles).most_common(1)[0][0] if all_body_text_styles else None


    for i, block in enumerate(blocks):
        is_potential_separator = block['is_bold'] or (most_common_body_style and block['font_size'] > most_common_body_style[0] * 1.2)
        
        can_merge_with_prev = False
        if current_passage:
            prev_block_in_passage = current_passage["original_blocks"][-1]
            same_page = block["page_number"] == prev_block_in_passage["page_number"]
            vertical_gap = block["bbox"][1] - prev_block_in_passage["bbox"][3]

            is_similar_style = (abs(block['font_size'] - prev_block_in_passage['font_size']) < 2 and
                                block['is_bold'] == prev_block_in_passage['is_bold'] and
                                block['color'] == prev_block_in_passage['color'])

            if same_page and vertical_gap < passage_proximity_threshold and is_similar_style:
                can_merge_with_prev = True
        
        if is_potential_separator:
            if current_passage:
                passages.append(current_passage)
                current_passage = None
        elif can_merge_with_prev and current_passage:
            current_passage["text"] += "\n\n" + block["text"]
            current_passage["end_bbox"][3] = block["bbox"][3]
            current_passage["original_blocks"].append(block)
        else:
            if current_passage:
                passages.append(current_passage)
            current_passage = {
                "text": block["text"],
                "page_number": block["page_number"],
                "start_bbox": list(block["bbox"]),
                "end_bbox": list(block["bbox"]),
                "font_size": block["font_size"],
                "is_bold": block["is_bold"],
                "color": block['color'],
                "original_blocks": [block]
            }
            
    if current_passage:
        passages.append(current_passage)

    for passage in passages:
        passage["bbox"] = tuple(passage["start_bbox"])
        passage['char_count'] = len(passage['text'])
        passage['word_count'] = len(passage['text'].split())
        
        page_height = 792.0
        passage['vertical_position'] = passage['bbox'][1] / page_height if page_height > 0 else 0
        
        all_blocks_font_sizes = [b['font_size'] for b in blocks if b['font_size'] > 0]
        modal_doc_font_size = Counter(all_blocks_font_sizes).most_common(1)[0][0] if all_blocks_font_sizes else 1.0
        passage['relative_font_size'] = passage['font_size'] / modal_doc_font_size if modal_doc_font_size > 0 else 0
        
        passage['is_bold_numeric'] = 1 if passage['is_bold'] else 0
        passage['color'] = passage['original_blocks'][0]['color'] if passage['original_blocks'] else 0 
        
        del passage["start_bbox"]
        del passage["end_bbox"]
        del passage["original_blocks"]

    return passages

# --- Verification Step (only runs if ingest.py is executed directly) ---
if __name__ == "__main__":
    sample_dir = "sample_dataset/input"
    sample_pdf_path = os.path.join(sample_dir, "file02.pdf")

    if not os.path.exists(sample_pdf_path):
        print(f"'{sample_pdf_path}' not found. Creating a dummy PDF for testing.")
        os.makedirs(sample_dir, exist_ok=True)
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((50, 72), "Document Main Title", fontsize=24, fontname="helv-bold")
        page.insert_text((50, 120), "1. Introduction", fontsize=18, fontname="helv-bold")
        page.insert_text((50, 150), "This is the first paragraph of the introduction. It provides some background information. This paragraph is meant to be quite long and flow over multiple lines. It describes the general context of the document.", fontsize=11, fontname="helv")
        page.insert_text((50, 200), "This is the second paragraph. It continues the discussion from the first one. It talks about the importance of the topic and sets the stage for what is to come. These two paragraphs should ideally be grouped into one passage.", fontsize=11, fontname="helv")
        page.insert_text((50, 250), "1.1. Background", fontsize=14, fontname="helv-bold")
        page.insert_text((50, 280), "More details about the background of the study. This paragraph specifically details historical context or previous work relevant to the subject matter.", fontsize=11, fontname="helv")
        page.insert_text((50, 320), "A final short sentence for background, designed to be merged with the one above if proximity allows.", fontsize=11, fontname="helv")
        page.insert_text((50, 350), "2. Methodology", fontsize=18, fontname="helv-bold")
        page.insert_text((50, 380), "Details on the methods used for data collection.", fontsize=11, fontname="helv")
        page.insert_text((50, 750), "Confidential Footer - Page 1", fontsize=9, fontname="helv")

        page2 = doc.new_page()
        page2.insert_text((50, 72), "3. Results and Discussion", fontsize=18, fontname="helv-bold")
        page2.insert_text((50, 100), "3.1. Key Findings", fontsize=14, fontname="helv-bold")
        page2.insert_text((50, 130), "Discussion of the main results obtained from the analysis.", fontsize=11, fontname="helv")
        page2.insert_text((50, 750), "Confidential Footer - Page 2", fontsize=9, fontname="helv")
        doc.save(sample_pdf_path)
        doc.close()
        print("Dummy PDF created.")

    start_time = time.monotonic()

    print(f"\nProcessing '{sample_pdf_path}' for ingestion test...")
    extracted_blocks, page_count = extract_logical_text_blocks(sample_pdf_path)
    print(f"1. Extraction complete. Found {len(extracted_blocks)} logical blocks.")
    
    non_header_footer_blocks = filter_header_footer_blocks(extracted_blocks, page_count)
    print(f"2. Header/Footer filtering complete. {len(non_header_footer_blocks)} blocks remaining.")

    weighted_font_threshold = calculate_weighted_mean_font_size(non_header_footer_blocks)
    print(f"Calculated weighted average font size threshold: {weighted_font_threshold:.2f}")

    short_blocks = filter_long_blocks(non_header_footer_blocks, weighted_font_threshold)
    print(f"3. Long block filtering complete. {len(short_blocks)} blocks remaining.")

    font_filtered_blocks = filter_small_fonts_by_weighted_mean(short_blocks, weighted_font_threshold)
    print(f"4. Small font filtering complete. {len(font_filtered_blocks)} blocks remaining.")

    final_blocks_unfeatured = post_process_blocks(font_filtered_blocks)
    print(f"5. Post-processing and cleaning complete. {len(final_blocks_unfeatured)} blocks remaining.")

    featured_blocks = engineer_layout_features(final_blocks_unfeatured)
    print(f"6. Successfully engineered layout features. Total: {len(featured_blocks)}")

    print("\n--- Testing Heading-Content Section Identification ---")
    sections_with_content = identify_headings_and_content_sections(featured_blocks)
    print(f"7. Identified {len(sections_with_content)} heading-content sections.")
    for i, sec in enumerate(sections_with_content):
        print(f"\n--- Section {i+1} (Page {sec['page_number']}) ---")
        print(f"Heading: '{sec['heading_text']}'")
        print(f"Content (first 200 chars): {sec['content_text'][:200]}...")
        # pprint.pprint(sec) # Uncomment for full section details


    print("\n--- Testing Passage Grouping ---")
    passages = group_blocks_into_passages(featured_blocks)
    print(f"8. Grouped {len(featured_blocks)} blocks into {len(passages)} passages.")
    for i, passage in enumerate(passages):
        print(f"\n--- Passage {i+1} (Page {passage['page_number']}) ---")
        print(f"Text (first 200 chars): {passage['text'][:200]}...")
        print(f"Word Count: {passage.get('word_count', 'N/A')}, Rel. Font Size: {passage.get('relative_font_size', 'N/A'):.2f}")
        # pprint.pprint(passage) # Uncomment for full passage details


    end_time = time.monotonic()
    duration = end_time - start_time
    print(f"\n--- Total ingestion test duration: {duration:.4f} seconds ---")

    if featured_blocks:
        output_dir = "sample_dataset/text-blocks"
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(sample_pdf_path))[0]
        output_json_path = os.path.join(output_dir, f"{base_name}_blocks.json")

        try:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(featured_blocks, f, ensure_ascii=False, indent=4)
            print(f"\nSuccessfully saved {len(featured_blocks)} blocks to '{output_json_path}'.")
        except Exception as e:
            print(f"\nError saving JSON file: {e}")