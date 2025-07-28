# Adobe Challenge 1B: Persona-Driven Document Intelligence

**Theme:** "Connect What Matters — For the User Who Matters"

## 🚀 Quick Start

1. **Clone the repository**
2. **Build the Docker container**: `docker build -t adobe-challenge1b .`
3. **Run with your collection**: `docker run adobe-challenge1b "Collection 1"`
4. **Check output**: Results saved to `data/Collection 1/challenge1b_output.json`

## 📋 Requirements

- Docker (for containerized deployment)
- OR Python 3.10+ with dependencies (for local development)
- Input PDFs in the specified collection directory
- Valid `challenge1b_input.json` configuration file

## 🧠 Challenge Brief

This system acts as an intelligent document analyst, extracting and prioritizing the most relevant sections from a collection of documents based on a specific persona and their job-to-be-done. It uses advanced semantic embeddings and multiprocessing for efficient document analysis.

## 🏗️ Project Structure

```
Adobe_Challenge_1b/
├── app/                           # Main application directory
│   ├── main_challenge1b.py       # Main pipeline script
│   ├── ingest.py                 # PDF text extraction and processing
│   ├── download_models.py        # Downloads transformer models
│   ├── requirements.txt          # Python dependencies
│   ├── data/                     # Input/output data collections
│   │   ├── Collection 1/
│   │   │   ├── challenge1b_input.json
│   │   │   ├── challenge1b_output.json
│   │   │   └── PDFs/
│   │   ├── Collection 2/
│   │   └── Collection 3/
│   └── transformers_models/      # Downloaded embedding models
│       └── sentence-transformers_all-MiniLM-L6-v2/
├── Dockerfile                    # Container configuration
└── README.md                     # This file
```

## 🚀 How It Works

### 1. **Document Processing Pipeline**
- **PDF Text Extraction**: Extracts logical text blocks from PDFs using layout analysis
- **Feature Engineering**: Analyzes font sizes, positioning, and layout features
- **Content Filtering**: Removes headers, footers, and irrelevant content
- **Section Detection**: Identifies document structure and main sections
- **Passage Grouping**: Groups related text blocks into coherent passages

### 2. **Semantic Analysis**
- **Persona Embedding**: Creates semantic embeddings from persona role and job-to-be-done
- **Content Embedding**: Generates embeddings for document sections and passages
- **Similarity Scoring**: Uses cosine similarity to match content with persona needs
- **Ranking & Selection**: Selects top 5 most relevant sections and passages

### 3. **Parallel Processing**
- **Multiprocessing**: Processes multiple PDFs simultaneously for performance
- **CPU-Only Inference**: Avoids CUDA conflicts in multiprocessing environment
- **Memory Optimization**: Loads models once per worker process

## 📥 Input Specification

**Document Collection**: 3–10 related PDFs per collection  
**Persona Definition**: Role description with specific expertise and focus areas  
**Job-to-be-Done**: Concrete task the persona needs to accomplish

### Input JSON Format
```json
{
  "documents": [
    {"filename": "document1.pdf"},
    {"filename": "document2.pdf"}
  ],
  "persona": {
    "role": "PhD Researcher in Computational Biology"
  },
  "job_to_be_done": {
    "task": "Prepare a literature review focusing on methodologies"
  }
}
```

## 📤 Output Specification

The system generates a JSON output with:

### 1. **Metadata**
- List of input documents
- Persona information  
- Job-to-be-done task
- Processing timestamp

### 2. **Extracted Sections** (Top 5)
- Document name
- Section title
- Importance rank
- Page number

### 3. **Subsection Analysis** (Top 5)
- Document name
- Refined text (truncated to complete sentences)
- Page number

## 2. The Development Process: From Parsing to Persona Matching

The solution's robustness for Challenge 1B stems from a carefully designed pipeline that combines precise PDF content extraction with advanced semantic understanding.

### 2.1. Feature Engineering and Content Structuring (`ingest.py`)

The `ingest.py` script is the foundational layer, responsible for transforming raw PDF content into a structured, feature-rich format suitable for analysis.

**Logical Text Block Extraction:** Uses PyMuPDF to parse each page, extracting text blocks along with detailed layout features such as font size, font name, bold status, color, and bounding box coordinates. It intelligently merges visually contiguous lines into coherent text blocks.

**Filtering:** Implements several filtering steps to remove irrelevant content like headers, footers, and very short or visually insignificant blocks.

**Layout Feature Engineering:** Augments each block with computed features like `relative_font_size` (relative to the document's modal font size), `word_count`, and `vertical_position` on the page.

**Hierarchical Heading and Section Identification (`identify_headings_and_content_sections`):**

This is a critical, refined component that goes beyond simple heading detection. It uses a sophisticated heuristic-based approach to identify top-level section headings within the document. This involves analyzing font prominence (boldness, size relative to body text), line length, and the nature of the text blocks immediately following (e.g., presence of sub-headings or body prose).

It then groups all subsequent content blocks until the next detected top-level heading, forming a complete conceptual section. This allows the `section_title` in the output to be a meaningful high-level heading (e.g., "Falafel" for a recipe).

The overall document title (if present on the first page) is intelligently identified and excluded from being an `extracted_section` heading.

**Passage Grouping (`group_blocks_into_passages`):**

Further refines the extracted content by grouping contiguous body text blocks into larger, semantically coherent "passages." This is essential for `subsection_analysis` to ensure complete ideas are presented, rather than fragmented sentences. It uses vertical proximity and consistent styling to define these passages.

### 2.2. Semantic Search and Ranking (`main_challenge1b.py`)

The `main_challenge1b.py` script orchestrates the entire persona-based content discovery and ranking process.

**Parallel Processing:** Leverages Python's `multiprocessing` module (`multiprocessing.Pool` with `starmap`) to process multiple PDF documents concurrently, significantly reducing overall execution time. Each worker process loads its own instance of the MiniLM model for independent processing.

**Dynamic Persona Query:** It reads the `persona` (role) and `job_to_be_done` (task) from the `challenge1b_input.json` to construct a rich, descriptive persona query. This query is then used for all semantic comparisons.

**Model Selection (`sentence-transformers/all-MiniLM-L6-v2`):**

This model was chosen due to its excellent balance of:
- **Size:** Approximately 90MB, well within typical memory constraints
- **Performance:** Provides high-quality semantic embeddings
- **Efficiency:** Fast inference speed, suitable for CPU-only execution
- **Robustness:** Adept at capturing nuanced semantic relationships

**Two-Stage Content Extraction and Ranking:**

**Stage 1: Extracted Sections (`extracted_sections`):**
- For each PDF, it identifies major sections using `identify_headings_and_content_sections`
- It calculates the semantic similarity between the persona query and the full content text associated with each identified section heading
- All identified and scored sections across all PDFs are then globally ranked by their semantic similarity score. The top 5 unique sections are selected. The `section_title` in the output will be the actual heading text from the PDF (e.g., "Falafel")

**Stage 2: Subsection Analysis (`subsection_analysis`):**
- This stage focuses its search only within the PDFs that contributed to the top 5 `extracted_sections`
- It takes all the identified "passages" (from `group_blocks_into_passages`) from these relevant PDFs
- It calculates the semantic similarity between the persona query and each passage
- All relevant passages are then globally ranked by their semantic similarity score. The top 5 unique passages are selected

**Complete Sentence Output:** A `truncate_to_complete_sentence` utility ensures that the `refined_text` for these passages is coherently truncated to a maximum display length, always ending on a complete sentence, even if the original passage was very long.

## 3. System Architecture and Final Pipeline

The final solution is a streamlined, efficient, and parallelized Python application.

- **`main_challenge1b.py`:** The primary script. It loads the single MiniLM model, sets up the multiprocessing pool, and orchestrates the entire workflow from reading input JSON to generating the final ranked output.

- **`ingest.py`:** Contains the core reusable functions for detailed PDF parsing, layout feature engineering, and intelligent content structuring (identifying heading-content sections and grouping passages).

- **`download_models.py`:** A utility script for the one-time download of the required transformer model, ensuring offline capability.

- **`requirements.txt`:** Manages all Python dependencies.


## 🐳 Running with Docker

### Build the Container
```bash
docker build -t adobe-challenge1b .
```

### Run with Different Collections
```bash
# Process Collection 1
docker run adobe-challenge1b "Collection 1"

# Process Collection 2  
docker run adobe-challenge1b "Collection 2"

# Process Collection 3
docker run adobe-challenge1b "Collection 3"
```

### What Happens During Build
1. **Dependencies Installation**: Installs all Python packages from `requirements.txt`
2. **Model Download**: Downloads `sentence-transformers/all-MiniLM-L6-v2` model offline
3. **Environment Setup**: Configures the container for document processing

### What Happens During Run
1. **Input Loading**: Reads `challenge1b_input.json` from specified collection
2. **PDF Processing**: Extracts and analyzes content from all PDFs in the collection
3. **Semantic Matching**: Matches content against persona requirements
4. **Output Generation**: Saves results to `challenge1b_output.json`

## 🛠️ Local Development

### Prerequisites
- Python 3.10+
- Required packages in `app/requirements.txt`

### Setup
```bash
cd app/
pip install -r requirements.txt
python download_models.py
```

### Run Locally
```bash
python main_challenge1b.py --collection "Collection 1"
```

## 🔧 Technical Features

- **Semantic Embeddings**: Uses transformer models for deep content understanding
- **Layout Analysis**: Intelligent PDF parsing with font and position analysis
- **Multiprocessing**: Parallel document processing for improved performance
- **Memory Efficient**: CPU-only inference to avoid resource conflicts
- **Containerized**: Fully dockerized for consistent deployment
- **Configurable**: Command-line arguments for different data collections
