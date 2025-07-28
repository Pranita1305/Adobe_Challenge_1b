# Adobe Challenge 1B: Persona-Driven Document Intelligence

**Theme:** "Connect What Matters — For the User Who Matters"

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
├── app_pratap/                   # Alternative implementation
├── app_pranita/                  # Alternative implementation  
├── app_shourya/                  # Alternative implementation
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

## 📊 Sample Test Cases

### Test Case 1: Academic Research
- **Documents**: 4 papers on "Graph Neural Networks for Drug Discovery"
- **Persona**: PhD Researcher in Computational Biology
- **Job**: Prepare a literature review focusing on methodologies, datasets, and benchmarks

### Test Case 2: Business Analysis  
- **Documents**: 3 tech company annual reports (2022–2024)
- **Persona**: Investment Analyst
- **Job**: Analyze revenue trends, R&D investment, and market strategy

### Test Case 3: Educational Content
- **Documents**: 5 chapters from organic chemistry textbooks
- **Persona**: Undergraduate Chemistry Student  
- **Job**: Identify key concepts and mechanisms for exam prep on reaction kinetics

## 🔧 Technical Features

- **Semantic Embeddings**: Uses transformer models for deep content understanding
- **Layout Analysis**: Intelligent PDF parsing with font and position analysis
- **Multiprocessing**: Parallel document processing for improved performance
- **Memory Efficient**: CPU-only inference to avoid resource conflicts
- **Containerized**: Fully dockerized for consistent deployment
- **Configurable**: Command-line arguments for different data collections

## 🎯 Solution Generalization

The system is designed to work across:
- **Multiple domains**: Research papers, textbooks, financial reports, technical documentation
- **Diverse personas**: Students, researchers, analysts, professionals
- **Different jobs**: Summarization, review, concept extraction, trend analysis

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

## 💡 Key Innovations

- **Persona-Driven Analysis**: Tailors document analysis to specific user roles and goals
- **Semantic Understanding**: Goes beyond keyword matching to understand content meaning
- **Intelligent Ranking**: Prioritizes content based on relevance to persona's job-to-be-done
- **Production Ready**: Containerized, scalable, and optimized for real-world deployment
