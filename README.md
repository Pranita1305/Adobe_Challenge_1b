# Adobe_Challenge_1b
Round 1B: Persona-Driven Document Intelligence
Theme: “Connect What Matters — For the User Who Matters”
🧠 Challenge Brief
You will build a system that acts as an intelligent document analyst, extracting and prioritizing the most relevant sections from a collection of documents based on a specific persona and their job-to-be-done.

📥 Input Specification
Document Collection: 3–10 related PDFs

Persona Definition: Role description with specific expertise and focus areas

Job-to-be-Done: Concrete task the persona needs to accomplish

Your solution should be generic enough to generalize across:

Multiple domains (e.g., research papers, textbooks, financial reports)

Diverse personas (e.g., student, researcher, analyst)

Different jobs (e.g., summarization, review, concept extraction)

📊 Sample Test Cases
Test Case 1: Academic Research
Documents: 4 papers on "Graph Neural Networks for Drug Discovery"

Persona: PhD Researcher in Computational Biology

Job: Prepare a literature review focusing on methodologies, datasets, and benchmarks

Test Case 2: Business Analysis
Documents: 3 tech company annual reports (2022–2024)

Persona: Investment Analyst

Job: Analyze revenue trends, R&D investment, and market strategy

Test Case 3: Educational Content
Documents: 5 chapters from organic chemistry textbooks

Persona: Undergraduate Chemistry Student

Job: Identify key concepts and mechanisms for exam prep on reaction kinetics

📤 Required Output (JSON)
The JSON output should include:

1. Metadata:
List of input documents

Persona

Job-to-be-done

Timestamp

2. Extracted Sections:
Document

Page number

Section title

Importance rank

3. Sub-Section Analysis:
Document

Refined text

Page number
