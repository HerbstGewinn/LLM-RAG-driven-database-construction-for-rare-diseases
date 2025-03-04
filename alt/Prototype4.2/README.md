# RAG-Driven Document Analysis System

## **Overview**
This project implements a Retrieval-Augmented Generation (RAG) pipeline designed to analyze PDF documents and extract key metadata, such as diseases, treatments, associated genes, and publication details. The pipeline integrates a vector database, LLM-based processing, and metadata validation to ensure accurate extraction and consistency.

---

## **Features**
1. **PDF Loading, Preprocessing and Splitting**:
   - Extracts text from PDFs and optionally removes reference section.  
   - Splits it into chunks for vector-based retrieval.
   
2. **RAG Workflow**:
   - Uses a vector database and a retriever to fetch context-relevant information.
   - Generates precise and context-aware answers using LLMs.
   
3. **Metadata Mapping**:
   - Extracts and maps publication metadata (e.g., DOI, authors, journal name) and disease-related information like ORDO codes and ChEBI IDs.

4. **Validation Pipeline**:
   - Ensures answers align with the facts from the document and validates extracted metadata for correctness (see `graph.png`).

5. **Dynamic Query Adjustment**:
   - Automatically adjusts queries in case of hallucinations or invalid answers.

---

## **Installation**

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt

2. 
   Visit https://www.orphadata.com/alignments/ and download the `en_product1.json` file. 
   This file must be in the programs directory. 

## **Usage**

### **Command-Line Interface**: 
Run the program using the following arguments:
- `--file` (required): Path to the directory containing PDF files.
- `--model` (optional): Specify the LLM model to use. Default: `qwen2.5:72b`.
- `--progression` (optional): Can be set to `False` in the event of a program abort in order to reset the current interim status of the extraction. 
- `--references` (optional): By default, the reference part of the PDFs is removed before the analysis. If you do not want this, you can set the argument to `True`. 

## **Outputs**

1. Extracted metadata and disease-related information are saved in a CSV file (`test_db.csv`).
2. Results are validated against a reference dataset (`validation_data.csv`).

## **File structure**

- ├── main.py  --> Main program orchestrating the pipeline 
- ├── pdf_retriever.py            --> PDF loading, text splitting, and retrieval logic
- ├── llm_access.py               --> Functions for LLM server connection and initialization
- ├── mapping.py                  --> Metadata mapping and enrichment
- ├── validation.py               --> Validation output for extracted data
- ├── graph_state.py              --> State variables of the langgraph
- ├── requirements.txt            --> Python dependencies

## **Dependencies**
Key Python libraries used:
- `langchain`: For LLM-based workflows
- `Chroma`: Vector database for context retrieval.
- `pdfminer`: For PDF text extraction.
- `pandas`: Data manipulation and storage.








