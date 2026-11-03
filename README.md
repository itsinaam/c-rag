# Corrective RAG (CRAG)

A practical implementation of a corrective retrieval-augmented generation pipeline for question answering over a local knowledge base, with fallback strategies for weak or incorrect retrieval results.

This project demonstrates how a RAG system can evaluate retrieved passages, decide whether the evidence is sufficient, and then recover using refinement, web search, or query rewriting before answering. The implementation is built around LangChain, LangGraph, FAISS, OpenAI models, and the Tavily search API.

## Overview

Traditional RAG systems often assume that retrieval is reliable. In practice, retrieved chunks may be weak, insufficient, irrelevant, or ambiguous. This project addresses that problem with a control flow inspired by Corrective RAG (CRAG):

- retrieve candidate documents from a local vector store
- score each retrieved chunk for relevance to the question
- classify the retrieval state as correct, ambiguous, or incorrect
- if retrieval is weak or noisy, refine the search or query
- if needed, perform a targeted web search to augment local context
- generate the final answer from the most relevant available evidence

This repository is structured as a hands-on learning workflow in Jupyter notebooks, progressing from a simple baseline to a complete end-to-end CRAG pipeline.

## Why This Project Matters

Large language models are powerful, but they are often brittle when the retrieved context is incomplete or off-target. By adding a retrieval evaluator and corrective decision layer, the answer pipeline becomes more reliable, especially for:

- vague or underspecified queries
- retrieval failures in domain-specific knowledge bases
- multi-hop or ambiguous questions
- low-quality or partial passages returned by similarity search

## Project Structure

```text
Corrective-Rag-CRAG/
├── 1_basic_rag.ipynb                 # Baseline RAG pipeline
├── 2_retrieval_refinement.ipynb     # Retrieval improvement strategies
├── 3_retrieval_evaluator.ipynb      # Relevance scoring and verdict logic
├── 4_web_search_refinement.ipynb    # Web search fallback for weak retrieval
├── 5_query_rewrite.ipynb            # Query rewriting for better search intent
├── 6_ambiguous.ipynb               # Handling ambiguous or poorly scoped questions
├── final_CRAG.ipynb                 # Integrated end-to-end Corrective RAG pipeline
├── documents/
│   ├── book link.txt               # Source book link for the PDF
│   └── ML Book.pdf                 # Local PDF used for indexing (not always committed)
├── requirements.txt
├── README.md
└── .env                            # Local environment file (create manually)
```

## Notebook Walkthrough

### 1. Basic RAG
Introduces the baseline retrieval-augmented generation flow using a PDF document, FAISS vector search, and an LLM answer generator.

### 2. Retrieval Refinement
Explores methods to improve retrieval quality, including chunk tuning, passage selection, and better search setup.

### 3. Retrieval Evaluator
Adds a scoring layer that judges whether each retrieved chunk is relevant enough to answer a question, enabling pass/fail logic.

### 4. Web Search Refinement
Covers fallback behavior where the local corpus is insufficient, allowing the system to fetch external information from the web via Tavily.

### 5. Query Rewrite
Improves ambiguous or weak user questions by rewriting them into better search queries before retrieval or external search.

### 6. Ambiguous Questions
Demonstrates how a RAG workflow can respond more carefully when the user intent is not fully clear.

### final_CRAG.ipynb
Combines all of the above into an integrated Corrective RAG architecture with retrieval scoring, state transitions, web search fallback, and final answer generation.

## Technical Stack

- Python 3.10+
- LangChain
- LangGraph
- FAISS
- OpenAI embeddings and chat models
- Tavily search API
- PyPDF for document loading
- Python dotenv for environment management

## Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd Corrective-Rag-CRAG
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
```

3. Install the project requirements:

```bash
pip install -r requirements.txt
```

4. Add your API keys in a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

If you do not use a `.env` file, you may set these variables in your shell environment before running the notebooks.

## Data Setup

This project uses a machine learning textbook PDF as the foundational knowledge source. The repository contains a link to the source document in `documents/book link.txt`.

You should place the actual PDF file in the `documents/` folder with the expected filename:

```text
documents/ML Book.pdf
```

If your file has a different name, update the notebook loader paths accordingly.

## Running the Notebooks

Open the notebooks in order, beginning with `1_basic_rag.ipynb`, and continue through the refinement and corrective stages. The final notebook, `final_CRAG.ipynb`, represents the complete system.

For local execution, use Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
```

or

```bash
jupyter lab
```

## Core Workflow

The final architecture follows a corrective decision loop:

1. Retrieve the top-k chunks from the vector store.
2. Evaluate each chunk for relevance and utility.
3. Determine whether the evidence is:
   - correct
   - ambiguous
   - incorrect
4. Use the verdict to decide the next action:
   - answer directly from local evidence
   - refine retrieval for better context
   - issue a web search for missing information
   - rewrite the query to improve search intent
5. Generate a final answer grounded in the selected evidence.

## Notes

- This repository is designed for experimentation and learning rather than production deployment.
- The logic is intentionally transparent to help understand the role of retrieval quality in RAG systems.
- The notebooks can be adapted to other document sets, knowledge bases, or enterprise search workflows.

## Recommended Next Steps

- Replace the sample textbook with your own domain-specific corpus.
- Add evaluation metrics for retrieval precision and answer quality.
- Convert the notebook flow into a reusable Python application or API service.
- Integrate logging, observability, and tracing for production-grade monitoring.

## References

- LangChain documentation: https://python.langchain.com/
- LangGraph documentation: https://langchain-ai.github.io/langgraph/
- Tavily search: https://tavily.com/
- Corrective RAG paper: https://arxiv.org/pdf/2401.15884

## License

This project does not currently include a formal license file. If you plan to reuse or distribute it, add an appropriate open-source license before publication.
