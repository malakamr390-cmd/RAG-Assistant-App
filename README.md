# RAG Lecture Assistant

A Retrieval-Augmented Generation (RAG) web application that answers questions about university lecture content using a local, open-source LLM.

## Overview

This project turns a set of PDF lecture notes into an interactive Q&A assistant. Users ask a question in a chat-style web interface, the system retrieves the most relevant chunks of lecture content from a vector database, and a locally-run LLM (via Ollama) generates a grounded answer with citations back to the source lecture and page number, rather than answering from its own general knowledge.

**Example:**
> **Q:** What is predictive analytics?
> **A:** Predictive analytics consists of techniques that use past data to predict future events or ascertain the impact of one variable on another. [Lecture_1.pdf p.3]

## Architecture

**Flow:**

1. User types a question into the Streamlit frontend (localhost:8501)
2. Frontend sends the question via HTTP POST to the FastAPI backend (localhost:8000/query)
3. Backend embeds the question using sentence-transformers and searches the Chroma vector store for the top-3 most relevant lecture chunks
4. Backend builds a prompt combining the retrieved chunks and the question, then sends it to Ollama (running llama3.2 locally)
5. Ollama generates a grounded answer citing the source lecture and page
6. The answer and its sources are sent back through the backend to the frontend and displayed to the user

Streamlit Frontend --> FastAPI Backend --> Chroma Vector Store (retrieval)
|
v
Ollama LLM (llama3.2)
|
v
Grounded answer + sources
|
v
Back to Streamlit Frontend


## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB (persistent, local) |
| LLM | Ollama (llama3.2, local inference) |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| PDF Parsing | pypdf |
| Testing | pytest, httpx |

## Project Structure
rag-assistant-project/
notebooks/
rag_pipeline.py - Full RAG pipeline: load, chunk, embed, retrieve, evaluate, export
backend/
app/
main.py - FastAPI app, CORS, startup loading
api/routes/query.py - GET /health, POST /query
core/config.py - Settings from .env
schemas/query.py - QueryRequest / QueryResponse
services/
retrieval.py - Load vector store, retrieve chunks
generation.py - Call Ollama LLM, build answer
tests/test_query.py
requirements.txt
.env.example
Dockerfile
frontend/
app.py - Streamlit chat interface
api_client.py - Backend API wrapper
requirements.txt
data/
raw/ - Source lecture PDFs
vector_store/ - Persisted Chroma DB + config
README.md


## Domain & Data

The knowledge base consists of 3 lecture PDFs (61 pages total) from the course BUA 304: Predictive Analytics, covering:
- Lecture 1: Introduction to Predictive Analytics, multiple regression fundamentals
- Lecture 2: Bias-Variance Tradeoff, subset selection, stepwise regression
- Lecture 3: Shrinkage methods (Ridge, Lasso, Elastic Net regression)

All 61 pages (100%) had directly extractable text - no scanned images or OCR was required.

To use your own documents instead, place PDF files in `data/raw/` and re-run `notebooks/rag_pipeline.py` to rebuild the vector store.

## Setup Instructions

### Prerequisites
- Python 3.12
- Ollama installed and running (https://ollama.com)
- Git

### 1. Clone the repository
```bash
git clone https://github.com/malakamr390-cmd/rag-assistant-app.git
cd rag-assistant-app
```

### 2. Create a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Pull the LLM model
```bash
ollama pull llama3.2
```

### 4. (Optional) Rebuild the vector store from scratch
The repository already includes a pre-built vector store in data/vector_store/. To rebuild it from the source PDFs:
```bash
pip install jupyter pandas numpy chromadb sentence-transformers pypdf ollama python-dotenv
python notebooks/rag_pipeline.py
```

## Environment Variables

Backend (backend/.env, optional - all have working defaults):

| Variable | Default | Description |
|---|---|---|
| VECTOR_STORE_PATH | ./data/vector_store | Path to the persisted Chroma vector store |
| EMBEDDING_MODEL | all-MiniLM-L6-v2 | Sentence-transformers model name |
| COLLECTION_NAME | lecture_chunks | Chroma collection name |
| LLM_MODEL | llama3.2 | Ollama model used for generation |
| TOP_K | 3 | Number of chunks retrieved per query |

Frontend (frontend/.env, required):

| Variable | Default | Description |
|---|---|---|
| API_BASE_URL | http://localhost:8000 | URL of the running backend |

## API Reference

### GET /health
Returns server status.

Response:
```json
{ "status": "ok" }
```

### POST /query
Ask a question and receive a grounded, cited answer.

Request body:
```json
{ "question": "What is predictive analytics?" }
```

Response:
```json
{
  "answer": "Predictive analytics consists of techniques that use past data to predict future events or ascertain the impact of one variable on another. [Lecture_1.pdf p.3]",
  "sources": ["Lecture_1.pdf p.3", "Lecture_1.pdf p.1"]
}
```

curl example:
```bash
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d "{\"question\": \"What is predictive analytics?\"}"
```

## Evaluation Results

10 test questions were run against the pipeline and manually verified against the source lecture content.

| # | Question | Correct? | Notes |
|---|---|---|---|
| 1 | What is predictive analytics? | Yes | Exact match to slide definition |
| 2 | Who is the instructor of this course? | Yes | Correct |
| 3 | What topics are covered in this course? | Yes | Matches syllabus exactly |
| 4 | Why does Ridge Regression often perform better than OLS? | Yes | Correct, cited |
| 5 | Why can Lasso set coefficients to exactly zero? | Partial | LLM formatting glitch (see below) |
| 6 | What is the key advantage of Elastic Net regression? | No | Not answered in source slides (exercise-only question); model correctly avoided fabricating |
| 7 | High training accuracy, poor test accuracy - bias or variance? | Yes | Correct |
| 8 | What are the six stages of CRISP-DM? | No | Retrieval miss - answer exists in corpus but wasn't retrieved |
| 9 | What is the difference between reducible and irreducible error? | Yes | Correct, well-cited |
| 10 | What is the difference between supervised and unsupervised learning? | Yes | Correct |

Accuracy: 7/10 fully correct, 1 partial formatting issue, 2 expected failures (unanswerable from source material).

### Failure Analysis

Of the 10 test questions, 7 produced fully correct, grounded answers with accurate citations. Two failure patterns were observed:

1. LLM formatting glitch (Q5): the small local model (llama3.2) occasionally echoed the citation placeholder format literally (e.g. "[filename p.X]") instead of filling it in, despite retrieving the correct source chunk. This is a known limitation of smaller local LLMs following structured output instructions.
2. Retrieval miss (Q8): the CRISP-DM process table exists in the source material (Lecture 1, page 3) but was not retrieved in the top-3 results, likely because the fixed-size chunking split the table away from strongly matching keywords in the question.
3. Correctly declined (Q6): one question was drawn from an exercise list in the slides, never actually answered in the lecture content itself. The system correctly avoided fabricating an answer, demonstrating that the grounding constraint (answer only from retrieved context) works as intended.

Suggested mitigations: increasing top_k from 3 to 5 to improve retrieval recall, and simplifying the prompt's citation instruction to reduce small-model formatting confusion.