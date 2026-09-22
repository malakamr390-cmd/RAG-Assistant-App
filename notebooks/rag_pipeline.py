from pathlib import Path
from pypdf import PdfReader
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"   # script is inside notebooks/, so go up one level

summary = []
pages = []   # every usable page: source filename + page number + text (needed later for citations)

for pdf_path in sorted(RAW_DIR.glob("*.pdf")):
    try:
        reader = PdfReader(pdf_path)
        n_pages = len(reader.pages)
        empty_pages = 0
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if len(text) < 50:          # near-empty -> likely an image/diagram slide
                empty_pages += 1
                continue
            pages.append({"source": pdf_path.name, "page": i, "text": text})
        summary.append({"file": pdf_path.name, "pages": n_pages,
                        "empty_pages": empty_pages, "status": "ok"})
    except Exception as e:
        summary.append({"file": pdf_path.name, "pages": 0,
                        "empty_pages": 0, "status": f"FAILED: {e}"})

df = pd.DataFrame(summary)
print(df)
print("\nFiles:", len(df), "| Total pages:", df["pages"].sum(),
      "| Pages with usable text:", len(pages))


print("\n--- Sample page ---")
print(pages[0]["source"], "| page", pages[0]["page"])
print(pages[0]["text"][:600])




#Chunking Strategy

print("\n=== 2.2 Chunking Strategy ===")

CHUNK_SIZE = 500      # characters per chunk
CHUNK_OVERLAP = 50    # characters shared between consecutive chunks

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap   # move forward, but re-include the overlap
    return chunks

all_chunks = []  # each item: source, page, chunk_id, text
for p in pages:
    page_chunks = chunk_text(p["text"])
    for idx, c in enumerate(page_chunks):
        all_chunks.append({
            "source": p["source"],
            "page": p["page"],
            "chunk_id": idx,
            "text": c
        })

print(f"Total pages: {len(pages)}")
print(f"Total chunks: {len(all_chunks)}")
print(f"Average chunks per page: {len(all_chunks)/len(pages):.1f}")
print("\n--- Sample chunk ---")
print(all_chunks[0])
#Chunking strategy: Fixed-size chunking with 500 characters per chunk and 50-character overlap. 
#This size keeps each chunk focused enough to stay topically coherent (roughly one paragraph or a slide's worth of content), 
#while staying small enough that retrieval returns precise, relevant text rather than diluting it with unrelated content from the same page. 
#The 50-character overlap (10% of chunk size) reduces the chance that a sentence or concept gets awkwardly split exactly at a chunk boundary, 
#at the cost of some redundancy across chunks (224 chunks from 61 pages, ~3.7 chunks/page on average).


#Embeddings & Vector Store
print("\n=== 2.3 Embeddings & Vector Store ===")

from sentence_transformers import SentenceTransformer
import chromadb

# Load a small, fast, well-tested embedding model (~80MB, downloads once)
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL_NAME)

texts = [c["text"] for c in all_chunks]
print(f"Generating embeddings for {len(texts)} chunks... (this may take 1-2 minutes)")
embeddings = embedder.encode(texts, show_progress_bar=True)
print("Embedding shape:", embeddings.shape)  # (num_chunks, vector_size)

# Set up a persistent Chroma vector store on disk
VECTOR_STORE_DIR = Path(__file__).resolve().parent.parent / "data" / "vector_store"
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
collection = client.get_or_create_collection(name="lecture_chunks")

ids = [f"{c['source']}_p{c['page']}_c{c['chunk_id']}" for c in all_chunks]
metadatas = [{"source": c["source"], "page": c["page"]} for c in all_chunks]

collection.add(
    ids=ids,
    embeddings=embeddings.tolist(),
    documents=texts,
    metadatas=metadatas,
)

print(f"Stored {collection.count()} chunks in Chroma at: {VECTOR_STORE_DIR}")


#Retrieval & Prompting
print("\n=== 2.4 Retrieval & Prompting ===")

import ollama as ollama_client

def retrieve(question, top_k=3):
    q_embedding = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=q_embedding, n_results=top_k)
    retrieved = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        retrieved.append({"text": doc, "source": meta["source"], "page": meta["page"]})
    return retrieved

def build_prompt(question, retrieved_chunks):
    context = "\n\n".join(
        f"[{c['source']} p.{c['page']}]: {c['text']}" for c in retrieved_chunks
    )
    prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Answer (cite the source in brackets like [filename p.X]):"""
    return prompt

def ask(question, top_k=3):
    retrieved = retrieve(question, top_k)
    prompt = build_prompt(question, retrieved)
    response = ollama_client.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response["message"]["content"]
    sources = list({f"{c['source']} p.{c['page']}" for c in retrieved})
    return answer, sources

# Test with one sample question
test_question = "What is predictive analytics?"
answer, sources = ask(test_question)
print(f"\nQ: {test_question}")
print(f"A: {answer}")
print(f"Sources: {sources}")

print("\n=== Testing retrieval with 10 sample questions ===")

test_questions = [
    "What is predictive analytics?",
    "Who is the instructor of this course?",
    "What topics are covered in this course?",
    "Why Ridge Regression Often Performs Better Than Ordinary Least Squares?",
    "Why Lasso Can Set Coefficients to Exactly Zero?",
    "What is the key advantage of Elastic Net regression?",  # kept as known-gap example
    "Given a regression model that has high training accuracy but performs poorly on new data, is the model suffering from high bias or high variance?",
    "What are the six stages of the CRISP-DM process?",
    "What is the difference between reducible and irreducible error?",
    "What is the difference between supervised and unsupervised learning?",
]

for q in test_questions:
    answer, sources = ask(q)
    print(f"\nQ: {q}")
    print(f"A: {answer}")
    print(f"Sources: {sources}")
# ============================================
# 2.6 Evaluation
# ============================================
#
# Results Table:
# | Question                              | Source(s)              | Correct? | Notes                                    |
# |----------------------------------------|-------------------------|----------|-------------------------------------------|
# | What is predictive analytics?         | Lecture_1.pdf p.1,3     | Yes      | Exact match to slide definition           |
# | Who is the instructor?                | Lecture_1.pdf p.1,2,14  | Yes      | Correct                                   |
# | What topics are covered?              | Lecture_1.pdf p.1,2     | Yes      | Matches syllabus exactly                  |
# | Why Ridge beats OLS?                  | Lecture_3.pdf p.2,3     | Yes      | Correct                                   |
# | Why Lasso sets coeffs to zero?        | Lecture_3.pdf p.4       | No       | LLM printed placeholder text, not content |
# | Elastic Net advantage?                | Lecture_3.pdf p.6,7     | No       | Same placeholder bug                      |
# | High bias or high variance?           | Lecture_2.pdf p.4,17    | Yes      | Correct                                   |
# | Six stages of CRISP-DM?               | Lecture_1.pdf p.29,3    | No       | Retrieval missed the right chunk          |
# | Reducible vs irreducible error?       | Lecture_1.pdf p.6,22    | Yes      | Correct                                   |
# | Supervised vs unsupervised learning?  | Lecture_1.pdf p.7,8     | Yes      | Correct                                   |
#
# Failure Analysis:
# Of the 10 test questions, 6 produced fully correct, grounded answers with
# accurate citations. Two failures were LLM-formatting related: for two
# Lasso/Elastic-Net questions, the model output the literal placeholder text
# "[filename p.X]" instead of a real answer, despite retrieving the correct
# source chunks -- a known limitation of smaller local LLMs like llama3.2
# when following structured citation instructions. One failure (CRISP-DM
# stages) was a genuine retrieval miss: although the answer existed in the
# corpus (Lecture_1, p.3), the relevant chunk was not returned in the top-3
# results, likely due to chunking splitting the CRISP-DM table away from
# strongly matching keywords. Mitigations: increasing top_k from 3 to 5,
# and simplifying the prompt's citation instruction.


print("\n=== 2.7 Export ===")

import json

config = {
    "embedding_model": EMBED_MODEL_NAME,
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
    "vector_store_path": str(VECTOR_STORE_DIR),
    "collection_name": "lecture_chunks",
    "top_k_default": 3,
    "llm_model": "llama3.2",
    "num_documents": len(df),
    "num_chunks": len(all_chunks),
}

config_path = VECTOR_STORE_DIR / "config.json"
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print(f"Saved config to: {config_path}")
print(json.dumps(config, indent=2))