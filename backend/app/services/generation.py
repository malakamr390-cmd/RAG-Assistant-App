import ollama as ollama_client

from backend.app.core.config import settings

def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[{c['source']} p.{c['page']}]: {c['text']}" for c in retrieved_chunks
    )
    prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Answer (cite the source in brackets like [filename p.X]):"""
    return prompt

def generate_answer(question: str, retrieved_chunks: list[dict]) -> str:
    prompt = build_prompt(question, retrieved_chunks)
    response = ollama_client.chat(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]