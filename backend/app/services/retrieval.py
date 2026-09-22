from sentence_transformers import SentenceTransformer
import chromadb

from backend.app.core.config import settings

# Loaded once when this module is first imported (not on every request)
_embedder = None
_collection = None

def load_retrieval_components():
    """Load the embedding model and vector store collection into memory."""
    global _embedder, _collection

    _embedder = SentenceTransformer(settings.embedding_model)

    client = chromadb.PersistentClient(path=settings.vector_store_path)
    _collection = client.get_collection(name=settings.collection_name)

    return _embedder, _collection

def retrieve(question: str, top_k: int = None):
    """Given a question, return the top_k most relevant chunks."""
    if _embedder is None or _collection is None:
        raise RuntimeError("Retrieval components not loaded. Call load_retrieval_components() first.")

    k = top_k or settings.top_k
    q_embedding = _embedder.encode([question]).tolist()
    results = _collection.query(query_embeddings=q_embedding, n_results=k)

    retrieved = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        retrieved.append({"text": doc, "source": meta["source"], "page": meta["page"]})
    return retrieved