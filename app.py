# app.py
import os
import io
import uuid
import pickle
from typing import List, Dict, Any

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from pypdf import PdfReader
import openai

# ---------------------------
# Configuration
# ---------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY environment variable before running the app.")
openai.api_key = OPENAI_API_KEY

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Embedding model to use with OpenAI (change if you want different model)
EMBEDDING_MODEL = "text-embedding-3-small"  # stable embedding choice
LLM_MODEL = "gpt-3.5-turbo"                # chat model for answers

# ---------------------------
# Helper functions
# ---------------------------

def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    """Extract all text from a PDF (simple extraction)."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages_text = []
    for p in reader.pages:
        txt = p.extract_text()
        if txt:
            pages_text.append(txt)
    return "\n\n".join(pages_text)

def chunk_text(text: str, max_chars: int = 1000, overlap: int = 200) -> List[str]:
    """
    Very simple character-based chunking.
    - max_chars: how many characters per chunk (~approx tokens).
    - overlap: overlapping characters between chunks to preserve context.
    """
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chars, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if start >= length:
            break
    return chunks

def embed_texts_openai(texts: List[str]) -> List[List[float]]:
    """Call OpenAI embeddings API in batches and return list of vectors."""
    if not texts:
        return []
    embeddings: List[List[float]] = []
    batch_size = 50
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = openai.Embedding.create(model=EMBEDDING_MODEL, input=batch)
        batch_emb = [item["embedding"] for item in resp["data"]]
        embeddings.extend(batch_emb)
    return embeddings

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def get_top_k_chunks(query: str, chunks: List[str], embeddings: List[np.ndarray], top_k: int = 5):
    """Return top_k (index, score) pairs and their chunk strings for a query."""
    if not chunks or not embeddings:
        return []
    q_emb = embed_texts_openai([query])[0]
    q_vec = np.array(q_emb, dtype=np.float32)
    scores = [cosine_similarity(q_vec, emb) for emb in embeddings]
    idxs = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in idxs:
        results.append({"index": int(idx), "score": float(scores[idx]), "chunk": chunks[int(idx)]})
    return results

def save_store(doc_id: str, store: Dict[str, Any]):
    with open(os.path.join(DATA_DIR, f"{doc_id}_store.pkl"), "wb") as f:
        pickle.dump(store, f)

def load_store(doc_id: str) -> Dict[str, Any]:
    path = os.path.join(DATA_DIR, f"{doc_id}_store.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Document store for id {doc_id} not found.")
    with open(path, "rb") as f:
        return pickle.load(f)

# ---------------------------
# FastAPI app & endpoints
# ---------------------------

app = FastAPI(title="Simple RAG PDF Q&A Backend")

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file.
    - Extract text
    - Chunk it
    - Create embeddings for chunks
    - Persist store to disk and return a document_id
    """
    contents = await file.read()
    text = extract_text_from_pdf_bytes(contents)
    if not text:
        raise HTTPException(status_code=400, detail="No extractable text found in PDF.")
    chunks = chunk_text(text, max_chars=1200, overlap=200)
    embeddings_raw = embed_texts_openai(chunks)
    embeddings_np = [np.array(e, dtype=np.float32) for e in embeddings_raw]

    doc_id = str(uuid.uuid4())
    store = {
        "filename": file.filename,
        "chunks": chunks,
        "embeddings": embeddings_np,
    }
    save_store(doc_id, store)
    return {"document_id": doc_id, "num_chunks": len(chunks)}

@app.post("/query")
def query_document(payload: Dict[str, Any]):
    """
    Query the uploaded document:
    JSON body:
    {
      "document_id": "<id returned by /upload_pdf>",
      "question": "your question here",
      "top_k": 5   # optional
    }
    """
    doc_id = payload.get("document_id")
    question = payload.get("question")
    top_k = int(payload.get("top_k", 5))
    if not doc_id or not question:
        raise HTTPException(status_code=400, detail="document_id and question are required.")

    try:
        store = load_store(doc_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Document not found.")

    chunks: List[str] = store["chunks"]
    embeddings: List[np.ndarray] = store["embeddings"]

    top_chunks = get_top_k_chunks(question, chunks, embeddings, top_k=top_k)
    # Build context string for the LLM
    context_texts = []
    for i, item in enumerate(top_chunks):
        context_texts.append(f"---chunk {item['index']} (score: {item['score']:.4f})---\n{item['chunk']}\n")

    # Construct prompt/messages for ChatCompletion
    system_msg = {
        "role": "system",
        "content": "You are a helpful assistant. Use the provided document chunks to answer user's question. If the information is not present, say you don't know."
    }
    user_msg = {
        "role": "user",
        "content": f"Context:\n\n{''.join(context_texts)}\nQuestion: {question}\n\nAnswer the question using only the context above and be concise."
    }

    resp = openai.ChatCompletion.create(
        model=LLM_MODEL,
        messages=[system_msg, user_msg],
        max_tokens=512,
        temperature=0.2,
    )

    answer = resp["choices"][0]["message"]["content"].strip()
    return {
        "answer": answer,
        "sources": top_chunks
    }

@app.get("/health")
def health():
    return {"status": "ok"}
