# app.py
import os
import io
import uuid
import pickle
from typing import List, Dict, Any

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader
from openai import AzureOpenAI
import re

# ---------------------------
# Configuration
# ---------------------------
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY = ""
CHAT_MODEL_URL = "https://legio-mdx0gh9f-eastus2.cognitiveservices.azure.com/"
CHAT_MODEL_VERSION = "2024-12-01-preview"

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Embedding model to use with OpenAI (change if you want different model)
EMBEDDING_MODEL = "text-embedding-3-large"  # stable embedding choice
LLM_MODEL = "gpt-4o-mini"                # chat model for answers

# ---------------------------
# Helper functions
# ---------------------------

class QueryRequest(BaseModel):
    document_id: str
    question: str

def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    """Extract all text from a PDF (simple extraction)."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages_text = []
    for p in reader.pages:
        txt = p.extract_text()
        if txt:
            pages_text.append(txt)
    return "\n\n".join(pages_text)

def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    """
    Chunk text by sentences instead of raw characters.
    Keeps chunks within max_chars, with optional overlap for context.
    """
    if not text:
        return []

    # Split by sentence-ish boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk += " " + sentence
        else:
            # Save the current chunk
            chunks.append(current_chunk.strip())

            # Start new chunk, include overlap
            if overlap > 0 and chunks:
                overlap_text = current_chunk[-overlap:]
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk = sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

def embed_texts_openai(texts: List[str]) -> List[List[float]]:
    """Call OpenAI embeddings API in batches and return list of vectors."""
    if not texts:
        return []

    embeddings: List[List[float]] = []
    batch_size = 50
    print("start embed_texts_openai")
    try:
        client = AzureOpenAI(
            azure_endpoint=CHAT_MODEL_URL,
            api_key=OPENAI_API_KEY,
            api_version=CHAT_MODEL_VERSION,
        )
        print("end embed_texts_openai")
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            resp = client.embeddings.create(
                model="text-embedding-3-large",  # or text-embedding-3-large
                input=batch
            )
            batch_emb = [item.embedding for item in resp.data]
            embeddings.extend(batch_emb)
        return embeddings
    except Exception as e:
        print(f"Error in embed_texts_openai: {e}")
        return []


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
        results.append({"index": int(idx), "score": float(
            scores[idx]), "chunk": chunks[int(idx)]})
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
    print("sart read")
    contents = await file.read()
    print("sart extract")
    text = extract_text_from_pdf_bytes(contents)
    print("end extract")
    if not text:
        raise HTTPException(
            status_code=400, detail="No extractable text found in PDF.")
    print("start chunk")
    chunks = chunk_text(text)
    print("end chunk")
    print("start embed")
    embeddings_raw = embed_texts_openai(chunks)
    print("end embed")
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
def query_document(payload: QueryRequest):
    """
    Query the uploaded document
    """
    doc_id = payload.document_id
    question = payload.question
    top_k = 5
    if not doc_id or not question:
        raise HTTPException(
            status_code=400, detail="document_id and question are required.")

    try:
        store = load_store(doc_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Document not found.")

    try:

        chunks: List[str] = store["chunks"]
        embeddings: List[np.ndarray] = store["embeddings"]

        top_chunks = get_top_k_chunks(
            question, chunks, embeddings, top_k=top_k)
        # Build context string for the LLM
        context_texts = []
        for i, item in enumerate(top_chunks):
            context_texts.append(
                f"---chunk {item['index']} (score: {item['score']:.4f})---\n{item['chunk']}\n")

        # Construct prompt/messages for ChatCompletion
        system_msg = {
            "role": "system",
            "content": "You are a helpful assistant. Use the provided document chunks to answer user's question. If the information is not present, say Please ask only questions related to the menu."
        }

        user_msg = {
            "role": "user",
            "content": f"Context:\n\n{''.join(context_texts)}\nQuestion: {question}\n\nAnswer the question using only the context above and be concise."
        }

        print("start chat completion")
        print("start client chat")

        client = AzureOpenAI(
            azure_endpoint=CHAT_MODEL_URL,
            api_key=OPENAI_API_KEY,
            api_version=CHAT_MODEL_VERSION,
        )
        print("end client chat")
        kargs = {
            "model": LLM_MODEL,
            "messages": [system_msg, user_msg],
            "temperature": 0,
        }

        # Make the actual API call - you need to unpack kargs with **
        response = client.chat.completions.create(**kargs)
        print("end chat completion")
        answer = response.choices[0].message.content
        return {
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8001, reload=True)
