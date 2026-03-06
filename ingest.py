"""
ingest.py
Run this ONCE to load your data, chunk it, and save FAISS index to disk.
Supports: .txt, .csv files inside the /data folder.

Usage:
    python ingest.py
"""

import os
import pickle
from langchain_community.document_loaders import TextLoader, CSVLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = "data"          
FAISS_PATH  = "faiss_index"   
CHUNKS_PATH = "chunks.pkl"    
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2" 

# ── Load all documents from /data ─────────────────────────────────────────────
def load_documents():
    docs = []

    # Load all .txt files — encoding="utf-8" fixes Windows cp1252 error
    txt_loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}   # ← Windows fix
    )
    docs.extend(txt_loader.load())

    # Load all .csv files — encoding="utf-8" fixes Windows cp1252 error
    for file in os.listdir(DATA_DIR):
        if file.endswith(".csv"):
            csv_loader = CSVLoader(
                file_path=os.path.join(DATA_DIR, file),
                csv_args={"delimiter": ","},
                encoding="utf-8"               # ← Windows fix
            )
            docs.extend(csv_loader.load())

    print(f"✅ Loaded {len(docs)} documents from '{DATA_DIR}/'")
    return docs

# ── Chunk documents ───────────────────────────────────────────────────────────
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ Split into {len(chunks)} chunks")
    return chunks

# ── Build and save FAISS index ────────────────────────────────────────────────
def build_faiss_index(chunks):
    print("⏳ Building FAISS index (this may take a moment)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    faiss_store = FAISS.from_documents(chunks, embeddings)
    faiss_store.save_local(FAISS_PATH)
    print(f"✅ FAISS index saved to '{FAISS_PATH}/'")

# ── Save raw chunks for BM25 (BM25 works on raw text, not vectors) ────────────
def save_chunks(chunks):
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"✅ Chunks saved to '{CHUNKS_PATH}' for BM25")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.listdir(DATA_DIR):
        print(f"⚠️  No files found in '{DATA_DIR}/'. Add .txt or .csv files and re-run.")
    else:
        docs   = load_documents()
        chunks = chunk_documents(docs)
        build_faiss_index(chunks)
        save_chunks(chunks)
        print("\n🎉 Ingestion complete! Now run: streamlit run app.py")