import os
import sys
import glob
import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def load_pdf(pdf_path):
    print(f"  Loading: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"  Pages loaded: {len(docs)}")
    return docs


def find_pdf():
    if len(sys.argv) > 1:
        pdf = sys.argv[1]
        if os.path.exists(pdf):
            return pdf
    pdfs = glob.glob("*.pdf")
    if pdfs:
        print(f"  Auto-found PDF: {pdfs[0]}")
        return pdfs[0]
    print("ERROR: No PDF found.")
    sys.exit(1)


def ingest_approach1(pdf_path):
    """Returns (vectorstore, num_chunks, chunks) — matches what app.py expects."""
    print("\n" + "="*50)
    print("  APPROACH 1 — Basic RAG (chunk=400)")
    print("="*50)
    docs = load_pdf(pdf_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    chunks = splitter.split_documents(docs)
    print(f"  Chunks created: {len(chunks)}")
    embeddings = get_embeddings()
    vs = FAISS.from_documents(chunks, embeddings)
    os.makedirs("storage/approach1/faiss_index", exist_ok=True)
    vs.save_local("storage/approach1/faiss_index")
    print(f"  Saved to storage/approach1/faiss_index")
    return vs, len(chunks), chunks


def ingest_approach2(pdf_path):
    """Returns (vectorstore, num_chunks, chunks) — matches what app.py expects."""
    print("\n" + "="*50)
    print("  APPROACH 2 — MMR (chunk=600)")
    print("="*50)
    docs = load_pdf(pdf_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
    chunks = splitter.split_documents(docs)
    print(f"  Chunks created: {len(chunks)}")
    embeddings = get_embeddings()
    vs = FAISS.from_documents(chunks, embeddings)
    os.makedirs("storage/approach2/faiss_index", exist_ok=True)
    vs.save_local("storage/approach2/faiss_index")
    print(f"  Saved to storage/approach2/faiss_index")
    return vs, len(chunks), chunks


def ingest_approach3(pdf_path):
    """Returns (vectorstore, num_chunks, chunks) — matches what app.py expects."""
    print("\n" + "="*50)
    print("  APPROACH 3 — Hybrid (chunk=800)")
    print("="*50)
    docs = load_pdf(pdf_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    chunks = splitter.split_documents(docs)
    print(f"  Chunks created: {len(chunks)}")
    embeddings = get_embeddings()
    vs = FAISS.from_documents(chunks, embeddings)
    os.makedirs("storage/approach3/faiss_index", exist_ok=True)
    vs.save_local("storage/approach3/faiss_index")
    try:
        from rank_bm25 import BM25Okapi
        texts = [doc.page_content for doc in chunks]
        tokenized = [t.split() for t in texts]
        bm25 = BM25Okapi(tokenized)
        with open("storage/approach3/bm25_index.pkl", "wb") as f:
            pickle.dump(bm25, f)
        print(f"  BM25 index saved")
    except Exception:
        pass
    print(f"  Saved to storage/approach3/faiss_index")
    return vs, len(chunks), chunks


if __name__ == "__main__":
    pdf = find_pdf()
    print(f"\nProcessing PDF: {pdf}")
    _, c1, _ = ingest_approach1(pdf)
    _, c2, _ = ingest_approach2(pdf)
    _, c3, _ = ingest_approach3(pdf)
    print("\n" + "="*50)
    print("  ALL DONE!")
    print(f"  Approach 1: {c1} chunks")
    print(f"  Approach 2: {c2} chunks")
    print(f"  Approach 3: {c3} chunks")
    print("="*50)