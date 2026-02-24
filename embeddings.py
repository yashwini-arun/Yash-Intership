from langchain_community.embeddings import HuggingFaceEmbeddings

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def embed_text(text):
    """Embed a single text string — used by app.py for vector visualizer."""
    embeddings = get_embeddings()
    return embeddings.embed_query(text)

def embed_documents(texts):
    """Embed a list of texts into vectors."""
    embeddings = get_embeddings()
    return embeddings.embed_documents(texts)