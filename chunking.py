from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from services.chunking_service import run_all_strategies
from services.vector_store_service import VectorStoreService
from utils.file_parser import extract_text
import re

router = APIRouter()


def make_collection_name(filename: str) -> str:
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', filename.rsplit('.', 1)[0])[:50]
    return f"rag_{name}" if name else "rag_document"


@router.post("/analyze")
async def analyze_chunking(
    file: UploadFile = File(...),
    chunk_size: int = Form(500),
    auto_index: bool = Form(True),
):
    content = await file.read()

    if not content:
        raise HTTPException(400, "File is empty.")

    if len(content) > 10_000_000:
        raise HTTPException(400, "File too large. Max 10MB.")

    # Extract text from any format
    try:
        text, file_format = extract_text(content, file.filename)
    except ImportError as e:
        raise HTTPException(422, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(422, f"Could not parse file: {str(e)}")

    if not text.strip():
        raise HTTPException(400, "No text could be extracted from this file.")

    strategies = run_all_strategies(text)
    collection_name = make_collection_name(file.filename)

    indexed = False
    if auto_index:
        try:
            semantic_chunks = strategies["semantic"]["chunks"]
            VectorStoreService.store_chunks(collection_name, semantic_chunks)
            indexed = True
        except Exception as e:
            print(f"Warning: ChromaDB indexing failed: {e}")

    return {
        "filename": file.filename,
        "file_format": file_format,
        "collection_name": collection_name,
        "total_chars": len(text),
        "total_words": len(text.split()),
        "indexed": indexed,
        "strategies": strategies,
    }