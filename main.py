from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from routes.chunking import router as chunking_router
from routes.prompting import router as prompting_router
from routes.rag import router as rag_router
from services.embedding_service import EmbeddingService


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load embedding model once at startup
    print("🚀 Loading embedding model...")
    EmbeddingService.get_instance()
    print("✅ Embedding model ready")
    yield
    print("🛑 Shutting down")


app = FastAPI(
    title="RAG Explorer API",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chunking_router,  prefix="/api/chunking",  tags=["Chunking"])
app.include_router(prompting_router, prefix="/api/prompting", tags=["Prompting"])
app.include_router(rag_router,       prefix="/api/rag",       tags=["RAG"])


@app.get("/")
def root():
    return {"message": "RAG Explorer v2 API is running ✅"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)