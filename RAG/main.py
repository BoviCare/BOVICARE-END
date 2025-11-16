import os
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import openai

from vector_service import VectorService
from rag import rerank_documents_with_openai, generate_rag_response

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Services ---

vector_service: Optional[VectorService] = None
openai_client: Optional[openai.OpenAI] = None

# --- Application Lifespan (Startup/Shutdown) ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events."""
    global vector_service, openai_client
    
    logger.info("Initializing BoviCare RAG services...")
    async with VectorService() as vector_service:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error("OPENAI_API_KEY environment variable not set. Reranking and response generation will fail.")
        else:
            openai_client = openai.OpenAI(api_key=openai_api_key)
        
        logger.info("BoviCare RAG services initialized successfully")
        yield

# --- FastAPI App Initialization ---

app = FastAPI(
    title="BoviCare RAG API",
    description="A RAG API for Bovine Disease Information with Hybrid Search and AI Reranking",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Mount static files and serve HTML
app.mount("/static", StaticFiles(directory="templates"), name="static")

# --- API Request/Response Models ---

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    use_reranking: Optional[bool] = True

class QueryResponse(BaseModel):
    query: str
    response: str
    sources: List[Dict[str, Any]]

# --- API Endpoints ---

@app.get("/")
async def root():
    """Serve the main HTML page."""
    return FileResponse("templates/index.html")

@app.get("/api")
async def api_info():
    """API information endpoint."""
    return {
        "message": "BoviCare RAG API - Advanced Disease Information System",
        "version": "1.0.0",
        "description": "AI-powered search and Q&A system for bovine diseases",
        "endpoints": {
            "ask": "/ask",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "vector_service": vector_service is not None}

@app.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):
    """Processes a query about bovine diseases, performs hybrid search, reranks, and generates a response."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if not vector_service:
        raise HTTPException(status_code=503, detail="VectorService is not available")

    try:
        # Perform hybrid search
        logger.info(f"Processing query: {request.query}")
        search_results = await vector_service.hybrid_search(
            query=request.query,
            top_k=request.top_k * 2  # Get more results for reranking
        )
        
        if not search_results:
            return QueryResponse(
                query=request.query,
                response="I couldn't find any relevant information about that query in our bovine disease database.",
                sources=[]
            )

        # Rerank documents if requested
        if request.use_reranking and openai_client:
            logger.info("Reranking documents with OpenAI...")
            reranked_results = await rerank_documents_with_openai(
                query=request.query,
                documents=search_results,
                openai_client=openai_client
            )
            final_results = reranked_results[:request.top_k]
        else:
            final_results = search_results[:request.top_k]

        # Generate response
        logger.info("Generating RAG response...")
        response_text = await generate_rag_response(
            query=request.query,
            context_docs=final_results,
            openai_client=openai_client
        )

        # Prepare sources for response
        sources = []
        for doc in final_results:
            source = {
                "disease_name": doc.get("disease_name", "N/A"),
                "disease_type": doc.get("disease_type", "N/A"),
                "chunk_index": doc.get("chunk_index", "N/A"),
                "section_type": doc.get("section_type", "N/A"),
                "page_number": doc.get("page_number", "N/A"),
                "relevance_score": doc.get("score", 0.0),
                "content_preview": doc.get("section_text", "")[:200] + "..." if len(doc.get("section_text", "")) > 200 else doc.get("section_text", "")
            }
            sources.append(source)

        return QueryResponse(
            query=request.query,
            response=response_text,
            sources=sources
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
