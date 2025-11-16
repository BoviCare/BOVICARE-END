import asyncio
from typing import List, Dict, Any, Optional

from .vector_service import VectorService
from .rag import rerank_documents_with_similarity, generate_rag_response


class RAGService:
    """Wrapper para orquestrar busca híbrida e geração de resposta."""

    def __init__(self, top_k_default: int = 5):
        self.vector_service: Optional[VectorService] = None
        self.top_k_default = top_k_default

    async def startup(self):
        if self.vector_service is None:
            self.vector_service = VectorService()
            await self.vector_service.initialize()

    async def shutdown(self):
        if self.vector_service is not None:
            await self.vector_service.__aexit__(None, None, None)
            self.vector_service = None

    async def ask(self, query: str, top_k: int | None = None) -> Dict[str, Any]:
        if self.vector_service is None:
            await self.startup()

        top_k = top_k or self.top_k_default

        documents = await self.vector_service.hybrid_search(
            query=query,
            top_k=top_k * 2
        )

        if not documents:
            return {
                "answer": "Não encontrei informações relevantes na base de diagnóstico.",
                "sources": []
            }

        ranked = await rerank_documents_with_similarity(query, documents)
        selected = ranked[:top_k]
        answer = generate_rag_response(query, selected)

        sources: List[Dict[str, Any]] = []
        for doc in selected:
            sources.append({
                "disease_name": doc.get("disease_name"),
                "section_type": doc.get("section_type"),
                "page_number": doc.get("page_number"),
                "content_preview": doc.get("section_text", "")[:200]
            })

        return {
            "answer": answer,
            "sources": sources
        }

