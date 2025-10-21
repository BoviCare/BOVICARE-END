import logging
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


async def rerank_documents_with_similarity(
    query: str,
    documents: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if not documents:
        return []

    query_embedding = embedding_model.encode(query, normalize_embeddings=True)
    scored_docs: List[tuple[float, Dict[str, Any]]] = []
    for doc in documents:
        text = doc.get("section_text", "")
        if not text:
            continue
        doc_embedding = embedding_model.encode(text, normalize_embeddings=True)
        score = float(util.cos_sim(query_embedding, doc_embedding))
        scored_docs.append((score, doc))

    scored_docs.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored_docs]


def generate_rag_response(
    query: str,
    context_docs: List[Dict[str, Any]]
) -> str:
    if not context_docs:
        return "Não encontrei informações relevantes para responder a essa pergunta na base de diagnóstico." 

    response_parts: List[str] = [
        f"Resumo das principais informações encontradas sobre '{query}':"
    ]

    for doc in context_docs[:3]:
        name = doc.get("disease_name", "Doença não identificada")
        section = doc.get("section_type", "seção")
        snippet = doc.get("section_text", "")
        if len(snippet) > 400:
            snippet = snippet[:400].rsplit(" ", 1)[0] + "..."
        response_parts.append(f"• Fonte: {name} ({section})\n  {snippet}")

    response_parts.append("Recomendo avaliar as fontes citadas acima para detalhes adicionais.")
    return "\n\n".join(response_parts)
