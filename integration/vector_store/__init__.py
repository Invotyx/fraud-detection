from integration.vector_store.encoder import get_encoder, encode
from integration.vector_store.store import (
    init_vector_tables,
    upsert_scope_embedding,
    get_scope_embedding,
    append_turn_embedding,
    get_turn_history,
    search_fraud_patterns,
    purge_expired_session_embeddings,
)
from integration.vector_store.fraud_patterns import (
    is_knowledge_base_seeded,
    seed_knowledge_base,
    retrieve_similar_patterns,
    format_rag_context,
)

__all__ = [
    "get_encoder",
    "encode",
    "init_vector_tables",
    "upsert_scope_embedding",
    "get_scope_embedding",
    "append_turn_embedding",
    "get_turn_history",
    "search_fraud_patterns",
    "purge_expired_session_embeddings",
    "is_knowledge_base_seeded",
    "seed_knowledge_base",
    "retrieve_similar_patterns",
    "format_rag_context",
]
