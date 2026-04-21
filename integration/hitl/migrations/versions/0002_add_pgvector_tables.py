"""
Alembic migration: enable pgvector extension and create vector storage tables.

Revision ID: 0002
Depends on: 0001

Tables created:
  session_embeddings  — per-session turn embeddings (replaces Redis flat lists)
  fraud_patterns      — knowledge-base of confirmed fraud patterns for RAG retrieval

Embedding dimension (768) matches the default model all-mpnet-base-v2.
If you change the model to one with a different output dimension, create a new
migration to ALTER the vector columns or drop/recreate the tables.
"""
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

# Embedding dimension — must match embeddings.dim in configs/classifiers.yaml
_DIM = 768


def upgrade():
    # Enable pgvector extension (idempotent)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ------------------------------------------------------------------
    # session_embeddings — stores per-session scope + turn embeddings
    # turn_n = 0  → declared task scope embedding
    # turn_n > 0  → request turn embeddings (newest has highest turn_n)
    # ------------------------------------------------------------------
    op.execute(f"""
        CREATE TABLE IF NOT EXISTS session_embeddings (
            id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id   TEXT NOT NULL,
            turn_n       INTEGER NOT NULL DEFAULT 0,
            content_hash TEXT,
            embedding    vector({_DIM}) NOT NULL,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
            expires_at   TIMESTAMPTZ NOT NULL
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_session_embeddings_session_turn
        ON session_embeddings (session_id, turn_n)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_session_embeddings_expires
        ON session_embeddings (expires_at)
    """)
    # IVFFlat cosine index — lists tuned for expected session cardinality
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_session_embeddings_vec
        ON session_embeddings USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 10)
    """)

    # ------------------------------------------------------------------
    # fraud_patterns — knowledge base for RAG retrieval
    # ------------------------------------------------------------------
    op.execute(f"""
        CREATE TABLE IF NOT EXISTS fraud_patterns (
            id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            pattern_text TEXT NOT NULL,
            pattern_type TEXT NOT NULL,
            embedding    vector({_DIM}) NOT NULL,
            severity     FLOAT NOT NULL DEFAULT 0.8,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_fraud_patterns_type
        ON fraud_patterns (pattern_type)
    """)
    # IVFFlat cosine index — lists tuned for seed corpus size
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_fraud_patterns_vec
        ON fraud_patterns USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 20)
    """)


def downgrade():
    op.execute("DROP TABLE IF EXISTS fraud_patterns")
    op.execute("DROP TABLE IF EXISTS session_embeddings")
    # Do NOT drop the vector extension — other tables may depend on it
