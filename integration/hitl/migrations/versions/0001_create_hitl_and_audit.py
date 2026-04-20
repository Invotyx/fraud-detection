"""
Alembic migration: create hitl_queue and audit_log tables.

Revision ID: 0001
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # -----------------------------------------------------------------------
    # hitl_queue
    # -----------------------------------------------------------------------
    op.create_table(
        "hitl_queue",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("request_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.Column("unified_risk_score", sa.Float, nullable=False),
        sa.Column("classifier_scores", postgresql.JSONB, nullable=False),
        sa.Column("llm_response", postgresql.JSONB, nullable=True),
        sa.Column("decision_pending", sa.String(20), nullable=True),
        sa.Column("reviewed_by", sa.String(100), nullable=True),
        sa.Column("reviewed_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("reviewer_decision", sa.String(20), nullable=True),
        sa.Column("reviewer_notes", sa.Text, nullable=True),
        sa.Column("escalated_at", sa.TIMESTAMP(timezone=True), nullable=True),
    )
    op.create_index("ix_hitl_queue_reviewed_at", "hitl_queue", ["reviewed_at"])
    op.create_index("ix_hitl_queue_created_at", "hitl_queue", ["created_at"])

    # -----------------------------------------------------------------------
    # audit_log — append-only enforced by revoking UPDATE/DELETE after creation
    # -----------------------------------------------------------------------
    op.create_table(
        "audit_log",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("trace_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.Column("input_hash", sa.String(64), nullable=False),
        sa.Column("raw_input_hash", sa.String(64), nullable=False),
        sa.Column("classifier_scores", postgresql.JSONB, nullable=False),
        sa.Column("llm_response", postgresql.JSONB, nullable=True),
        sa.Column("unified_risk_score", sa.Float, nullable=False),
        sa.Column("decision", sa.String(20), nullable=False),
        sa.Column("flags", postgresql.JSONB, nullable=False),
        sa.Column("hitl_required", sa.Boolean,
                  nullable=False, server_default="false"),
        sa.Column("processing_time_ms", sa.Integer, nullable=True),
    )
    op.create_index("ix_audit_log_trace_id", "audit_log", ["trace_id"])
    op.create_index("ix_audit_log_created_at", "audit_log", ["created_at"])

    # Enforce append-only: revoke UPDATE and DELETE from the app user
    op.execute("REVOKE UPDATE, DELETE ON audit_log FROM PUBLIC")


def downgrade():
    op.drop_table("hitl_queue")
    op.drop_table("audit_log")
