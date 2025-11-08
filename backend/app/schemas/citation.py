"""Citation serialization helpers."""
from __future__ import annotations

from ..models import Citation


def citation_to_dict(citation: Citation) -> dict:
    return {
        "id": citation.id,
        "documentId": citation.document_id,
        "title": citation.title,
        "x": citation.x,
        "y": citation.y,
        "citations": citation.citations,
        "year": citation.year,
        "createdAt": citation.created_at.isoformat() if citation.created_at else None,
        "updatedAt": citation.updated_at.isoformat() if citation.updated_at else None,
    }
