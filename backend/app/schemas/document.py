"""Document serialization helpers."""
from __future__ import annotations

from ..models import Document
from .citation import citation_to_dict


def document_to_dict(document: Document, *, include_citations: bool = False, include_ocr: bool = False) -> dict:
    payload = {
        "id": document.id,
        "title": document.title,
        "fileUrl": document.file_url,
        "fileSize": document.file_size,
        "uploadDate": document.upload_date.isoformat() if document.upload_date else None,
        "status": document.status,
        "userId": document.user_id,
        "createdAt": document.created_at.isoformat() if document.created_at else None,
        "updatedAt": document.updated_at.isoformat() if document.updated_at else None,
    }

    if include_citations:
        payload["citations"] = [citation_to_dict(c) for c in document.citations]
    
    if include_ocr:
        payload["ocrMetadata"] = document.ocr_metadata
        payload["hasOcrText"] = bool(document.ocr_text)
        payload["hasCitationGraph"] = bool(document.citation_graph)

    return payload
