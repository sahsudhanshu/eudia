"""Citation routes scoped to a document."""
from __future__ import annotations

from datetime import datetime
from http import HTTPStatus

from flask import Blueprint, jsonify, request
from flask_jwt_extended import current_user, jwt_required

from ..extensions import db
from ..models import Citation, Document
from ..schemas import citation_to_dict

bp = Blueprint("citations", __name__, url_prefix="/api/documents/<document_id>/citations")


def _get_document_or_404(document_id: str) -> Document | None:
    return Document.query.filter_by(id=document_id, user_id=current_user.id).one_or_none()


@bp.get("")
@jwt_required()
def list_citations(document_id: str):
    document = _get_document_or_404(document_id)
    if document is None:
        return jsonify({"error": "Document not found"}), HTTPStatus.NOT_FOUND

    citations = Citation.query.filter_by(document_id=document.id).order_by(Citation.year.desc()).all()
    return jsonify([citation_to_dict(c) for c in citations])


@bp.post("")
@jwt_required()
def create_citation(document_id: str):
    document = _get_document_or_404(document_id)
    if document is None:
        return jsonify({"error": "Document not found"}), HTTPStatus.NOT_FOUND

    payload = request.get_json(silent=True) or {}

    title = (payload.get("title") or "").strip()
    x = payload.get("x")
    y = payload.get("y")
    count = payload.get("citations")
    year = payload.get("year")

    if not title:
        return jsonify({"error": "Title is required"}), HTTPStatus.BAD_REQUEST
    try:
        x_value = float(x)
        y_value = float(y)
    except (TypeError, ValueError):
        return jsonify({"error": "x and y must be numeric"}), HTTPStatus.BAD_REQUEST
    if not (0 <= x_value <= 100 and 0 <= y_value <= 100):
        return jsonify({"error": "x and y must be between 0 and 100"}), HTTPStatus.BAD_REQUEST

    try:
        citations_value = int(count)
        year_value = int(year)
    except (TypeError, ValueError):
        return jsonify({"error": "citations and year must be integers"}), HTTPStatus.BAD_REQUEST
    if citations_value < 0:
        return jsonify({"error": "citations must be non-negative"}), HTTPStatus.BAD_REQUEST

    citation = Citation(
        document_id=document.id,
        title=title,
        x=x_value,
        y=y_value,
        citations=citations_value,
        year=year_value,
    )
    db.session.add(citation)
    db.session.commit()

    return jsonify({"citation": citation_to_dict(citation)}), HTTPStatus.CREATED


@bp.delete("/<citation_id>")
@jwt_required()
def delete_citation(document_id: str, citation_id: str):
    document = _get_document_or_404(document_id)
    if document is None:
        return jsonify({"error": "Document not found"}), HTTPStatus.NOT_FOUND

    citation = Citation.query.filter_by(id=citation_id, document_id=document.id).one_or_none()
    if citation is None:
        return jsonify({"error": "Citation not found"}), HTTPStatus.NOT_FOUND

    db.session.delete(citation)
    db.session.commit()

    return jsonify({"success": True}), HTTPStatus.OK


@bp.put("/<citation_id>")
@jwt_required()
def update_citation(document_id: str, citation_id: str):
    document = _get_document_or_404(document_id)
    if document is None:
        return jsonify({"error": "Document not found"}), HTTPStatus.NOT_FOUND

    citation = Citation.query.filter_by(id=citation_id, document_id=document.id).one_or_none()
    if citation is None:
        return jsonify({"error": "Citation not found"}), HTTPStatus.NOT_FOUND

    payload = request.get_json(silent=True) or {}

    if "title" in payload:
        title = (payload.get("title") or "").strip()
        if not title:
            return jsonify({"error": "Title cannot be empty"}), HTTPStatus.BAD_REQUEST
        citation.title = title

    if "x" in payload:
        try:
            x_value = float(payload.get("x"))
        except (TypeError, ValueError):
            return jsonify({"error": "x must be numeric"}), HTTPStatus.BAD_REQUEST
        if not 0 <= x_value <= 100:
            return jsonify({"error": "x must be between 0 and 100"}), HTTPStatus.BAD_REQUEST
        citation.x = x_value

    if "y" in payload:
        try:
            y_value = float(payload.get("y"))
        except (TypeError, ValueError):
            return jsonify({"error": "y must be numeric"}), HTTPStatus.BAD_REQUEST
        if not 0 <= y_value <= 100:
            return jsonify({"error": "y must be between 0 and 100"}), HTTPStatus.BAD_REQUEST
        citation.y = y_value

    if "citations" in payload:
        try:
            citation_count = int(payload.get("citations"))
        except (TypeError, ValueError):
            return jsonify({"error": "citations must be an integer"}), HTTPStatus.BAD_REQUEST
        if citation_count < 0:
            return jsonify({"error": "citations must be non-negative"}), HTTPStatus.BAD_REQUEST
        citation.citations = citation_count

    if "year" in payload:
        try:
            year_value = int(payload.get("year"))
        except (TypeError, ValueError):
            return jsonify({"error": "year must be an integer"}), HTTPStatus.BAD_REQUEST
        citation.year = year_value

    citation.updated_at = datetime.utcnow()
    db.session.commit()

    return jsonify({"citation": citation_to_dict(citation)}), HTTPStatus.OK
