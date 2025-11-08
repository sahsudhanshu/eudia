"""Document management routes."""
from __future__ import annotations

import uuid
from http import HTTPStatus
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request
from flask_jwt_extended import current_user, jwt_required
from werkzeug.utils import secure_filename

from ..extensions import db
from ..models import Document
from ..schemas import document_to_dict

bp = Blueprint("documents", __name__, url_prefix="/api/documents")


@bp.get("")
@jwt_required()
def list_documents():
    query = Document.query.filter_by(user_id=current_user.id)

    search = request.args.get("search")
    if search:
        query = query.filter(Document.title.ilike(f"%{search}%"))

    limit = min(int(request.args.get("limit", 20)), 100)
    offset = max(int(request.args.get("offset", 0)), 0)

    documents = (
        query.order_by(Document.upload_date.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return jsonify([document_to_dict(document) for document in documents])


@bp.post("")
@jwt_required()
def create_document():
    payload = request.get_json(silent=True) or {}

    title = (payload.get("title") or "").strip()
    file_url = (payload.get("fileUrl") or "").strip()
    file_size = payload.get("fileSize")

    if not title:
        return jsonify({"error": "Title is required"}), HTTPStatus.BAD_REQUEST
    if not file_url:
        return jsonify({"error": "File URL is required"}), HTTPStatus.BAD_REQUEST
    if file_size is None:
        return jsonify({"error": "File size is required"}), HTTPStatus.BAD_REQUEST

    try:
        file_size_value = int(file_size)
    except (TypeError, ValueError) as exc:
        return (
            jsonify({"error": "File size must be a positive integer"}),
            HTTPStatus.BAD_REQUEST,
        )
    if file_size_value <= 0:
        return (
            jsonify({"error": "File size must be a positive integer"}),
            HTTPStatus.BAD_REQUEST,
        )

    document = Document(
        title=title,
        file_url=file_url,
        file_size=file_size_value,
        user_id=current_user.id,
    )
    db.session.add(document)
    db.session.commit()

    return jsonify({"document": document_to_dict(document)}), HTTPStatus.CREATED


@bp.post("/upload")
@jwt_required()
def upload_document_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), HTTPStatus.BAD_REQUEST

    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify({"error": "No file selected"}), HTTPStatus.BAD_REQUEST

    if file.mimetype != "application/pdf":
        return jsonify({"error": "Only PDF files are allowed"}), HTTPStatus.BAD_REQUEST

    if file.content_length and file.content_length > current_app.config["MAX_CONTENT_LENGTH"]:
        return jsonify({"error": "File size exceeds limit"}), HTTPStatus.BAD_REQUEST

    upload_folder = Path(current_app.config["UPLOAD_FOLDER"])
    upload_folder.mkdir(parents=True, exist_ok=True)

    safe_name = secure_filename(file.filename) or "document.pdf"
    stored_name = f"{uuid.uuid4().hex}_{safe_name}"
    storage_path = upload_folder / stored_name
    file.save(storage_path)

    document = Document(
        title=safe_name.rsplit(".", 1)[0],
        file_url=f"/uploads/{stored_name}",
        file_size=storage_path.stat().st_size,
        status="processing",
        user_id=current_user.id,
    )

    db.session.add(document)
    db.session.commit()

    # Process OCR and citation graph asynchronously (or synchronously for now)
    try:
        import sys
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        sys.path.insert(0, str(project_root))
        
        from lexai.agents.ocr_agent import process_pdf
        from app.services.citation_graph_service import generate_citation_graph
        
        # Run OCR
        ocr_result = process_pdf(str(storage_path))
        
        # Cache OCR results
        document.ocr_text = ocr_result.get("raw_text", "")
        document.ocr_metadata = {
            "title": ocr_result.get("title"),
            "citations": ocr_result.get("citations", []),
            "articles": ocr_result.get("articles", []),
        }
        
        # Generate citation graph
        if document.ocr_text:
            citation_graph = generate_citation_graph(
                document.ocr_text, 
                document.title
            )
            document.citation_graph = citation_graph
        
        document.status = "completed"
        
    except Exception as e:
        current_app.logger.error(f"OCR/Graph processing failed: {str(e)}")
        document.status = "error"
    
    db.session.commit()

    return (
        jsonify(
            {
                "document": document_to_dict(document),
                "message": "File uploaded and processed successfully",
            }
        ),
        HTTPStatus.CREATED,
    )


@bp.get("/<document_id>")
@jwt_required()
def get_document(document_id: str):
    document = Document.query.filter_by(id=document_id, user_id=current_user.id).one_or_none()
    if document is None:
        return jsonify({"error": "Document not found"}), HTTPStatus.NOT_FOUND

    return jsonify({"document": document_to_dict(document, include_citations=True)})


@bp.put("/<document_id>")
@jwt_required()
def update_document(document_id: str):
    document = Document.query.filter_by(id=document_id, user_id=current_user.id).one_or_none()
    if document is None:
        return jsonify({"error": "Document not found"}), HTTPStatus.NOT_FOUND

    payload = request.get_json(silent=True) or {}

    title = payload.get("title")
    status = payload.get("status")

    if title is not None:
        cleaned_title = title.strip()
        if not cleaned_title:
            return jsonify({"error": "Title cannot be empty"}), HTTPStatus.BAD_REQUEST
        document.title = cleaned_title

    if status is not None:
        cleaned_status = status.strip()
        if not cleaned_status:
            return jsonify({"error": "Status cannot be empty"}), HTTPStatus.BAD_REQUEST
        document.update_status(cleaned_status)

    db.session.commit()

    return jsonify({"document": document_to_dict(document)}), HTTPStatus.OK


@bp.delete("/<document_id>")
@jwt_required()
def delete_document(document_id: str):
    document = Document.query.filter_by(id=document_id, user_id=current_user.id).one_or_none()
    if document is None:
        return jsonify({"error": "Document not found"}), HTTPStatus.NOT_FOUND

    db.session.delete(document)
    db.session.commit()

    return jsonify({"success": True}), HTTPStatus.OK
