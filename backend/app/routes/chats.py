"""Chat routes for storing model conversations."""
from __future__ import annotations

from datetime import datetime
from http import HTTPStatus

from flask import Blueprint, jsonify, request
from flask_jwt_extended import current_user, jwt_required

from ..extensions import db
from ..models import Chat, Document, Message
from ..schemas import chat_to_dict, message_to_dict

bp = Blueprint("chats", __name__, url_prefix="/api/chats")


def _ensure_document_access(document_id: str | None) -> bool:
    if not document_id:
        return True
    document = Document.query.filter_by(id=document_id, user_id=current_user.id).one_or_none()
    return document is not None


@bp.get("")
@jwt_required()
def list_chats():
    chats = Chat.query.filter_by(user_id=current_user.id).order_by(Chat.updated_at.desc()).all()
    return jsonify([chat_to_dict(chat, include_messages=False) for chat in chats])


@bp.post("")
@jwt_required()
def create_chat():
    payload = request.get_json(silent=True) or {}
    title = (payload.get("title") or "Conversation").strip() or "Conversation"
    document_id = payload.get("documentId")

    if not _ensure_document_access(document_id):
        return jsonify({"error": "Document not found"}), HTTPStatus.NOT_FOUND

    chat = Chat(title=title, user_id=current_user.id, document_id=document_id)
    db.session.add(chat)
    db.session.commit()

    return jsonify({"chat": chat_to_dict(chat)}), HTTPStatus.CREATED


@bp.get("/<chat_id>")
@jwt_required()
def get_chat(chat_id: str):
    chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).one_or_none()
    if chat is None:
        return jsonify({"error": "Chat not found"}), HTTPStatus.NOT_FOUND

    return jsonify({"chat": chat_to_dict(chat)})


@bp.post("/<chat_id>/messages")
@jwt_required()
def add_message(chat_id: str):
    chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).one_or_none()
    if chat is None:
        return jsonify({"error": "Chat not found"}), HTTPStatus.NOT_FOUND

    payload = request.get_json(silent=True) or {}
    role = (payload.get("role") or "").strip()
    content = (payload.get("content") or "").strip()
    metadata_payload = payload.get("metadata")

    if role not in {"user", "assistant", "system"}:
        return jsonify({"error": "role must be user, assistant, or system"}), HTTPStatus.BAD_REQUEST
    if not content:
        return jsonify({"error": "content is required"}), HTTPStatus.BAD_REQUEST

    now = datetime.utcnow()
    message = Message(
        chat_id=chat.id,
        role=role,
        content=content,
    metadata_json=metadata_payload,
        created_at=now,
        updated_at=now,
    )
    chat.updated_at = now
    db.session.add(message)
    db.session.commit()

    return jsonify({"message": message_to_dict(message)}), HTTPStatus.CREATED
