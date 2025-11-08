"""Chat serialization helpers."""
from __future__ import annotations

from ..models import Chat, Message
from .document import document_to_dict


def chat_to_dict(chat: Chat, *, include_messages: bool = True) -> dict:
    payload = {
        "id": chat.id,
        "title": chat.title,
        "userId": chat.user_id,
        "documentId": chat.document_id,
        "createdAt": chat.created_at.isoformat() if chat.created_at else None,
        "updatedAt": chat.updated_at.isoformat() if chat.updated_at else None,
    }

    if chat.document:
        payload["document"] = document_to_dict(chat.document)

    if include_messages:
        payload["messages"] = [message_to_dict(message) for message in chat.messages]

    return payload


def message_to_dict(message: Message) -> dict:
    return {
        "id": message.id,
        "chatId": message.chat_id,
        "role": message.role,
        "content": message.content,
        "metadata": message.metadata_json,
        "createdAt": message.created_at.isoformat() if message.created_at else None,
        "updatedAt": message.updated_at.isoformat() if message.updated_at else None,
    }
