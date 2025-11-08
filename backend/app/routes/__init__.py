"""Blueprint registration for the Eudia backend."""
from __future__ import annotations

from flask import Flask


def register_blueprints(app: Flask) -> None:
    from .auth import bp as auth_bp
    from .documents import bp as documents_bp
    from .citations import bp as citations_bp
    from .chats import bp as chats_bp
    from .ocr import bp as ocr_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(documents_bp)
    app.register_blueprint(citations_bp)
    app.register_blueprint(chats_bp)
    app.register_blueprint(ocr_bp)
