"""Eudia backend Flask application factory."""
from __future__ import annotations

import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory

from .config import config_by_name
from .extensions import init_extensions
from .routes import register_blueprints


def create_app(config_name: str | None = None) -> Flask:
    """Application factory used by Flask CLI and tests."""
    load_dotenv()

    app = Flask(__name__)

    config_key = config_name or os.getenv("FLASK_CONFIG", "default")
    app.config.from_object(config_by_name[config_key])

    init_extensions(app)
    register_blueprints(app)

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # Add CORS headers to all responses
    @app.after_request
    def after_request(response):
        origin = request.headers.get("Origin")
        allow_origin = origin if origin else "*"

        response.headers["Access-Control-Allow-Origin"] = allow_origin
        existing_vary = response.headers.get("Vary")
        if existing_vary:
            vary_values = [value.strip() for value in existing_vary.split(",")]
            if "Origin" not in vary_values:
                vary_values.append("Origin")
            response.headers["Vary"] = ", ".join(vary_values)
        else:
            response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
        response.headers["Access-Control-Allow-Methods"] = "GET, PUT, POST, DELETE, OPTIONS"
        if origin:
            response.headers["Access-Control-Allow-Credentials"] = "true"
        return response

    @app.get("/health")
    def health_check():
        return jsonify({"status": "ok"})

    @app.get("/uploads/<path:filename>")
    def serve_upload(filename: str):
        return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=False)

    return app
