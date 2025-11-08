# Eudia Backend Service

This folder contains the Flask backend that powers authentication, document storage, citation management and chat persistence for the Eudia platform.

## Prerequisites

- Python 3.11+
- PostgreSQL (recommended) or another database supported by SQLAlchemy
- Virtual environment tooling (`venv`, `conda`, etc.)

## Quick start

1. Copy `.env.example` to `.env` and update the connection string and secret key:

   ```bash
   cp .env.example .env
   ```

2. Install dependencies and run the development server:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows PowerShell
   pip install -r requirements.txt
   flask db upgrade
   flask run --port 8000
   ```

   The API will be available at `http://127.0.0.1:8000`.

3. (Optional) Seed demo data:

   ```bash
   flask seed
   ```

Set `CORS_ORIGINS` in `.env` if the frontend runs on a different origin (comma-separated list, defaults to local Next.js dev URLs).

## Project structure

```
backend/
├── app/
│   ├── __init__.py          # Application factory
│   ├── config.py            # Environment specific settings
│   ├── extensions.py        # SQLAlchemy, JWT, Bcrypt instances
│   ├── models/              # SQLAlchemy models
│   ├── routes/              # Blueprint routes (auth, documents, chats, ...)
│   ├── schemas/             # Response serializers
│   └── services/            # Business logic helpers
├── manage.py                # Flask CLI entry point
├── requirements.txt         # Python dependencies
└── README.md
```

## Database migrations

The project uses `Flask-Migrate` (Alembic) for schema management. Typical workflow:

```bash
flask db init       # only once
flask db migrate -m "Add chats"
flask db upgrade
```

## Authentication

Authentication relies on JWT access and refresh tokens returned by `/api/auth/login`. The frontend stores the `accessToken` for API requests and can call `/api/auth/refresh` to obtain a fresh token when required.

## Available endpoints

- `POST /api/auth/register`
- `POST /api/auth/login`
- `POST /api/auth/refresh`
- `GET /api/auth/me`
- `POST /api/auth/logout`
- `GET /api/documents`
- `POST /api/documents`
- `GET /api/documents/<id>`
- `PUT /api/documents/<id>`
- `DELETE /api/documents/<id>`
- `GET /api/documents/<id>/citations`
- `POST /api/documents/<id>/citations`
- `DELETE /api/documents/<id>/citations/<citation_id>`
- `GET /api/chats`
- `POST /api/chats`
- `GET /api/chats/<id>`
- `POST /api/chats/<id>/messages`

Add additional routes (e.g. document upload, model inference) beside this structure as new features are implemented.
