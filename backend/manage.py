"""Flask application management entry point."""
from __future__ import annotations

from flask.cli import FlaskGroup

from app import create_app
from app.extensions import db

app = create_app()
cli = FlaskGroup(create_app=create_app)


@cli.command("seed")
def seed_database() -> None:
    """Populate the database with a minimal default dataset."""
    from app.models import Document, User

    if User.query.first() is not None:
        print("Database already seeded")
        return

    user = User(name="Demo User", email="demo@example.com")
    user.set_password("password123")

    document = Document(
        title="Sample Judgment",
        file_url="https://example.com/sample.pdf",
        file_size=1024,
        status="completed",
        user=user,
    )

    db.session.add_all([user, document])
    db.session.commit()
    print("Seed data created")


if __name__ == "__main__":
    cli()
