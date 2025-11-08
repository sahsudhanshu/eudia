"""Application extensions for the Eudia backend service."""
from __future__ import annotations

from flask_bcrypt import Bcrypt
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

# Instances are created without an app context and initialised during app startup.
db = SQLAlchemy()
migrate = Migrate()
bcrypt = Bcrypt()
jwt = JWTManager()
cors = CORS()


def init_extensions(app) -> None:
    """Bind core Flask extensions to the application instance."""
    db.init_app(app)
    migrate.init_app(app, db)
    bcrypt.init_app(app)
    jwt.init_app(app)
    # Allow all origins for dev convenience; tighten before production rollout
    cors.init_app(
        app,
        resources={
            r"/api/*": {
                "origins": "*",
                "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
            }
        },
        supports_credentials=True,
    )

    # Lazily import models to avoid circular imports when the module is imported.
    from . import models  # noqa: F401  # pylint: disable=unused-import

    @jwt.user_lookup_loader
    def _load_user_from_identity(_jwt_header, jwt_data):
        """Return the authenticated user for the current JWT identity."""
        from .models import User

        identity = jwt_data.get("sub")
        if identity is None:
            return None
        return User.query.filter_by(id=identity).one_or_none()
