"""Authentication related business logic."""
from __future__ import annotations

from sqlalchemy.exc import IntegrityError

from ..extensions import bcrypt, db
from ..models import User


def create_user(*, name: str, email: str, password: str) -> User:
    user = User(name=name, email=email)
    user.set_password(password)

    db.session.add(user)
    try:
        db.session.commit()
    except IntegrityError as exc:  # email uniqueness violation
        db.session.rollback()
        raise ValueError("Email already registered") from exc

    return user


def authenticate_user(*, email: str, password: str) -> User | None:
    user = User.query.filter_by(email=email).one_or_none()
    if user is None:
        return None
    if not user.check_password(password):
        return None
    return user


def get_user_by_id(user_id: str) -> User | None:
    if not user_id:
        return None
    return User.query.filter_by(id=user_id).one_or_none()
