"""Initialize the database."""
from app import create_app
from app.extensions import db
from app.models import User, Document, Citation, Chat, Message

app = create_app()

with app.app_context():
    print("Creating all database tables...")
    db.create_all()
    print("Database initialized successfully!")
    
    # Check if we need to seed
    if User.query.first() is None:
        print("Seeding database with demo user...")
        user = User(name="Demo User", email="demo@example.com")
        user.set_password("password123")
        db.session.add(user)
        db.session.commit()
        print("Demo user created: demo@example.com / password123")
    else:
        print("Database already has users")
