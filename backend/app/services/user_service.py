"""
User Service

Handles user operations through the repository layer.
This demonstrates the correct Service → Repository → monk/ architecture.
"""

import uuid
from datetime import datetime, timezone

from app.core.config import get_settings
from app.models.user import User, UserMetadata, UserPreferences
from app.repositories.user_repository import UserRepository


class UserService:
    """Service for handling user operations through repository layer."""

    def __init__(self):
        self.user_repo = UserRepository()
        self.settings = get_settings()

    async def create_user(
        self, username: str, email: str, hashed_password: str, role: str = "researcher"
    ) -> User:
        """Create a new user."""

        # Check if user already exists
        existing_user = self.user_repo.find_by_username(username)
        if existing_user:
            raise ValueError(f"User with username '{username}' already exists")

        existing_email = self.user_repo.find_by_email(email)
        if existing_email:
            raise ValueError(f"User with email '{email}' already exists")

        # Create new user
        user = User(
            id=str(uuid.uuid4()),
            username=username,
            email=email,
            hashed_password=hashed_password,
            role=role,
            status="active",
            created_at=datetime.now(timezone.utc),
            last_login=None,
            preferences=UserPreferences(),
            metadata=UserMetadata(),
        )

        # Save through repository
        saved_user = self.user_repo.create(user)
        return saved_user

    async def get_user_by_id(self, user_id: str) -> User | None:
        """Get user by ID."""
        return self.user_repo.get_by_id(user_id)

    async def get_user_by_username(self, username: str) -> User | None:
        """Get user by username."""
        return self.user_repo.find_by_username(username)

    async def get_user_by_email(self, email: str) -> User | None:
        """Get user by email."""
        return self.user_repo.find_by_email(email)

    async def list_users(self) -> list[User]:
        """Get all users."""
        return self.user_repo.get_all()

    async def update_user(self, user_id: str, **kwargs) -> User | None:
        """Update user information."""
        user = self.user_repo.get_by_id(user_id)
        if not user:
            return None

        # Update fields
        for field, value in kwargs.items():
            if hasattr(user, field):
                setattr(user, field, value)

        # Save through repository
        return self.user_repo.update(user_id, user)

    async def update_last_login(self, user_id: str) -> User | None:
        """Update user's last login time."""
        return await self.update_user(user_id, last_login=datetime.now(timezone.utc))

    async def deactivate_user(self, user_id: str) -> bool:
        """Deactivate a user."""
        result = await self.update_user(user_id, status="inactive")
        return result is not None

    async def activate_user(self, user_id: str) -> bool:
        """Activate a user."""
        result = await self.update_user(user_id, status="active")
        return result is not None

    async def delete_user(self, user_id: str) -> bool:
        """Delete a user."""
        return self.user_repo.delete(user_id)

    async def get_active_users(self) -> list[User]:
        """Get all active users."""
        all_users = self.user_repo.get_all()
        return [user for user in all_users if user.status == "active"]

    async def get_users_by_role(self, role: str) -> list[User]:
        """Get users by role."""
        all_users = self.user_repo.get_all()
        return [user for user in all_users if user.role == role]
