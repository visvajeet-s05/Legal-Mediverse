"""
Authentication API Routes
=========================
Handles user registration, login, and JWT token management.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, EmailStr
from backend.app.core.database import get_db
from backend.app.core.security import (
    create_access_token,
    get_password_hash,
    verify_password,
    create_guest_session,
)
from backend.app.models.models import User
from datetime import timedelta

router = APIRouter(prefix="/auth", tags=["Authentication"])

# ─── Request Schemas ────────────────────────────────────────────────────────


class RegisterRequest(BaseModel):
    full_name: str
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    role: str = "PATIENT"
    full_name: str = ""


# ─── Routes ─────────────────────────────────────────────────────────────────


@router.post("/register", response_model=AuthResponse)
async def register(
    req: RegisterRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Register a new user account.
    Returns a JWT access token on success.
    """
    # Check if email already exists
    result = await db.execute(select(User).where(User.email == req.email))
    existing_user = result.scalar_one_or_none()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    # Validate password length
    if len(req.password) < 8:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Password must be at least 8 characters long.",
        )

    try:
        # Create new user
        hashed_password = get_password_hash(req.password)
        new_user = User(
            full_name=req.full_name,
            email=req.email,
            password_hash=hashed_password,
        )
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)

        # Generate JWT token
        access_token = create_access_token(
            data={"sub": str(new_user.id), "role": "PATIENT"},
            expires_delta=timedelta(days=7),
        )

        return AuthResponse(
            access_token=access_token,
            user_id=str(new_user.id),
            role="PATIENT",
            full_name=new_user.full_name,
        )
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}",
        )


@router.post("/login", response_model=AuthResponse)
async def login(
    req: LoginRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Authenticate a user with email and password.
    Returns a JWT access token on success.
    """
    result = await db.execute(select(User).where(User.email == req.email))
    user = result.scalar_one_or_none()

    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    # Determine role (default to PATIENT, can be extended)
    role = "PATIENT"

    access_token = create_access_token(
        data={"sub": str(user.id), "role": role},
        expires_delta=timedelta(days=7),
    )

    return AuthResponse(
        access_token=access_token,
        user_id=str(user.id),
        role=role,
        full_name=user.full_name,
    )


@router.post("/guest-session")
async def guest_session():
    """
    Create an anonymous guest session with limited access.
    Returns a short-lived JWT token.
    """
    session = create_guest_session()
    return session

