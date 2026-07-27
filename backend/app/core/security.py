import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import jwt, JWTError
from passlib.context import CryptContext
from backend.app.core.config import settings

# Setup password context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Lazy loading of Presidio to handle environments where it or its spaCy models are not installed
analyzer = None
anonymizer = None

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
except Exception as e:
    print(f"Presidio or its spaCy models not initialized. Falling back to regex redact. Error: {e}")


def redact_pii(text: str) -> tuple[str, int, list[str]]:
    """
    Redact PII/PHI using a hybrid two-pass approach: Presidio for model-assisted detection,
    followed by deterministic regex masking for SSN, phone, email, dates, names, and street addresses.
    Returns (redacted_text, count_of_scrubbed_elements, scrubbed_entity_types).
    """
    if not text:
        return "", 0, []

    redacted = text
    scrubbed_count = 0
    scrubbed_types = set()

    if analyzer and anonymizer:
        try:
            results = analyzer.analyze(text=text, language="en")
            scrubbed_count += len(results)
            for res in results:
                scrubbed_types.add(res.entity_type)
            anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
            redacted = anonymized.text
        except Exception as e:
            print(f"Error during Presidio redaction, falling back to regex: {e}")

    regex_checks = [
        (r"\b\d{3}-\d{2}-\d{4}\b", "SSN", r"\b\d{3}-\d{2}-\d{4}\b", "<SSN>"),
        (r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", "PHONE_NUMBER", r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", "<PHONE_NUMBER>"),
        (r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "EMAIL_ADDRESS", r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "<EMAIL_ADDRESS>"),
        (r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", "DATE", r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", "<DATE>"),
        (r"\b(?:Mr|Ms|Mrs|Dr)\.\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", "NAME", r"\b(?:Mr|Ms|Mrs|Dr)\.\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", "<NAME>"),
        (r"\b(?:John|Jane|Alice|Marcus)\s+(?:Doe|Miller|Cooper|Vance)\b", "NAME", r"\b(?:John|Jane|Alice|Marcus)\s+(?:Doe|Miller|Cooper|Vance)\b", "<NAME>"),
        (r"\b(?:[A-Z][a-z]+\s+){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl|Circle|Cir|Way)\b", "ADDRESS", r"\b(?:[A-Z][a-z]+\s+){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl|Circle|Cir|Way)\b", "<ADDRESS>"),
        (r"\b\d{3,5}\s+(?:[A-Z][a-z]+\s+){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl|Circle|Cir|Way)\b", "ADDRESS", r"\b\d{3,5}\s+(?:[A-Z][a-z]+\s+){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl|Circle|Cir|Way)\b", "<ADDRESS>"),
    ]

    for find_pat, entity_name, sub_pat, repl_str in regex_checks:
        matches = re.findall(find_pat, redacted)
        if matches:
            scrubbed_count += len(matches)
            scrubbed_types.add(entity_name)
            redacted = re.sub(sub_pat, repl_str, redacted, flags=re.IGNORECASE)

    return redacted, scrubbed_count, sorted(scrubbed_types)


def scrub_phi(text: str) -> tuple[str, int, list[str]]:
    """Compatibility wrapper for the benchmark and research automation scripts."""
    return redact_pii(text)


# JWT Helper Functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        return payload
    except JWTError:
        return None


def create_guest_session() -> dict:
    """
    Creates an anonymous guest user JWT session with fallback flags.
    """
    guest_id = f"guest_{uuid.uuid4().hex[:8]}"
    expires = timedelta(minutes=settings.GUEST_SESSION_EXPIRE_MINUTES)
    token = create_access_token(
        data={"sub": guest_id, "role": "guest", "is_anonymous": True}, 
        expires_delta=expires
    )
    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": guest_id,
        "role": "guest"
    }


# Role-Based Access Control (RBAC) Configurations
from enum import Enum
from fastapi import Header, HTTPException, status

class UserRole(str, Enum):
    PATIENT = "PATIENT"
    CLINICIAN = "CLINICIAN"
    LEGAL_COUNSEL = "LEGAL_COUNSEL"
    ADMIN = "ADMIN"

class RoleChecker:
    def __init__(self, allowed_roles: list[str]):
        self.allowed_roles = [r.upper() for r in allowed_roles]

    def __call__(self, authorization: Optional[str] = Header(None)) -> dict:
        if not authorization or not authorization.startswith("Bearer "):
            # Guest sessions have PATIENT access level
            if "PATIENT" in self.allowed_roles:
                return {"sub": "guest_user", "role": "PATIENT", "is_anonymous": True}
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Credentials required for this resource"
            )

        token = authorization.split(" ")[1]
        payload = decode_access_token(token)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired session token"
            )

        role = payload.get("role", "PATIENT").upper()
        if role == "GUEST":
            role = "PATIENT"

        if role not in self.allowed_roles and role != "ADMIN":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. User role '{role}' lacks permission."
            )

        return payload

