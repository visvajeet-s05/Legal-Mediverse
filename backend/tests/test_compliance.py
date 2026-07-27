import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.core.security import create_access_token
from backend.app.core.database import get_db

class MockAsyncSession:
    def add(self, instance):
        pass

    async def commit(self):
        pass

    async def rollback(self):
        pass

    async def refresh(self, instance):
        if hasattr(instance, "id") and instance.id is None:
            instance.id = 1

    async def close(self):
        pass

async def override_get_db():
    yield MockAsyncSession()

app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)

token_counsel = create_access_token({"sub": "advocate_user", "role": "LEGAL_COUNSEL"})

def test_aca_appeal_generation():
    response = client.post(
        "/api/v1/law/appeal",
        data={
            "denial_letter": "Claim denied due to missing medical necessity documentation for MRI procedure.",
            "patient_name": "Jane Miller",
            "policy_id": "POL-999-MED",
            "insurance_provider": "Blue Shield Health",
            "claim_id": "CLM-44021"
        },
        headers={"Authorization": f"Bearer {token_counsel}"}
    )
    assert response.status_code == 200
    json_data = response.json()
    assert "appeal_letter" in json_data
    assert "2719" in json_data["applicable_statute"] or "2719" in json_data["appeal_letter"]
    assert "180-day" in json_data["appeal_letter"] or "180" in json_data["appeal_letter"]
    assert "Jane Miller" in json_data["appeal_letter"]

def test_aca_urgent_appeal_generation():
    response = client.post(
        "/api/v1/law/appeal",
        data={
            "denial_letter": "Urgent cardiac intervention denied.",
            "patient_name": "Marcus Vance",
            "policy_id": "POL-777-URG",
            "insurance_provider": "Aetna Healthcare",
            "is_urgent": "true"
        },
        headers={"Authorization": f"Bearer {token_counsel}"}
    )
    assert response.status_code == 200
    json_data = response.json()
    assert "appeal_letter" in json_data
    assert "72-hour" in json_data["appeal_letter"] or "72" in json_data["appeal_letter"] or "29 CFR" in json_data["appeal_letter"]

def test_contract_redline_audit():
    response = client.post(
        "/api/v1/law/redline",
        data={"contract_text": "Patient agrees to waive all rights to a jury trial and accepts sole responsibility for out-of-network balance billing."},
        headers={"Authorization": f"Bearer {token_counsel}"}
    )
    assert response.status_code == 200
    json_data = response.json()
    assert "overall_risk_score" in json_data
    assert "predatory_clauses" in json_data
    assert len(json_data["predatory_clauses"]) > 0

def test_hipaa_release_generation():
    response = client.post(
        "/api/v1/law/hipaa-request",
        data={
            "patient_name": "Alice Cooper",
            "dob": "10/12/1985",
            "provider_name": "Valley Care Clinic",
            "date_range": "2025 - 2026",
            "target_recipient": "Legal Mediverse Advocate"
        }
    )
    assert response.status_code == 200
    json_data = response.json()
    assert "hipaa_letter" in json_data
    assert "45 CFR" in json_data["hipaa_letter"]
    assert "thirty (30)" in json_data["hipaa_letter"] or "30" in json_data["hipaa_letter"]
    assert "Alice Cooper" in json_data["hipaa_letter"]
    assert "Valley Care Clinic" in json_data["hipaa_letter"]


def test_legal_pdf_structure():
    response = client.post(
        "/api/v1/law/appeal/pdf",
        data={
            "denial_letter": "Urgent cardiac intervention denied.",
            "patient_name": "Marcus Vance",
            "policy_id": "POL-777-URG",
            "insurance_provider": "Aetna Healthcare",
            "is_urgent": "true"
        },
        headers={"Authorization": f"Bearer {token_counsel}"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/pdf")
    assert len(response.content) > 100
