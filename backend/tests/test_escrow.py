import os
os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.core.security import create_access_token
from backend.app.core.database import get_db
from backend.app.core.config import settings
from decimal import Decimal
from backend.app.models.models import CrowdfundCampaign


class MockScalarResult:
    def __init__(self, data):
        self._data = data

    def all(self):
        return self._data

    def scalar_one_or_none(self):
        return self._data[0] if self._data else None


class MockAsyncSession:
    def __init__(self):
        self.campaigns = []
        self._next_id = 1

    def add(self, instance):
        if hasattr(instance, "id") and instance.id is None:
            instance.id = self._next_id
            self._next_id += 1
        self.campaigns.append(instance)

    async def commit(self):
        pass

    async def rollback(self):
        pass

    async def refresh(self, instance):
        if hasattr(instance, "id") and instance.id is None:
            instance.id = self._next_id
            self._next_id += 1

    async def execute(self, statement):
        # Return a campaign that is already verified with low fraud risk
        # so that release-milestone tests can proceed past the fraud guard.
        verified_campaign = CrowdfundCampaign(
            id=1,
            creator_id=1,
            title="Burn Unit Emergency Surgery Fund",
            description="Emergency surgery for severe burn victim.",
            target_amount=Decimal("10000.00"),
            current_amount=Decimal("5000.00"),
            escrow_address="0xBurnUnitEscrowVault123",
            bill_verification_status="verified",
            total_bill_amount=Decimal("10000.00"),
        )
        # Attach fraud_risk_score attribute for the release-milestone guard
        verified_campaign.fraud_risk_score = 0.02
        return MockScalarResult(self.campaigns if self.campaigns else [verified_campaign])

    async def close(self):
        pass


async def override_get_db():
    yield MockAsyncSession()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)

token_clinician = create_access_token({"sub": "doctor_user", "role": "CLINICIAN"})


def test_list_and_create_campaigns():
    # Test GET campaigns (auto-seed)
    response = client.get("/api/v1/community/campaigns")
    assert response.status_code == 200
    json_data = response.json()
    assert isinstance(json_data, list)

    # Test POST campaign with on_chain fields
    post_res = client.post(
        "/api/v1/community/campaigns",
        data={
            "creator_id": 1,
            "title": "Burn Unit Emergency Surgery Fund",
            "description": "Emergency surgery for severe burn victim.",
            "target_amount": 10000.0,
            "escrow_address": "0xBurnUnitEscrowVault123",
            "on_chain_campaign_id": 1,
            "on_chain_tx_hash": "0xabc123def456abc123def456abc123def456abc123def456abc123def456abc1"
        }
    )
    assert post_res.status_code == 200
    post_json = post_res.json()
    assert post_json["title"] == "Burn Unit Emergency Surgery Fund"
    assert post_json.get("on_chain_campaign_id") == 1


def test_donate_to_campaign():
    res = client.post(
        "/api/v1/community/campaigns/1/donate",
        data={"amount": 250.0}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["amount_donated"] == 250.0
    assert "tx_hash" in data
    assert data["tx_hash"].startswith("0x")


def test_donate_to_campaign_with_tx_hash():
    """Test donation with a real client-provided tx_hash (simulates Wagmi flow)."""
    res = client.post(
        "/api/v1/community/campaigns/1/donate",
        data={
            "amount": 100.0,
            "tx_hash": "0xabc123def456abc123def456abc123def456abc123def456abc123def456abc1"
        }
    )
    assert res.status_code == 200
    data = res.json()
    assert data["amount_donated"] == 100.0
    assert data["tx_hash"] == "0xabc123def456abc123def456abc123def456abc123def456abc123def456abc1"
    assert "new_total" in data


def test_release_milestone_funds():
    res = client.post("/api/v1/community/campaigns/1/release-milestone")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "verified"
    assert "release_tx_hash" in data
    assert "fraud_risk_score" in data
    assert data["fraud_risk_score"] < 0.10


def test_ocr_fraud_risk_validation():
    response = client.post(
        "/api/v1/community/campaigns/1/verify-bill",
        files={"bill_image": ("medical_bill.jpg", b"\xFF\xD8\xFF\xE0\x00\x10JFIF", "image/jpeg")},
        headers={"Authorization": f"Bearer {token_clinician}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "ocr_verification" in data
    assert "fraud_risk_score" in data
    assert data["fraud_risk_score"] < 0.10
    assert "itemized_breakdown" in data
    assert "provider_name" in data
    assert "total_due" in data

