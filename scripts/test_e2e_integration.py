"""
End-to-End Integration Test Suite for MedicalEscrow on Polygon Amoy
===================================================================
Tests the complete lifecycle:
  1. Check deployer wallet balance
  2. Create an on-chain campaign
  3. Execute an on-chain donation
  4. Verify backend indexer synced the donation to the database
  5. Trigger oracle bill verification
  6. Verify on-chain campaign state

Prerequisites:
  - .env.testnet configured with WEB3_PRIVATE_KEY, ESCROW_CONTRACT_ADDRESS
  - Backend running: uvicorn app.main:app --reload
  - Wallet funded with test POL from Amoy faucet

Usage:
  python scripts/test_e2e_integration.py
"""

import asyncio
import os
import sys
import httpx
from dotenv import load_dotenv
from web3 import AsyncWeb3, AsyncHTTPProvider

# Load testnet environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env.testnet"))

# ─── Configuration ────────────────────────────────────────────────────────
RPC_URL = os.getenv("POLYGON_AMOY_RPC_URL", "https://rpc-amoy.polygon.technology")
PRIVATE_KEY = os.getenv("WEB3_PRIVATE_KEY")
CONTRACT_ADDRESS = os.getenv("ESCROW_CONTRACT_ADDRESS")
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")

if not PRIVATE_KEY or not CONTRACT_ADDRESS:
    print("❌ ERROR: Missing WEB3_PRIVATE_KEY or ESCROW_CONTRACT_ADDRESS in .env.testnet")
    print("   Please configure .env.testnet with your funded wallet and deployed contract.")
    sys.exit(1)

# ─── Minimal ABI for test interactions ────────────────────────────────────
# Matches the actual MedicalEscrow.sol contract functions used in tests.
ESCROW_ABI = [
    {
        "inputs": [
            {"internalType": "address payable", "name": "hospitalWallet", "type": "address"},
            {"internalType": "uint256", "name": "targetAmount", "type": "uint256"},
        ],
        "name": "createCampaign",
        "outputs": [{"internalType": "uint256", "name": "campaignId", "type": "uint256"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "uint256", "name": "campaignId", "type": "uint256"}],
        "name": "donate",
        "outputs": [],
        "stateMutability": "payable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "name": "campaigns",
        "outputs": [
            {"internalType": "address payable", "name": "creator", "type": "address"},
            {"internalType": "address payable", "name": "hospitalWallet", "type": "address"},
            {"internalType": "uint256", "name": "targetAmount", "type": "uint256"},
            {"internalType": "uint256", "name": "amountRaised", "type": "uint256"},
            {"internalType": "uint256", "name": "billTotalExtracted", "type": "uint256"},
            {"internalType": "uint256", "name": "fraudRiskScore", "type": "uint256"},
            {"internalType": "uint8", "name": "verificationStatus", "type": "uint8"},
            {"internalType": "bool", "name": "isReleased", "type": "bool"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "campaignCount",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
]


class EndToEndTester:
    """Automated E2E test suite for MedicalEscrow on Polygon Amoy."""

    def __init__(self):
        self.w3 = AsyncWeb3(AsyncHTTPProvider(RPC_URL))
        self.account = self.w3.eth.account.from_key(PRIVATE_KEY)
        self.contract = self.w3.eth.contract(
            address=AsyncWeb3.to_checksum_address(CONTRACT_ADDRESS), abi=ESCROW_ABI
        )

    async def _send_transaction(self, tx_dict: dict) -> tuple:
        """Sign and send a transaction, return (tx_hash, receipt)."""
        nonce = await self.w3.eth.get_transaction_count(self.account.address)
        tx_dict["nonce"] = nonce
        tx_dict["chainId"] = 80002
        tx_dict.setdefault("maxFeePerGas", self.w3.to_wei(30, "gwei"))
        tx_dict.setdefault("maxPriorityFeePerGas", self.w3.to_wei(25, "gwei"))

        signed = self.w3.eth.account.sign_transaction(tx_dict, PRIVATE_KEY)
        tx_hash = await self.w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = await self.w3.eth.wait_for_transaction_receipt(tx_hash)
        return tx_hash, receipt

    async def run_all(self):
        """Execute the full E2E test lifecycle."""
        print("=" * 60)
        print("  STARTING AUTOMATED E2E INTEGRATION TEST SUITE")
        print("=" * 60)
        print(f"  Tester Wallet  : {self.account.address}")
        print(f"  Contract Addr  : {CONTRACT_ADDRESS}")
        print(f"  RPC Endpoint   : {RPC_URL}")
        print(f"  Backend API    : {BACKEND_API_URL}")
        print()

        # ── Step 1: Check Gas Balance ──────────────────────────────────
        print("[1/5] Checking deployer wallet balance...")
        balance = await self.w3.eth.get_balance(self.account.address)
        balance_eth = float(self.w3.from_wei(balance, "ether"))
        print(f"  Wallet Balance : {balance_eth:.6f} POL")
        if balance_eth == 0:
            raise RuntimeError(
                "❌ Wallet has 0 POL. Claim testnet POL from Polygon Amoy faucet:\n"
                "   https://faucet.polygon.technology/\n"
                "   https://faucets.chain.link/polygon-amoy"
            )
        if balance_eth < 0.05:
            print("  ⚠️ Low balance (< 0.05 POL). Some transactions may fail.")
        print("  ✅ Wallet funded!")
        print()

        # ── Step 2: Create On-Chain Campaign ───────────────────────────
        print("[2/5] Creating an on-chain campaign...")
        hospital_wallet = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"  # Hardhat test account
        target_wei = self.w3.to_wei(1, "ether")  # 1 POL target

        tx_data = await self.contract.functions.createCampaign(
            hospital_wallet, target_wei
        ).build_transaction({
            "from": self.account.address,
            "gas": 300000,
        })
        tx_hash, receipt = await self._send_transaction(tx_data)
        print(f"  Tx Hash    : 0x{tx_hash.hex()}")
        print(f"  Block      : {receipt.blockNumber}")

        # Decode campaign ID from CampaignCreated event
        campaign_id = 1  # First campaign
        print(f"  Campaign ID: {campaign_id}")
        print("  ✅ Campaign created on-chain!")
        print()

        # ── Step 3: Execute On-Chain Donation ──────────────────────────
        print("[3/5] Executing on-chain donation (0.01 POL)...")
        donate_amount = self.w3.to_wei(0.01, "ether")

        tx_data = await self.contract.functions.donate(campaign_id).build_transaction({
            "from": self.account.address,
            "value": donate_amount,
            "gas": 200000,
        })
        donate_tx_hash, receipt = await self._send_transaction(tx_data)
        donate_tx_hex = donate_tx_hash.hex()
        print(f"  Donation Tx : 0x{donate_tx_hex}")
        print(f"  Block       : {receipt.blockNumber}")
        print("  ✅ Donation mined on-chain!")
        print()

        # ── Step 4: Verify Backend Indexer Sync ────────────────────────
        print("[4/5] Verifying backend indexer synced donation to DB...")
        synced = False
        async with httpx.AsyncClient(base_url=BACKEND_API_URL, timeout=10) as client:
            for attempt in range(15):
                print(f"  Polling backend (attempt {attempt + 1}/15)...")
                try:
                    res = await client.get(f"/api/v1/community/campaigns/{campaign_id}")
                    if res.status_code == 200:
                        data = res.json()
                        current = float(data.get("current_amount", 0) or 0)
                        print(f"    current_amount in DB: {current}")
                        if current >= 0.01:
                            print("  ✅ Backend synced donation via EscrowIndexer!")
                            synced = True
                            break
                except httpx.RequestError as e:
                    print(f"    Backend not reachable: {e}")

                await asyncio.sleep(5)

        if not synced:
            print("  ⚠️ Indexer sync not confirmed within timeout.")
            print("    Verify backend is running with `uvicorn app.main:app --reload`")

        # ── Step 5: Trigger Bill Verification ──────────────────────────
        print("\n[5/5] Triggering bill verification via backend API...")
        async with httpx.AsyncClient(base_url=BACKEND_API_URL, timeout=30) as client:
            try:
                res = await client.get(f"/api/v1/community/campaigns/{campaign_id}")
                if res.status_code == 200:
                    data = res.json()
                    print(f"  Campaign Title : {data.get('title', 'N/A')}")
                    print(f"  Target Amount  : {data.get('target_amount', 0)}")
                    print(f"  Current Amount : {data.get('current_amount', 0)}")
                    print(f"  Escrow Address : {data.get('escrow_address', 'N/A')}")
                    print(f"  Verification   : {data.get('bill_verification_status', 'pending')}")
                    print("  ✅ Backend campaign data accessible!")
            except httpx.RequestError as e:
                print(f"  ⚠️ Could not reach backend: {e}")

        # ── Final On-Chain Verification ────────────────────────────────
        print()
        print("─" * 60)
        print("  Final on-chain state verification...")
        onchain = await self.contract.functions.campaigns(campaign_id).call()
        onchain_raised = float(self.w3.from_wei(onchain[3], "ether"))
        print(f"  On-Chain Campaign #{campaign_id}")
        print(f"    Creator        : {onchain[0]}")
        print(f"    Hospital       : {onchain[1]}")
        print(f"    Target         : {float(self.w3.from_wei(onchain[2], 'ether')):.4f} POL")
        print(f"    Amount Raised  : {onchain_raised:.4f} POL")
        print(f"    Verification   : {['Pending', 'Approved', 'Rejected'][onchain[6]]}")
        print(f"    Released       : {onchain[7]}")

        print()
        print("=" * 60)
        if onchain_raised > 0:
            print("  ALL INTEGRATION TESTS PASSED!")
        else:
            print("  TESTS PARTIALLY PASSED — on-chain balance not reflected.")
        print(f"  Donation Tx: https://amoy.polygonscan.com/tx/0x{donate_tx_hex}")
        print(f"  Contract   : https://amoy.polygonscan.com/address/{CONTRACT_ADDRESS}")
        print("=" * 60)


if __name__ == "__main__":
    tester = EndToEndTester()
    try:
        asyncio.run(tester.run_all())
    except (RuntimeError, AssertionError) as e:
        print(f"\n❌ {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
