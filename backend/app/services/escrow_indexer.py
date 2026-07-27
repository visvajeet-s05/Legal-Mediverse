"""
Production Escrow Event Indexer
================================
Background service that polls Polygon blockchain for MedicalEscrow contract events,
syncs them to the local database, and exposes operational metrics.

Key Production Features:
  - DB-backed persistent state tracking (survives restarts)
  - Chain reorg safety via configurable confirmation blocks
  - Multi-RPC provider fallback for high availability
  - Prometheus metrics for Grafana dashboards
  - Sentry error tracking
  - Slack alerting on critical failures
  - Structured JSON logging for CloudWatch/Loki

Events indexed:
  - DonationReceived        -> updates campaign.current_amount, creates Donation record
  - BillVerificationFulfilled -> updates bill_verification_status
  - FundsReleased           -> sets is_released = True
  - RefundIssued            -> logs refund event
"""

import asyncio
import time
from decimal import Decimal
from typing import Any, List, Optional

import httpx
import sentry_sdk
from web3 import AsyncWeb3
from web3.exceptions import Web3RPCError
from web3.providers.async_rpc import AsyncHTTPProvider
from web3.types import LogReceipt, RPCEndpoint, RPCResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.core.database import AsyncSessionLocal
from backend.app.core.logging_config import escrow_indexer_logger as logger
from backend.app.models.models import (
    CrowdfundCampaign,
    Donation,
    AuditLedger,
    IndexerState,
    STATE_KEY_LAST_BLOCK,
)
from backend.app.services.indexer_metrics import (
    INDEXER_CURRENT_BLOCK,
    CHAIN_TIP_BLOCK,
    INDEXER_LAG_BLOCKS,
    EVENTS_PROCESSED_TOTAL,
    INDEXER_ERRORS_TOTAL,
    PROCESSING_TIME,
)

# ─── Constants ─────────────────────────────────────────────────────────────

BLOCK_BATCH_SIZE = 999          # Safe batch size for standard RPC limits
POLL_INTERVAL_SECONDS = 12      # Poll every 12 seconds (one Polygon block on average)
CONFIRMATION_BLOCKS = 32        # Wait 32 confirmations before processing (reorg safety)
SLACK_LAG_WARNING_THRESHOLD = 50
SLACK_LAG_CRITICAL_THRESHOLD = 200
SLACK_WEBHOOK_URL = getattr(settings, "SLACK_ALERT_WEBHOOK_URL", None)

# ─── Minimal ABI for event signatures we index ────────────────────────────
EVENT_ABIS = [
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "uint256", "name": "campaignId", "type": "uint256"},
            {"indexed": True, "internalType": "address", "name": "donor", "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "amount", "type": "uint256"},
        ],
        "name": "DonationReceived",
        "type": "event",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "uint256", "name": "campaignId", "type": "uint256"},
            {"indexed": False, "internalType": "bool", "name": "isVerified", "type": "bool"},
            {"indexed": False, "internalType": "uint256", "name": "billTotal", "type": "uint256"},
            {"indexed": False, "internalType": "uint256", "name": "fraudRiskScore", "type": "uint256"},
        ],
        "name": "BillVerificationFulfilled",
        "type": "event",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "uint256", "name": "campaignId", "type": "uint256"},
            {"indexed": True, "internalType": "address", "name": "hospital", "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "amount", "type": "uint256"},
        ],
        "name": "FundsReleased",
        "type": "event",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "uint256", "name": "campaignId", "type": "uint256"},
            {"indexed": True, "internalType": "address", "name": "donor", "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "amount", "type": "uint256"},
        ],
        "name": "RefundIssued",
        "type": "event",
    },
]


# ─── Fallback RPC Provider ──────────────────────────────────────────────────

class FallbackAsyncHTTPProvider(AsyncHTTPProvider):
    """
    Multi-RPC provider with automatic failover.
    Iterates through a prioritized list of RPC endpoints upon request failure
    or rate-limiting. Rotates through providers in round-robin fashion.
    """

    def __init__(self, endpoint_urls: List[str], request_kwargs: Optional[dict] = None):
        if not endpoint_urls:
            raise ValueError("At least one RPC endpoint URL must be provided.")
        self.endpoint_urls = endpoint_urls
        self.current_index = 0
        super().__init__(endpoint_url=self.endpoint_urls[0], request_kwargs=request_kwargs)

    async def make_request(self, method: RPCEndpoint, params: Any) -> RPCResponse:
        total_endpoints = len(self.endpoint_urls)
        attempts = 0

        while attempts < total_endpoints:
            current_url = self.endpoint_urls[self.current_index]
            self.endpoint_url = current_url

            try:
                response = await super().make_request(method, params)

                if "error" in response:
                    error_code = response["error"].get("code")
                    error_msg = response["error"].get("message", "")
                    if error_code in [-32005, 429] or "rate limit" in error_msg.lower():
                        logger.warning(
                            "Rate limit hit on RPC, switching fallback",
                            extra={"endpoint": current_url, "error": error_msg},
                        )
                        self._rotate_endpoint()
                        attempts += 1
                        continue
                return response

            except (asyncio.TimeoutError, Web3RPCError, Exception) as exc:
                logger.warning(
                    "RPC provider failure, rotating endpoint",
                    extra={"endpoint": current_url, "error": str(exc)[:100]},
                )
                self._rotate_endpoint()
                attempts += 1
                await asyncio.sleep(1)

        raise ConnectionError(
            f"All {total_endpoints} RPC providers failed for method: {method}"
        )

    def _rotate_endpoint(self):
        self.current_index = (self.current_index + 1) % len(self.endpoint_urls)
        logger.info(
            "Switched to fallback RPC provider",
            extra={"endpoint": self.endpoint_urls[self.current_index]},
        )


# ─── Slack Alert Helper ────────────────────────────────────────────────────

async def send_slack_alert(message: str) -> None:
    """Send a high-priority alert to the configured Slack channel."""
    if not SLACK_WEBHOOK_URL:
        return
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            await client.post(
                SLACK_WEBHOOK_URL,
                json={"text": f"🚨 *Escrow Indexer Alert:* {message}"},
            )
        except Exception as exc:
            logger.error("Failed to send Slack alert", extra={"error": str(exc)})


# ─── Indexer Class ─────────────────────────────────────────────────────────

class EscrowIndexer:
    """
    Poll-based event indexer for MedicalEscrow contract with production-grade
    resilience, monitoring, and persistent state tracking.
    """

    def __init__(
        self,
        rpc_url: str,
        contract_address: str,
        poll_interval: int = POLL_INTERVAL_SECONDS,
        max_block_batch: int = BLOCK_BATCH_SIZE,
        confirmation_blocks: int = CONFIRMATION_BLOCKS,
        fallback_rpc_urls: Optional[List[str]] = None,
    ):
        # Build prioritized RPC list
        rpc_urls = [rpc_url]
        if fallback_rpc_urls:
            for url in fallback_rpc_urls:
                if url not in rpc_urls:
                    rpc_urls.append(url)

        provider = FallbackAsyncHTTPProvider(rpc_urls)
        self.w3 = AsyncWeb3(provider)
        self.contract_address = AsyncWeb3.to_checksum_address(contract_address)
        self.contract = self.w3.eth.contract(address=self.contract_address, abi=EVENT_ABIS)
        self.poll_interval = poll_interval
        self.max_block_batch = max_block_batch
        self.confirmation_blocks = confirmation_blocks
        self.start_block_fallback = getattr(settings, "ESCROW_START_BLOCK", 0)
        self.last_processed_block = 0

        logger.info(
            "EscrowIndexer initialized",
            extra={
                "contract_address": self.contract_address,
                "poll_interval": poll_interval,
                "confirmation_blocks": confirmation_blocks,
                "rpc_urls": rpc_urls,
            },
        )

    # ── Persistent State Helpers ──────────────────────────────────────────

    async def _get_last_processed_block(self, db: AsyncSession) -> int:
        """Retrieve last processed block from DB, falling back to ESCROW_START_BLOCK."""
        result = await db.execute(
            select(IndexerState).where(IndexerState.key == STATE_KEY_LAST_BLOCK)
        )
        state = result.scalars().first()
        if state and state.value is not None:
            return int(state.value)
        return self.start_block_fallback

    async def _set_last_processed_block(self, db: AsyncSession, block_number: int) -> None:
        """Upsert the last processed block height in the database."""
        result = await db.execute(
            select(IndexerState).where(IndexerState.key == STATE_KEY_LAST_BLOCK)
        )
        state = result.scalars().first()

        if not state:
            state = IndexerState(key=STATE_KEY_LAST_BLOCK, value=block_number)
            db.add(state)
        else:
            state.value = block_number

        await db.commit()

    # ── Event Handlers ────────────────────────────────────────────────────

    async def _handle_donation_received(
        self, event: LogReceipt, db: AsyncSession
    ) -> str:
        args = event.get("args", {})
        tx_hash = event.get("transactionHash", b"").hex()
        campaign_id = args.get("campaignId")
        amount_wei = args.get("amount", 0)
        donor_address = args.get("donor", "")
        block_number = event.get("blockNumber", 0)

        if not campaign_id or not tx_hash:
            logger.warning("Skipping DonationReceived with missing args")
            return "DonationReceived"

        amount_matic = float(self.w3.from_wei(amount_wei, "ether"))
        logger.info(
            "DonationReceived event",
            extra={
                "tx_hash": tx_hash[:10] + "...",
                "campaign_id": campaign_id,
                "amount_matic": amount_matic,
                "donor": donor_address[:10] + "...",
                "block_number": block_number,
            },
        )

        result = await db.execute(
            select(CrowdfundCampaign).where(
                CrowdfundCampaign.on_chain_campaign_id == int(campaign_id)
            )
        )
        campaign = result.scalars().first()
        if not campaign:
            logger.warning(
                "No DB campaign found for on_chain_campaign_id",
                extra={"campaign_id": campaign_id},
            )
            return "DonationReceived"

        current = float(campaign.current_amount or 0.0)
        campaign.current_amount = Decimal(str(current + amount_matic))

        donation = Donation(
            campaign_id=campaign.id,
            donor_address=donor_address,
            amount=Decimal(str(amount_matic)),
            tx_hash=tx_hash,
            block_number=int(block_number) if block_number else None,
        )
        db.add(donation)

        db.add(
            AuditLedger(
                user_id="escrow_indexer",
                domain="community",
                action="donation_synced",
                request_redacted=f"Donation event for campaign {campaign_id}",
                response_raw={
                    "tx_hash": tx_hash,
                    "amount_matic": amount_matic,
                    "donor": donor_address,
                    "block_number": block_number,
                },
            )
        )
        await db.commit()
        logger.info(
            "Donation synced to DB",
            extra={"campaign_title": campaign.title, "campaign_db_id": campaign.id},
        )
        return "DonationReceived"

    async def _handle_bill_verification_fulfilled(
        self, event: LogReceipt, db: AsyncSession
    ) -> str:
        args = event.get("args", {})
        campaign_id = args.get("campaignId")
        is_verified = args.get("isVerified", False)
        bill_total = args.get("billTotal", 0)
        fraud_score = args.get("fraudRiskScore", 0)

        if not campaign_id:
            return "BillVerificationFulfilled"

        logger.info(
            "BillVerificationFulfilled event",
            extra={"campaign_id": campaign_id, "is_verified": is_verified},
        )

        result = await db.execute(
            select(CrowdfundCampaign).where(
                CrowdfundCampaign.on_chain_campaign_id == int(campaign_id)
            )
        )
        campaign = result.scalars().first()
        if not campaign:
            logger.warning(
                "No DB campaign for on_chain_campaign_id",
                extra={"campaign_id": campaign_id},
            )
            return "BillVerificationFulfilled"

        campaign.bill_verification_status = "verified" if is_verified else "failed"
        if bill_total:
            campaign.total_bill_amount = Decimal(str(float(self.w3.from_wei(bill_total, "ether"))))
        if fraud_score:
            campaign.fraud_risk_score = Decimal(str(float(fraud_score) / 1_000_000_000_000_000_000))

        db.add(
            AuditLedger(
                user_id="escrow_indexer",
                domain="community",
                action="bill_verification_synced",
                request_redacted=f"Verification event for campaign {campaign_id}",
                response_raw={
                    "is_verified": is_verified,
                    "bill_total": str(bill_total),
                    "fraud_score": str(fraud_score),
                },
            )
        )
        await db.commit()
        return "BillVerificationFulfilled"

    async def _handle_funds_released(
        self, event: LogReceipt, db: AsyncSession
    ) -> str:
        args = event.get("args", {})
        campaign_id = args.get("campaignId")
        if not campaign_id:
            return "FundsReleased"

        logger.info("FundsReleased event", extra={"campaign_id": campaign_id})

        result = await db.execute(
            select(CrowdfundCampaign).where(
                CrowdfundCampaign.on_chain_campaign_id == int(campaign_id)
            )
        )
        campaign = result.scalars().first()
        if not campaign:
            logger.warning("No DB campaign for on_chain_campaign_id", extra={"campaign_id": campaign_id})
            return "FundsReleased"

        campaign.is_released = True
        campaign.bill_verification_status = "verified"

        db.add(
            AuditLedger(
                user_id="escrow_indexer",
                domain="community",
                action="funds_released_synced",
                request_redacted=f"Release event for campaign {campaign_id}",
                response_raw={"campaign_id": campaign_id},
            )
        )
        await db.commit()
        return "FundsReleased"

    async def _handle_refund_issued(
        self, event: LogReceipt, db: AsyncSession
    ) -> str:
        args = event.get("args", {})
        campaign_id = args.get("campaignId")
        amount_wei = args.get("amount", 0)
        tx_hash = event.get("transactionHash", b"").hex()
        if not campaign_id:
            return "RefundIssued"

        amount_matic = float(self.w3.from_wei(amount_wei, "ether"))
        logger.info(
            "RefundIssued event",
            extra={
                "campaign_id": campaign_id,
                "amount_matic": amount_matic,
                "tx_hash": tx_hash[:10] + "...",
            },
        )

        db.add(
            AuditLedger(
                user_id="escrow_indexer",
                domain="community",
                action="refund_synced",
                request_redacted=f"Refund event for campaign {campaign_id}",
                response_raw={"tx_hash": tx_hash, "amount_matic": amount_matic},
            )
        )
        await db.commit()
        return "RefundIssued"

    # ── Block Range Processing ──────────────────────────────────────────

    async def process_batch(
        self, db: AsyncSession, from_block: int, to_block: int, latest_chain_block: int
    ) -> None:
        """Fetch event logs for a block range and dispatch to handlers with monitoring."""
        start_time = time.time()

        CHAIN_TIP_BLOCK.set(latest_chain_block)
        INDEXER_CURRENT_BLOCK.set(to_block)
        lag = latest_chain_block - to_block
        INDEXER_LAG_BLOCKS.set(lag)

        if lag > SLACK_LAG_WARNING_THRESHOLD:
            logger.warning("Indexer falling behind chain tip", extra={"lag_blocks": lag})
        if lag > SLACK_LAG_CRITICAL_THRESHOLD:
            await send_slack_alert(
                f"Indexer critically lagging behind by {lag} blocks! "
                f"Last processed: {to_block}, Chain tip: {latest_chain_block}"
            )

        try:
            logger.info("Scanning block range", extra={"from_block": from_block, "to_block": to_block})

            donations = await self.contract.events.DonationReceived.get_logs(
                fromBlock=from_block, toBlock=to_block
            )
            for evt in donations:
                en = await self._handle_donation_received(evt, db)
                EVENTS_PROCESSED_TOTAL.labels(event_name=en).inc()

            verifications = await self.contract.events.BillVerificationFulfilled.get_logs(
                fromBlock=from_block, toBlock=to_block
            )
            for evt in verifications:
                en = await self._handle_bill_verification_fulfilled(evt, db)
                EVENTS_PROCESSED_TOTAL.labels(event_name=en).inc()

            releases = await self.contract.events.FundsReleased.get_logs(
                fromBlock=from_block, toBlock=to_block
            )
            for evt in releases:
                en = await self._handle_funds_released(evt, db)
                EVENTS_PROCESSED_TOTAL.labels(event_name=en).inc()

            refunds = await self.contract.events.RefundIssued.get_logs(
                fromBlock=from_block, toBlock=to_block
            )
            for evt in refunds:
                en = await self._handle_refund_issued(evt, db)
                EVENTS_PROCESSED_TOTAL.labels(event_name=en).inc()

            await self._set_last_processed_block(db, to_block)

            total_events = len(donations) + len(verifications) + len(releases) + len(refunds)
            logger.info(
                "Block range processed successfully",
                extra={
                    "from_block": from_block,
                    "to_block": to_block,
                    "events_found": total_events,
                },
            )

        except Exception as e:
            INDEXER_ERRORS_TOTAL.labels(error_type=type(e).__name__).inc()
            sentry_sdk.capture_exception(e)
            logger.error(
                "Error processing block batch",
                extra={"from_block": from_block, "to_block": to_block, "error": str(e)},
                exc_info=True,
            )
            raise
        finally:
            PROCESSING_TIME.observe(time.time() - start_time)

    async def initialize_from_db(self, db: AsyncSession) -> int:
        """Initialize starting block from DB state, or fall back to chain tip."""
        last_processed = await self._get_last_processed_block(db)

        if last_processed == 0:
            last_processed = await self.w3.eth.block_number
            await self._set_last_processed_block(db, last_processed)
            logger.info("Indexer initialized at chain tip (no previous state)", extra={"start_block": last_processed})
        else:
            logger.info("Indexer resuming from persisted state", extra={"last_processed_block": last_processed})

        return last_processed

    # ── Main Loop ────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Run the continuous polling loop with reorg-safe processing."""
        logger.info("EscrowIndexer starting", extra={
            "contract": self.contract_address,
            "batch_size": self.max_block_batch,
            "confirmation_blocks": self.confirmation_blocks,
        })

        async with AsyncSessionLocal() as init_db:
            try:
                self.last_processed_block = await self.initialize_from_db(init_db)
            except Exception as e:
                logger.error("Failed to initialize indexer state from DB", extra={"error": str(e)}, exc_info=True)
                self.last_processed_block = await self.w3.eth.block_number

        while True:
            try:
                latest_block = await self.w3.eth.block_number
                safe_tip = max(0, latest_block - self.confirmation_blocks)

                if safe_tip > self.last_processed_block:
                    from_block = self.last_processed_block + 1
                    to_block = min(safe_tip, from_block + self.max_block_batch)

                    async with AsyncSessionLocal() as db:
                        try:
                            await self.process_batch(db, from_block, to_block, latest_block)
                            self.last_processed_block = to_block
                        except Exception as batch_error:
                            logger.error(
                                "Batch processing failed, will retry in next cycle",
                                extra={"from_block": from_block, "to_block": to_block, "error": str(batch_error)},
                            )
                else:
                    logger.debug(
                        "Indexer synced to safe tip",
                        extra={
                            "last_processed": self.last_processed_block,
                            "chain_tip": latest_block,
                            "safe_tip": safe_tip,
                        },
                    )

            except Exception as exc:
                INDEXER_ERRORS_TOTAL.labels(error_type="loop_error").inc()
                sentry_sdk.capture_exception(exc)
                logger.error("Indexer main loop error", extra={"error": str(exc)}, exc_info=True)
                await send_slack_alert(f"Indexer main loop crashed: {str(exc)[:200]}")

            await asyncio.sleep(self.poll_interval)

    async def stop(self) -> None:
        """Graceful shutdown — persist current state."""
        logger.info("EscrowIndexer stopping...")
        try:
            async with AsyncSessionLocal() as db:
                await self._set_last_processed_block(db, self.last_processed_block)
                logger.info("Indexer state persisted on shutdown", extra={"last_block": self.last_processed_block})
        except Exception as e:
            logger.error("Failed to persist state on shutdown", extra={"error": str(e)})


# ─── Standalone Entry Point ──────────────────────────────────────────────

async def main():
    """Entry point for running the indexer as a standalone process."""
    contract_addr = getattr(settings, "ESCROW_CONTRACT_ADDRESS", "")
    if not contract_addr or contract_addr == "0x0000000000000000000000000000000000000000":
        logger.warning("ESCROW_CONTRACT_ADDRESS not set. Indexer will not start.")
        return

    # Determine RPC URLs based on environment
    if settings.APP_ENV.value == "production":
        rpc_url = getattr(settings, "POLYGON_MAINNET_RPC_URL", "")
        fallback_urls = getattr(settings, "POLYGON_FALLBACK_RPC_URLS", [])
    else:
        rpc_url = getattr(settings, "POLYGON_AMOY_RPC_URL", "https://rpc-amoy.polygon.technology")
        fallback_urls = []

    if not rpc_url:
        logger.warning("No RPC URL configured. Indexer will not start.")
        return

    indexer = EscrowIndexer(
        rpc_url=rpc_url,
        contract_address=contract_addr,
        poll_interval=POLL_INTERVAL_SECONDS,
        fallback_rpc_urls=fallback_urls,
    )
    await indexer.start()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
