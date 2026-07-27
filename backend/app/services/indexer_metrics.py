"""
Prometheus Metrics for Escrow Indexer
======================================
Exposes key operational metrics that can be scraped by Prometheus and
visualized in Grafana dashboards.

Metrics exposed:
  - escrow_indexer_last_processed_block (Gauge)
  - escrow_indexer_chain_tip_block (Gauge)
  - escrow_indexer_lag_blocks (Gauge)
  - escrow_indexer_events_processed_total (Counter, labels: event_name)
  - escrow_indexer_errors_total (Counter, labels: error_type)
  - escrow_indexer_loop_duration_seconds (Histogram)
"""

from prometheus_client import Gauge, Counter, Histogram

# ─── Lag & Progress Metrics ────────────────────────────────────────────────

INDEXER_CURRENT_BLOCK = Gauge(
    "escrow_indexer_last_processed_block",
    "Last block successfully processed by the indexer",
)

CHAIN_TIP_BLOCK = Gauge(
    "escrow_indexer_chain_tip_block",
    "Latest block height on the blockchain",
)

INDEXER_LAG_BLOCKS = Gauge(
    "escrow_indexer_lag_blocks",
    "Number of blocks the indexer is behind chain tip",
)

# ─── Event Processing Metrics ──────────────────────────────────────────────

EVENTS_PROCESSED_TOTAL = Counter(
    "escrow_indexer_events_processed_total",
    "Total smart contract events indexed",
    ["event_name"],
)

# ─── Error Metrics ─────────────────────────────────────────────────────────

INDEXER_ERRORS_TOTAL = Counter(
    "escrow_indexer_errors_total",
    "Total errors encountered in indexer loop",
    ["error_type"],
)

# ─── Performance Metrics ───────────────────────────────────────────────────

PROCESSING_TIME = Histogram(
    "escrow_indexer_loop_duration_seconds",
    "Time taken to process a single block batch",
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

