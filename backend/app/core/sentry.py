"""
Sentry SDK Initialization
==========================
Centralized error tracking with performance monitoring for production deployments.
"""

import logging
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration
from backend.app.core.config import settings

logger = logging.getLogger("sentry")


def init_sentry() -> None:
    """
    Initialize Sentry SDK for error tracking and performance monitoring.
    Only initializes if SENTRY_DSN is configured in the environment.

    To enable, add to your .env or env.production:
        SENTRY_DSN=https://your-dsn@oXXXXX.ingest.sentry.io/XXXXXX
        SENTRY_ENVIRONMENT=production
        SENTRY_TRACES_SAMPLE_RATE=0.1
    """
    sentry_dsn = getattr(settings, "SENTRY_DSN", None)
    if not sentry_dsn:
        logger.info("Sentry DSN not configured. Skipping Sentry initialization.")
        return

    environment = getattr(settings, "SENTRY_ENVIRONMENT", settings.APP_ENV.value)
    traces_sample_rate = float(getattr(settings, "SENTRY_TRACES_SAMPLE_RATE", 0.1))

    sentry_logging = LoggingIntegration(
        level=logging.INFO,  # Capture info and above as breadcrumbs
        event_level=logging.ERROR,  # Send errors as events
    )

    sentry_sdk.init(
        dsn=sentry_dsn,
        environment=environment,
        traces_sample_rate=traces_sample_rate,
        integrations=[sentry_logging],
        # Send request and context data with events
        send_default_pii=False,
        # Attach stack traces to all logged messages
        attach_stacktrace=True,
    )

    logger.info("Sentry initialized for environment: %s", environment)

