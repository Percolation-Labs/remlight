"""OpenTelemetry instrumentation setup for REMLight agents.

Provides:
- OTLP exporter configuration for Phoenix
- OpenInference span processor for Pydantic AI
- Idempotent setup (safe to call multiple times)
"""

from loguru import logger

from remlight.settings import settings

# Global flag to track if instrumentation is initialized
_instrumentation_initialized = False


def setup_instrumentation() -> None:
    """
    Initialize OpenTelemetry instrumentation.

    Idempotent - safe to call multiple times, only initializes once.

    Configures:
    - OTLP HTTP exporter to Phoenix
    - OpenInference span processor for Pydantic AI traces

    Environment variables:
        OTEL__ENABLED - Enable instrumentation (default: false)
        OTEL__SERVICE_NAME - Service name (default: remlight-api)
        OTEL__COLLECTOR_ENDPOINT - Phoenix endpoint (default: http://localhost:6006)
        OTEL__PROTOCOL - Protocol (http or grpc, default: http)
    """
    global _instrumentation_initialized

    if _instrumentation_initialized:
        logger.debug("OTEL instrumentation already initialized, skipping")
        return

    if not settings.otel.enabled:
        logger.debug("OTEL instrumentation disabled (OTEL__ENABLED=false)")
        return

    logger.info("Initializing OpenTelemetry instrumentation...")

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter as HTTPExporter,
        )

        # Create resource with service metadata
        resource = Resource(
            attributes={
                SERVICE_NAME: settings.otel.service_name,
                "deployment.environment": settings.environment,
            }
        )

        # Create tracer provider
        tracer_provider = TracerProvider(resource=resource)

        # Configure OTLP HTTP exporter (Phoenix uses HTTP)
        endpoint = f"{settings.otel.collector_endpoint}/v1/traces"
        exporter = HTTPExporter(
            endpoint=endpoint,
            timeout=settings.otel.export_timeout,
        )

        # Add span processor
        tracer_provider.add_span_processor(BatchSpanProcessor(exporter))

        # Set as global tracer provider
        trace.set_tracer_provider(tracer_provider)

        logger.info(f"OTLP exporter configured: {endpoint}")

        # Add OpenInference span processor for Pydantic AI
        try:
            from openinference.instrumentation.pydantic_ai import (
                OpenInferenceSpanProcessor,
            )

            tracer_provider.add_span_processor(OpenInferenceSpanProcessor())
            logger.info("Added OpenInference span processor for Pydantic AI")

        except ImportError:
            logger.warning(
                "openinference-instrumentation-pydantic-ai not installed - "
                "traces will lack OpenInference attributes"
            )

        _instrumentation_initialized = True
        logger.info("OpenTelemetry instrumentation initialized successfully")

    except ImportError as e:
        logger.warning(f"OpenTelemetry packages not installed: {e}")
    except Exception as e:
        logger.error(f"Failed to initialize OTEL instrumentation: {e}")
        # Don't raise - allow application to continue without tracing
