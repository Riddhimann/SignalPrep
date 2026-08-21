import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("operation", "latency_ms", "failure_reason", "session_id"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)


@contextmanager
def log_latency(logger: logging.Logger, operation: str, **context: str) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    except Exception as exc:
        logger.exception(
            "operation_failed",
            extra={
                "operation": operation,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "failure_reason": type(exc).__name__,
                **context,
            },
        )
        raise
    else:
        logger.info(
            "operation_completed",
            extra={
                "operation": operation,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                **context,
            },
        )
