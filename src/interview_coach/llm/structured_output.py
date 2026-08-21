from pydantic import BaseModel


def compact_schema(model: type[BaseModel]) -> dict:
    """Expose the exact contract used for structured generation and interviews."""
    return model.model_json_schema()
