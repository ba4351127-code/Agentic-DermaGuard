import re
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from backend.config import settings
from backend.pipeline import (
    AgentContractError,
    AgentsNotReady,
    pipeline,
)

app = FastAPI(
    title="Agentic DermaGuard API",
    version="0.2.0",
    description="Multi-agent skin-lesion screening prototype.",
)


class PredictionResponse(BaseModel):
    image_id: str
    prediction: str
    decision: Literal["AUTO_TRIAGE", "REFER", "RE_IMAGE"]
    confidence: float | None
    class_probabilities: dict[str, float]
    agents: dict[str, dict[str, Any]]
    rationale: list[str]


@app.get("/health", tags=["operations"])
def health() -> dict[str, str]:
    return {"status": "ok", "service": "Agentic DermaGuard API"}


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict(
    image: UploadFile = File(...),
    image_id: str | None = Form(default=None),
    age: int | None = Form(default=None, ge=0, le=120),
    sex: str | None = Form(default=None),
    lesion_location: str | None = Form(default=None),
) -> dict[str, Any]:
    suffix = Path(image.filename or "").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(415, "Upload a JPG, JPEG, or PNG image.")

    if image.content_type not in {"image/jpeg", "image/png", "application/octet-stream"}:
        raise HTTPException(415, "Unsupported image content type.")

    image_bytes = await image.read(settings.max_image_bytes + 1)
    if not image_bytes:
        raise HTTPException(400, "The uploaded image is empty.")
    if len(image_bytes) > settings.max_image_bytes:
        raise HTTPException(413, "The uploaded image exceeds the configured size limit.")

    resolved_id = image_id or Path(image.filename or "image").stem
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", resolved_id):
        raise HTTPException(422, "image_id may contain letters, numbers, dot, underscore, or hyphen.")

    metadata = {
        "image_id": resolved_id,
        "age": age,
        "sex": sex,
        "lesion_location": lesion_location,
    }

    try:
        return await run_in_threadpool(
            pipeline.predict,
            image_bytes,
            resolved_id,
            metadata,
        )
    except AgentsNotReady as exc:
        raise HTTPException(503, str(exc)) from exc
    except AgentContractError as exc:
        raise HTTPException(502, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(502, str(exc)) from exc