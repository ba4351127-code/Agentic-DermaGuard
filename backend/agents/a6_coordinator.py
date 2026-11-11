from collections.abc import Mapping
from math import isfinite
from typing import Any

from backend.config import CLASS_CODES, Settings, settings


def coordinate(
    quality: Mapping[str, Any],
    diagnosis: Mapping[str, Any],
    lesion: Mapping[str, Any],
    context: Mapping[str, Any],
    safety: Mapping[str, Any],
    config: Settings = settings,
) -> dict[str, Any]:
    """Apply the documented decision policy and return a traceable rationale."""
    prediction = diagnosis.get("predicted_class")
    if prediction not in CLASS_CODES:
        raise ValueError("A2 must return a valid predicted_class before coordination.")

    confidence = diagnosis.get("confidence")
    confidence_is_valid = (
        isinstance(confidence, (int, float))
        and isfinite(float(confidence))
        and 0.0 <= float(confidence) <= 1.0
    )

    quality_status = str(quality.get("quality_status", "")).upper()
    veto = bool(safety.get("veto_auto_triage", True))
    rationale = [
        f"A1 image quality: {quality_status or 'UNKNOWN'}.",
        f"A2 prediction: {prediction}; confidence: "
        f"{float(confidence):.3f}." if confidence_is_valid
        else f"A2 prediction: {prediction}; confidence unavailable.",
        f"A5 safety veto: {veto}.",
    ]

    missing_context = context.get("missing_field_flags", [])
    if missing_context:
        rationale.append(
            "A4 context fields missing: " + ", ".join(map(str, missing_context)) + "."
        )

    lesion_score = lesion.get("suspicion_score")
    if isinstance(lesion_score, (int, float)):
        rationale.append(f"A3 suspicion score: {float(lesion_score):.3f}.")

    if quality_status == "FAIL":
        decision = "RE_IMAGE"
        rationale.append("Policy: request a new image because A1 quality failed.")
    elif quality_status != "PASS":
        decision = "REFER"
        rationale.append("Policy: refer because image quality is unknown.")
    elif veto:
        decision = "REFER"
        rationale.append("Policy: refer because A5 vetoed automatic triage.")
    elif config.auto_triage_min_confidence is None:
        decision = "REFER"
        rationale.append("Policy: refer because the confidence threshold is unset.")
    elif config.lesion_refer_threshold is None:
        decision = "REFER"
        rationale.append("Policy: refer because the lesion review threshold is unset.")
    elif not confidence_is_valid:
        decision = "REFER"
        rationale.append("Policy: refer because A2 confidence is unavailable.")
    elif float(confidence) < config.auto_triage_min_confidence:
        decision = "REFER"
        rationale.append("Policy: confidence is below the validation threshold.")
    else:
        decision = "AUTO_TRIAGE"
        rationale.append(
            "Policy: quality passed, A5 did not veto, and configured thresholds passed."
        )

    return {
        "agent_id": "A6",
        "predicted_label": prediction,
        "triage_decision": decision,
        "rationale": rationale,
    }