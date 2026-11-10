def coordinate(
    predicted_label: str | None = None,
    confidence: float | None = None,
    quality_status: str = "UNKNOWN",
    safety_veto: bool = True,
    safety_triggers: list[str] | None = None,
) -> dict:
    """A6 scaffold: fail closed until agents and thresholds are ready."""
    rationale = []

    if safety_veto:
        decision = "REFER"
        rationale.append("Safety review requires human review.")
        rationale.extend(f"Safety trigger: {item}" for item in (safety_triggers or []))
    elif quality_status.upper() == "FAIL":
        decision = "RE_IMAGE"
        rationale.append("Image quality failed; request a new image.")
    else:
        decision = "REFER"
        rationale.append("Coordinator policy and thresholds are not calibrated.")

    return {
        "agent_id": "A6",
        "predicted_label": predicted_label or "unknown",
        "confidence": confidence,
        "decision": decision,
        "rationale": rationale,
    }