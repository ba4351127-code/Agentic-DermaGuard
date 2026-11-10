def review_safety(
    quality: dict | None = None,
    diagnosis: dict | None = None,
    lesion: dict | None = None,
    context: dict | None = None,
) -> dict:
    """A5 scaffold: require review until safety checks are implemented."""
    outputs = {
        "A1": quality,
        "A2": diagnosis,
        "A3": lesion,
        "A4": context,
    }
    missing = [agent for agent, result in outputs.items() if result is None]

    triggers = ["SAFETY_RULES_NOT_IMPLEMENTED"]
    if missing:
        triggers.append("MISSING_AGENT_OUTPUTS:" + ",".join(missing))

    return {
        "agent_id": "A5",
        "status": "NOT_IMPLEMENTED",
        "safety_flag": "REVIEW_REQUIRED",
        "veto_auto_triage": True,
        "triggers": triggers,
        "justification": "Safety checks are pending; send the case for human review.",
    }