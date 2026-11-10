def analyze_context(
    age: int | None = None,
    sex: str | None = None,
    lesion_location: str | None = None,
) -> dict:
    """A4 scaffold: report missing context; risk scoring is not implemented."""
    values = {
        "age": age,
        "sex": sex,
        "lesion_location": lesion_location,
    }
    missing_fields = [
        name for name, value in values.items()
        if value is None or (isinstance(value, str) and not value.strip())
    ]

    return {
        "agent_id": "A4",
        "status": "NOT_IMPLEMENTED",
        "context_evidence": [],
        "risk_indicator": "UNKNOWN",
        "missing_field_flags": missing_fields,
    }