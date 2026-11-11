from collections.abc import Mapping
from math import isfinite
from typing import Any

from backend.config import CLASS_CODES, Settings, settings


def _is_probability(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def review_safety(
    quality: Mapping[str, Any] | None,
    diagnosis: Mapping[str, Any] | None,
    lesion: Mapping[str, Any] | None,
    context: Mapping[str, Any] | None,
    config: Settings = settings,
) -> dict[str, Any]:
    """Review agent outputs and veto automatic triage when checks fail."""
    triggers: list[str] = []
    justifications: list[str] = []
    veto = False

    def flag(code: str, reason: str, *, blocks_auto_triage: bool = True) -> None:
        nonlocal veto
        triggers.append(code)
        justifications.append(reason)
        if blocks_auto_triage:
            veto = True

    outputs = {
        "A1": quality,
        "A2": diagnosis,
        "A3": lesion,
        "A4": context,
    }
    for agent_id, output in outputs.items():
        if not isinstance(output, Mapping):
            flag(f"{agent_id}_OUTPUT_MISSING", f"{agent_id} output is missing or invalid.")

    if isinstance(quality, Mapping):
        status = str(quality.get("quality_status", "")).upper()
        if status == "FAIL":
            flag("IMAGE_QUALITY_FAIL", "A1 marked the image as unsuitable.")
        elif status != "PASS":
            flag("IMAGE_QUALITY_UNKNOWN", "A1 did not return PASS or FAIL.")

    if isinstance(diagnosis, Mapping):
        predicted_class = diagnosis.get("predicted_class")
        confidence = diagnosis.get("confidence")
        probabilities = diagnosis.get("class_probabilities")

        if predicted_class not in CLASS_CODES:
            flag("INVALID_DIAGNOSIS_CLASS", "A2 returned an unrecognized class code.")

        if not _is_probability(confidence):
            flag("INVALID_DIAGNOSIS_CONFIDENCE", "A2 confidence is missing or invalid.")

        if not isinstance(probabilities, Mapping):
            flag("INVALID_CLASS_PROBABILITIES", "A2 class probabilities are missing.")
        else:
            valid_keys = set(probabilities) == set(CLASS_CODES)
            valid_values = all(_is_probability(v) for v in probabilities.values())
            total = sum(float(v) for v in probabilities.values()) if valid_values else -1.0

            if not valid_keys or not valid_values or abs(total - 1.0) > 0.02:
                flag(
                    "INVALID_CLASS_PROBABILITIES",
                    "A2 probabilities must cover all seven classes and sum to about 1.",
                )
            elif predicted_class in CLASS_CODES:
                top_class = max(probabilities, key=lambda key: float(probabilities[key]))
                if top_class != predicted_class:
                    flag(
                        "DIAGNOSIS_PROBABILITY_CONFLICT",
                        "A2 predicted class does not match its highest class probability.",
                    )

    if isinstance(lesion, Mapping):
        score = lesion.get("suspicion_score")
        if not _is_probability(score):
            flag("INVALID_LESION_SCORE", "A3 suspicion score is missing or outside 0–1.")
        elif config.lesion_refer_threshold is None:
            flag(
                "LESION_THRESHOLD_UNSET",
                "The lesion review threshold has not been selected on validation data.",
            )
        elif float(score) >= config.lesion_refer_threshold:
            flag(
                "LESION_REVIEW_THRESHOLD",
                "A3 suspicion score reached the configured review threshold.",
            )

        reliability_flags = lesion.get("reliability_flags", [])
        if not isinstance(reliability_flags, list):
            flag("INVALID_RELIABILITY_FLAGS", "A3 reliability flags must be a list.")
        elif reliability_flags:
            flag(
                "LESION_EVIDENCE_UNRELIABLE",
                "A3 reported reliability concerns: " + ", ".join(map(str, reliability_flags)),
            )

        conflict_flags = lesion.get("conflict_flags", [])
        if conflict_flags:
            flag(
                "LESION_CONFLICT",
                "A3 reported conflicting evidence: " + ", ".join(map(str, conflict_flags)),
            )

    if isinstance(context, Mapping):
        if str(context.get("status", "")).upper() != "READY":
            flag("CONTEXT_AGENT_NOT_READY", "A4 context output is not marked READY.")

        missing_fields = context.get("missing_field_flags", [])
        if not isinstance(missing_fields, list):
            flag("INVALID_CONTEXT_FLAGS", "A4 missing-field flags must be a list.")
        elif missing_fields:
            # Missing demographics are recorded for the rationale. They are not
            # automatically treated as a diagnosis or as a safety veto.
            flag(
                "CONTEXT_FIELDS_MISSING",
                "A4 reports missing fields: " + ", ".join(map(str, missing_fields)),
                blocks_auto_triage=False,
            )

        context_conflicts = context.get("conflict_flags", [])
        if context_conflicts:
            flag(
                "CONTEXT_CONFLICT",
                "A4 reported conflicting context: " + ", ".join(map(str, context_conflicts)),
            )

    return {
        "agent_id": "A5",
        "safety_flag": "SAFETY_FLAG" if veto else "CLEAR",
        "veto_auto_triage": veto,
        "triggers": triggers,
        "justification": justifications,
    }