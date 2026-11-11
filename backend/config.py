import os
from dataclasses import dataclass


# Confirm these codes match the codes supplied by the organizers.
CLASS_CODES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
DECISIONS = ("AUTO_TRIAGE", "REFER", "RE_IMAGE")


def _optional_probability(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None

    value = float(raw)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


@dataclass(frozen=True)
class Settings:
    auto_triage_min_confidence: float | None
    lesion_refer_threshold: float | None
    max_image_bytes: int

    @classmethod
    def from_environment(cls) -> "Settings":
        max_bytes = int(os.getenv("DERMAGUARD_MAX_IMAGE_BYTES", "10000000"))
        if max_bytes <= 0:
            raise ValueError("DERMAGUARD_MAX_IMAGE_BYTES must be positive")

        return cls(
            auto_triage_min_confidence=_optional_probability(
                "DERMAGUARD_AUTO_TRIAGE_MIN_CONFIDENCE"
            ),
            lesion_refer_threshold=_optional_probability(
                "DERMAGUARD_LESION_REFER_THRESHOLD"
            ),
            max_image_bytes=max_bytes,
        )


settings = Settings.from_environment()