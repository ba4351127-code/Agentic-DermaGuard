
import cv2
import numpy as np
import json
from pathlib import Path


def load_quality_thresholds():
    config_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "dataset_info.json"
    )

    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    thresholds = config["quality_thresholds"]

    required_keys = [
        "blur_low",
        "bright_low",
        "bright_high",
        "glare_high",
        "hair_high",
        "dark_high"
    ]

    for key in required_keys:
        if key not in thresholds:
            raise KeyError(f"Missing quality threshold: {key}")

    return {
        key: float(thresholds[key])
        for key in required_keys
    }


QUALITY_THRESHOLDS = load_quality_thresholds()


def failure_response(issue, reason):
    """Return a consistent response for invalid or failed processing."""

    return {
        "agent_id": "A1",
        "quality_status": "FAIL",
        "quality_score": 0.0,
        "issues": [issue],
        "reasons": [reason],
        "metrics": {}
    }


def run_agent(image_bytes: bytes, metadata: dict) -> dict:
    """
    A1 - Image Quality Agent

    Input:
        image_bytes: Uploaded image in bytes
        metadata: Patient metadata dictionary

    Output:
        Quality status, score, issues, reasons and metrics.
    """

    try:
        # 1. Validate input
        if not isinstance(image_bytes, (bytes, bytearray, memoryview)):
            return failure_response(
                "invalid_image",
                "Image input must be bytes."
            )

        if not image_bytes:
            return failure_response(
                "invalid_image",
                "Uploaded image is empty."
            )

        if not isinstance(metadata, dict):
            return failure_response(
                "invalid_metadata",
                "Metadata must be a dictionary."
            )

        # 2. Decode uploaded image
        image_array = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return failure_response(
                "invalid_image",
                "The uploaded image could not be decoded."
            )

        # 3. Resize image to 224 x 224
        im = cv2.resize(img_bgr, (224, 224))

        # 4. Convert to grayscale
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # 5. Calculate blur
        blur = float(cv2.Laplacian(
            gray, cv2.CV_64F
        ).var())

        # 6. Calculate brightness
        brightness = float(gray.mean())

        # 7. Calculate dark fraction
        dark_frac = float((gray < 20).mean())

        # 8. Calculate glare fraction
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

        glare_frac = float(
            ((hsv[..., 2] > 240) & (hsv[..., 1] < 40)).mean()
        )

        # 9. Calculate hair fraction
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (9, 9)
        )

        blackhat = cv2.morphologyEx(
            gray, cv2.MORPH_BLACKHAT, kernel
        )

        hair_frac = float((blackhat > 15).mean())
        # Calculate contrast
        contrast = float(gray.std())
         # 10. Store metrics
        metrics = {
            "laplacian_variance": blur,
            "mean_brightness": brightness,
            "contrast": contrast,
            "dark_fraction": dark_frac,
            "glare_fraction": glare_frac,
            "hair_fraction": hair_frac
        }
        # 11. Detect quality issues
        issues = []
        notes = []

        QT = QUALITY_THRESHOLDS

        if blur < QT["blur_low"]:
            issues.append("blurry")

        if brightness < QT["bright_low"]:
            issues.append("too dark")

        if brightness > QT["bright_high"]:
            issues.append("over-exposed")

        if glare_frac > QT["glare_high"]:
            issues.append("glare")

        if hair_frac > QT["hair_high"]:
            issues.append("hair/ruler obstruction")

        # Avoid double-counting dark fraction
        if dark_frac > QT["dark_high"]:
            if "too dark" in issues:
                notes.append("large dark area (supports too dark)")
            else:
                issues.append("heavy black border/vignette")

        # 12. Calculate quality score
        quality_score = max(
            0.0, 1.0 - 0.25 * len(issues)
        )

        # 13. Decide PASS or FAIL
        hard_issues = {
            "blurry",
            "too dark",
            "over-exposed"
        }

        if (
            any(issue in hard_issues for issue in issues)
            or len(issues) >= 2
        ):
            quality_status = "FAIL"
        else:
            quality_status = "PASS"

        reasons = issues + notes

        if not reasons:
            reasons = ["ok"]

        # 14. Return A1 output
        return {
            "agent_id": "A1",
            "quality_status": quality_status,
            "quality_score": float(quality_score),
            "issues": issues,
            "reasons": reasons,
            "metrics": metrics
        }

    except Exception as error:
        return failure_response(
            "processing_error",
            f"Image processing failed: {str(error)}"
        )
