
import cv2
import numpy as np


def failure_response(issue, reason):
    """Safe response for invalid or unprocessable images."""
    return {
        "agent_id": "A3",
        "suspicion_score": 0.0,
        "reliability_flags": [issue],
        "conflict_flags": [],
        "feature_evidence": {
            "available": False,
            "features": {},
            "limitations": [reason]
        }
    }


def run_agent(image_bytes: bytes, metadata: dict) -> dict:
    """
    A3 - Lesion Analysis Agent

    Extracts heuristic visual features:
    - Asymmetry
    - Border irregularity
    - Colour variation

    The suspicion score is a prototype heuristic score,
    NOT a clinically validated probability.
    """

    # 1. Validate inputs
    if not isinstance(metadata, dict):
        return failure_response(
            "invalid_metadata",
            "Metadata must be a dictionary."
        )

    if not isinstance(image_bytes, bytes) or not image_bytes:
        return failure_response(
            "invalid_image",
            "Image bytes are empty or invalid."
        )

    try:
        # 2. Decode image
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            return failure_response(
                "invalid_image",
                "Uploaded image could not be decoded."
            )

        # 3. Resize image
        image = cv2.resize(image, (224, 224))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 4. Estimate lesion mask
        # Compare image pixels with the median brightness near the border.
        border_pixels = np.concatenate([
            gray[0, :],
            gray[-1, :],
            gray[:, 0],
            gray[:, -1]
        ])

        border_median = float(np.median(border_pixels))

        threshold = min(max(border_median - 10, 0), 255)

        mask = (gray < threshold).astype(np.uint8) * 255

        # Clean the estimated mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, kernel
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, kernel
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        reliability_flags = []
        conflict_flags = []
        limitations = []

        if not contours:
            return failure_response(
                "lesion_not_detected",
                "A reliable lesion region could not be identified."
            )

        # Select largest detected region
        contour = max(contours, key=cv2.contourArea)
        lesion_mask = np.zeros_like(gray)

        cv2.drawContours(
            lesion_mask, [contour], -1, 255, thickness=-1
        )

        lesion_area = float(cv2.countNonZero(lesion_mask))
        image_area = float(gray.shape[0] * gray.shape[1])
        area_fraction = lesion_area / image_area

        if area_fraction < 0.02 or area_fraction > 0.85:
            reliability_flags.append("uncertain_lesion_segmentation")
            limitations.append(
                "Estimated lesion area is unusually small or large."
            )

        # 5. Asymmetry calculation
        def calculate_asymmetry(binary_mask):
            height, width = binary_mask.shape

            left = binary_mask[:, :width // 2]
            right = binary_mask[:, width - width // 2:]
            right = cv2.flip(right, 1)

            min_width = min(left.shape[1], right.shape[1])
            left = left[:, :min_width]
            right = right[:, :min_width]

            horizontal_difference = np.logical_xor(
                left > 0, right > 0
            ).mean()

            top = binary_mask[:height // 2, :]
            bottom = binary_mask[height - height // 2:, :]
            bottom = cv2.flip(bottom, 0)

            min_height = min(top.shape[0], bottom.shape[0])
            top = top[:min_height, :]
            bottom = bottom[:min_height, :]

            vertical_difference = np.logical_xor(
                top > 0, bottom > 0
            ).mean()

            return float(
                (horizontal_difference + vertical_difference) / 2
            )

        asymmetry = calculate_asymmetry(lesion_mask)

        # 6. Border irregularity
        perimeter = float(cv2.arcLength(contour, True))
        area = float(cv2.contourArea(contour))

        if perimeter > 0 and area > 0:
            circularity = float(
                (4 * np.pi * area) / (perimeter ** 2)
            )
            circularity = min(max(circularity, 0.0), 1.0)
            border_irregularity = 1.0 - circularity
        else:
            circularity = 0.0
            border_irregularity = 0.0
            reliability_flags.append("unreliable_border_measurement")

        # 7. Colour variation inside estimated lesion
        lesion_pixels = image[lesion_mask > 0]

        if len(lesion_pixels) >= 20:
            color_std = float(
                np.mean(np.std(lesion_pixels.astype(np.float32), axis=0))
            )
            color_variation = min(color_std / 80.0, 1.0)
        else:
            color_std = 0.0
            color_variation = 0.0
            reliability_flags.append("insufficient_lesion_pixels")

        # 8. Check upstream image quality, if supplied
        if metadata.get("quality_status") == "FAIL":
            reliability_flags.append("upstream_quality_fail")
            limitations.append(
                "A1 marked the image as unsuitable for reliable analysis."
            )

        # 9. Detect feature disagreement
        # These are internal heuristic disagreements, not medical conflicts.
        if asymmetry < 0.15 and (
            border_irregularity > 0.65 or color_variation > 0.65
        ):
            conflict_flags.append("mixed_visual_feature_evidence")

        # 10. Calculate prototype suspicion score
        suspicion_score = (
            0.35 * asymmetry +
            0.35 * border_irregularity +
            0.30 * color_variation
        )

        suspicion_score = float(
            np.clip(suspicion_score, 0.0, 1.0)
        )

        # 11. Evidence dictionary
        feature_evidence = {
            "available": True,
            "segmentation_method": "border-brightness threshold heuristic",
            "features": {
                "asymmetry": round(asymmetry, 4),
                "border_irregularity": round(border_irregularity, 4),
                "circularity": round(circularity, 4),
                "color_std": round(color_std, 4),
                "color_variation": round(color_variation, 4),
                "lesion_area_fraction": round(area_fraction, 4)
            },
            "limitations": limitations + [
                "Lesion segmentation is heuristic and may be inaccurate.",
                "Visual features are not a clinical diagnosis.",
                "Suspicion score is not a clinically validated probability."
            ]
        }

        return {
            "agent_id": "A3",
            "suspicion_score": suspicion_score,
            "reliability_flags": reliability_flags,
            "conflict_flags": conflict_flags,
            "feature_evidence": feature_evidence
        }

    except Exception as error:
        return failure_response(
            "processing_error",
            f"Image analysis failed safely: {type(error).__name__}"
        )
