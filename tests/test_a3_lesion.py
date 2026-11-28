
import os
import math
from pathlib import Path

import pytest

from backend.agents.a3_lesion import run_agent


@pytest.fixture
def sample_image_bytes():
    image_path = os.getenv("DERMAGUARD_TEST_IMAGE")

    if not image_path:
        pytest.skip("Set DERMAGUARD_TEST_IMAGE to run image tests.")

    path = Path(image_path)

    if not path.is_file():
        pytest.fail(f"Test image not found: {path}")

    return path.read_bytes()


def test_valid_image(sample_image_bytes):
    result = run_agent(sample_image_bytes, {})

    assert result["agent_id"] == "A3"
    assert isinstance(result["suspicion_score"], (int, float))
    assert math.isfinite(result["suspicion_score"])
    assert 0 <= result["suspicion_score"] <= 1
    assert isinstance(result["reliability_flags"], list)
    assert isinstance(result["conflict_flags"], list)
    assert isinstance(result["feature_evidence"], dict)


def test_invalid_image_bytes():
    result = run_agent(b"not an image", {})

    assert result["agent_id"] == "A3"
    assert isinstance(result["reliability_flags"], list)
    assert "invalid_image" in result["reliability_flags"]


def test_empty_image_bytes():
    result = run_agent(b"", {})

    assert result["agent_id"] == "A3"
    assert isinstance(result["reliability_flags"], list)
    assert len(result["reliability_flags"]) > 0


def test_invalid_metadata():
    result = run_agent(b"some image data", None)

    assert result["agent_id"] == "A3"
    assert isinstance(result["reliability_flags"], list)


def test_upstream_quality_failure(sample_image_bytes):
    result = run_agent(
        sample_image_bytes,
        {"quality_status": "FAIL"}
    )

    assert "upstream_quality_fail" in result["reliability_flags"]


def test_output_contract(sample_image_bytes):
    result = run_agent(sample_image_bytes, {})

    required_keys = {
        "agent_id",
        "suspicion_score",
        "reliability_flags",
        "conflict_flags",
        "feature_evidence"
    }

    assert required_keys.issubset(result.keys())
