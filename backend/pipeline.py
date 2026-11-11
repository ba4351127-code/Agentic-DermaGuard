from importlib import import_module
from collections.abc import Mapping
from typing import Any

from backend.agents.a5_safety import review_safety
from backend.agents.a6_coordinator import coordinate


class AgentsNotReady(RuntimeError):
    pass


class AgentContractError(RuntimeError):
    pass


def _to_dict(output: Any, agent_id: str) -> dict[str, Any]:
    if hasattr(output, "model_dump"):
        output = output.model_dump()
    elif hasattr(output, "dict"):
        output = output.dict()

    if not isinstance(output, Mapping):
        raise AgentContractError(f"{agent_id} must return a dictionary-like result.")
    return dict(output)


def _run_module_agent(
    agent_id: str,
    module_path: str,
    image_bytes: bytes,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    try:
        module = import_module(module_path)
        runner = getattr(module, "run_agent")
    except (ImportError, AttributeError) as exc:
        raise AgentsNotReady(
            f"{agent_id} is unavailable. Expected {module_path}.run_agent."
        ) from exc

    return _to_dict(runner(image_bytes, metadata), agent_id)


class DermaGuardPipeline:
    def predict(
        self,
        image_bytes: bytes,
        image_id: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        a1 = _run_module_agent(
            "A1", "backend.agents.a1_quality", image_bytes, metadata
        )
        a2 = _run_module_agent(
            "A2", "backend.agents.a2_diagnosis", image_bytes, metadata
        )
        a3 = _run_module_agent(
            "A3", "backend.agents.a3_lesion", image_bytes, metadata
        )

        try:
            from backend.agents.a4_context import analyze_context

            a4 = _to_dict(
                analyze_context(
                    age=metadata.get("age"),
                    sex=metadata.get("sex"),
                    lesion_location=metadata.get("lesion_location"),
                ),
                "A4",
            )
        except (ImportError, AttributeError) as exc:
            raise AgentsNotReady("A4 context agent is unavailable.") from exc

        a5 = review_safety(a1, a2, a3, a4)
        a6 = coordinate(a1, a2, a3, a4, a5)

        probabilities = a2.get("class_probabilities", {})
        if not isinstance(probabilities, dict):
            raise AgentContractError("A2 class_probabilities must be a dictionary.")

        return {
            "image_id": image_id,
            "prediction": a6["predicted_label"],
            "decision": a6["triage_decision"],
            "confidence": a2.get("confidence"),
            "class_probabilities": probabilities,
            "agents": {
                "A1": a1,
                "A2": a2,
                "A3": a3,
                "A4": a4,
                "A5": a5,
                "A6": a6,
            },
            "rationale": a6["rationale"],
        }


pipeline = DermaGuardPipeline()