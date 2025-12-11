"""Utility functions for working with observations."""

from app.models.structured_outputs import StepObservation


def split_observations_to_dict(
    observations: list[StepObservation],
) -> tuple[list[dict], list[dict]]:
    """
    Split observations into rules and data observations as dictionaries.

    Args:
        observations: List of StepObservation objects

    Returns:
        Tuple of (rules_list, data_observations_list) where each is a list of dicts
    """
    rules = []
    data_observations = []

    for obs in observations:
        obs_dict = {
            "step_number": obs.step_number,
            "kind": obs.kind,
            "source": obs.source,
            "title": obs.title,
            "summary": obs.summary,
            "importance": obs.importance,
            "relevance": obs.relevance,
        }

        # Add raw_output if it exists
        if obs.raw_output:
            obs_dict["raw_output"] = obs.raw_output

        if obs.kind == "rule":
            rules.append(obs_dict)
        else:
            data_observations.append(obs_dict)

    return rules, data_observations
