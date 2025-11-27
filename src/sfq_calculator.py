"""Compute the Security Friction Quotient (SFQ).

This module defines functions for normalising raw friction metrics, computing
residual risk from policy effectiveness values, and combining all components
into the Security Friction Quotient.  It assumes that all inputs have been
clamped to the ranges defined in the simulation parameters before
normalisation.

Example usage:

>>> from yaml import safe_load
>>> from pathlib import Path
>>> import numpy as np
>>> params = safe_load(Path('data/simulation_parameters.yaml').read_text())
>>> policies = safe_load(Path('src/policy_definitions.yaml').read_text())
>>> scenario_weights = params['scenario_weights']
>>> baseline_clamp = params['clamping']
>>> component_weights = params['component_weights']
>>> # compute residual risk for baseline policy
>>> R = compute_residual_risk(policies['baseline_password_only']['residual_effectiveness'], scenario_weights)
>>> # normalise friction components and compute SFQ
>>> L_norm = normalize_metric(1.5, *baseline_clamp['latency'])
>>> F_norm = normalize_metric(0.02, *baseline_clamp['failure_rate'])
>>> P_norm = normalize_metric(0.3, *baseline_clamp['prompts_per_user'])
>>> H_norm = normalize_metric(12.8, *baseline_clamp['helpdesk_per_100_users'])
>>> sfq = compute_sfq(L_norm, F_norm, P_norm, H_norm, R, component_weights)
"""

from __future__ import annotations

from typing import Dict, Mapping


def normalize_metric(value: float, min_value: float, max_value: float) -> float:
    """Normalise a metric to the range [0, 1] given its empirical min and max.

    Values outside the provided range are clipped to the boundaries.

    Args:
        value: The raw metric value (already clamped if needed).
        min_value: The lower bound for the metric.
        max_value: The upper bound for the metric.

    Returns:
        A float in [0, 1] representing the normalised value.
    """
    if max_value <= min_value:
        raise ValueError(f"Invalid normalisation range: {min_value} >= {max_value}")
    clipped = max(min(value, max_value), min_value)
    return (clipped - min_value) / (max_value - min_value)


def compute_residual_risk(
    effectiveness: Mapping[str, float],
    scenario_weights: Mapping[str, float]
) -> float:
    """Compute the residual risk index R for a given policy.

    Residual risk is calculated as the weighted sum of per‑scenario residual
    probabilities, where residual = 1 - effectiveness.  Scenario weights should
    sum to 1.0.  Policies that mitigate attacks more effectively will have
    lower residual risk.

    Args:
        effectiveness: A mapping from scenario name to mitigation effectiveness
            between 0 and 1.
        scenario_weights: A mapping from scenario name to prevalence weight.

    Returns:
        Residual risk R in the interval [0, 1].
    """
    residual = 0.0
    total_weight = 0.0
    for scenario, weight in scenario_weights.items():
        total_weight += weight
        eff = effectiveness.get(scenario, 0.0)
        eff = max(0.0, min(1.0, eff))  # clamp to [0, 1]
        residual += (1.0 - eff) * weight
    if abs(total_weight - 1.0) > 1e-6:
        # Normalise weights if they do not sum exactly to one due to rounding
        residual /= total_weight
    return residual


def compute_sfq(
    latency_norm: float,
    failure_norm: float,
    prompts_norm: float,
    helpdesk_norm: float,
    residual_risk: float,
    weights: Mapping[str, float]
) -> float:
    """Compute the Security Friction Quotient given normalised components.

    Args:
        latency_norm: Normalised latency (\hat{L}).
        failure_norm: Normalised failure rate (\hat{F}).
        prompts_norm: Normalised MFA prompt frequency (\hat{P}).
        helpdesk_norm: Normalised helpdesk ticket rate (\hat{H}).
        residual_risk: Residual risk index R \in [0,1].  In the formula this
            enters as (1 - \hat{R}).
        weights: A mapping specifying the weight for each component.  Keys
            should include 'latency', 'failure', 'prompts', 'helpdesk' and
            'residual'.

    Returns:
        The SFQ value in the interval [0, 1].
    """
    # Extract weights with defaults if missing
    w_L = weights.get('latency', 0.0)
    w_F = weights.get('failure', 0.0)
    w_P = weights.get('prompts', 0.0)
    w_H = weights.get('helpdesk', 0.0)
    w_R = weights.get('residual', 0.0)
    # Ensure weights sum to one
    total = w_L + w_F + w_P + w_H + w_R
    if abs(total - 1.0) > 1e-6:
        # Normalise weights if necessary
        w_L /= total
        w_F /= total
        w_P /= total
        w_H /= total
        w_R /= total
    sfq = (
        w_L * latency_norm
        + w_F * failure_norm
        + w_P * prompts_norm
        + w_H * helpdesk_norm
        + w_R * (1.0 - residual_risk)
    )
    return sfq
