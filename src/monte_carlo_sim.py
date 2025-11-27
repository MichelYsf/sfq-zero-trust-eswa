"""Run Monte Carlo simulations to evaluate identity policies under the SFQ metric.

This script loads simulation parameters and policy definitions from YAML files,
runs the specified number of random draws per policy and writes aggregated
results to a CSV file.  It also prints a summary table with mean SFQ scores
and 95\u00a0% confidence intervals.  By default the script uses simplified
assumptions: the baseline latency is approximated by the median of a
lognormal distribution (\u005c(\exp(\mu)\u005c)), and policy deltas and Gaussian noise
are added directly to this median and to other friction components.  Values
are clamped and normalised before computing SFQ.

Usage:

    python -m src.monte_carlo_sim

The output CSV will be saved to `data/simulation_results.csv` in the repository
root.
"""

from __future__ import annotations

import math
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm

from .sfq_calculator import normalize_metric, compute_residual_risk, compute_sfq


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file and return its contents."""
    return yaml.safe_load(path.read_text())


def run_simulation(policy_name: str, policy: Dict[str, Any], params: Dict[str, Any]) -> pd.DataFrame:
    """Run Monte Carlo simulations for a single policy.

    Args:
        policy_name: The identifier of the policy in the YAML file.
        policy: The policy definition (deltas and residual effectiveness).
        params: The simulation parameters loaded from YAML.

    Returns:
        A DataFrame with one row per run containing raw and normalised values and the SFQ.
    """
    num_runs = int(params['num_runs'])
    scenario_weights = params['scenario_weights']
    weights = params['component_weights']
    clamp = params['clamping']
    baseline = params['baseline']
    noise = params['noise']

    # Determine baseline median latency from lognormal parameters
    latency_dist = baseline['latency_distribution']
    if latency_dist['type'] != 'lognormal':
        raise ValueError(f"Unsupported latency distribution: {latency_dist['type']}")
    baseline_latency = math.exp(latency_dist['mu'])  # median of lognormal

    baseline_failure = baseline['failure_rate']
    baseline_prompts = baseline['prompts_per_user']
    baseline_helpdesk = baseline['helpdesk_per_100_users']

    # Policy deltas
    deltas = policy.get('deltas', {})
    delta_lat = deltas.get('latency', 0.0)
    delta_fail = deltas.get('failure_rate', 0.0)
    delta_prompt = deltas.get('prompts_per_user', 0.0)
    delta_helpdesk = deltas.get('helpdesk_per_100_users', 0.0)

    # Residual effectiveness for this policy
    effectiveness = policy.get('residual_effectiveness', {})

    # Precompute residual risk (constant per policy)
    residual_risk = compute_residual_risk(effectiveness, scenario_weights)

    rng = np.random.default_rng()
    records = []
    for _ in range(num_runs):
        # Draw noise for each component
        lat_noise = rng.normal(0.0, noise['latency'])
        fail_noise = rng.normal(0.0, noise['failure_rate'])
        prompt_noise = rng.normal(0.0, noise['prompts_per_user'])
        helpdesk_noise = rng.normal(0.0, noise['helpdesk_per_100_users'])

        # Raw metrics with delta and noise
        L = baseline_latency + delta_lat + lat_noise
        F = baseline_failure + delta_fail + fail_noise
        P = baseline_prompts + delta_prompt + prompt_noise
        H = baseline_helpdesk + delta_helpdesk + helpdesk_noise

        # Clamp values
        L = max(min(L, clamp['latency'][1]), clamp['latency'][0])
        F = max(min(F, clamp['failure_rate'][1]), clamp['failure_rate'][0])
        P = max(min(P, clamp['prompts_per_user'][1]), clamp['prompts_per_user'][0])
        H = max(min(H, clamp['helpdesk_per_100_users'][1]), clamp['helpdesk_per_100_users'][0])

        # Normalise
        L_norm = normalize_metric(L, *clamp['latency'])
        F_norm = normalize_metric(F, *clamp['failure_rate'])
        P_norm = normalize_metric(P, *clamp['prompts_per_user'])
        H_norm = normalize_metric(H, *clamp['helpdesk_per_100_users'])

        sfq_value = compute_sfq(L_norm, F_norm, P_norm, H_norm, residual_risk, weights)

        records.append({
            'policy': policy_name,
            'latency': L,
            'failure_rate': F,
            'prompts_per_user': P,
            'helpdesk_per_100_users': H,
            'residual_risk': residual_risk,
            'latency_norm': L_norm,
            'failure_norm': F_norm,
            'prompts_norm': P_norm,
            'helpdesk_norm': H_norm,
            'sfq': sfq_value,
        })

    return pd.DataFrame.from_records(records)


def summarise_results(df: pd.DataFrame, baseline_policy: str) -> pd.DataFrame:
    """Compute summary statistics and effect sizes per policy.

    Args:
        df: DataFrame containing per‑run metrics for all policies.
        baseline_policy: Name of the baseline policy used for effect sizes.

    Returns:
        A summary DataFrame indexed by policy name with columns:
            mean_sfq, ci_lower, ci_upper, effect_size_d.
    """
    summary = []
    baseline_sfq = df[df['policy'] == baseline_policy]['sfq']
    baseline_mean = baseline_sfq.mean()
    baseline_var = baseline_sfq.var(ddof=1)
    baseline_n = len(baseline_sfq)
    for policy in df['policy'].unique():
        sfq_vals = df[df['policy'] == policy]['sfq']
        mean_val = sfq_vals.mean()
        # 95 % CI from empirical quantiles
        ci_lower = sfq_vals.quantile(0.025)
        ci_upper = sfq_vals.quantile(0.975)
        # Effect size vs baseline (Cohen's d)
        if policy == baseline_policy:
            effect_size = 0.0
        else:
            var1 = baseline_var
            var2 = sfq_vals.var(ddof=1)
            n1 = baseline_n
            n2 = len(sfq_vals)
            pooled_sd = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            effect_size = (mean_val - baseline_mean) / pooled_sd if pooled_sd > 0 else float('nan')
        summary.append({
            'policy': policy,
            'mean_sfq': mean_val,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'effect_size_d': effect_size,
        })
    return pd.DataFrame(summary).sort_values('mean_sfq')


def main() -> None:
    """Entry point for running the full simulation across all policies."""
    repo_root = Path(__file__).resolve().parents[1]
    param_path = repo_root / 'data' / 'simulation_parameters.yaml'
    policy_path = repo_root / 'src' / 'policy_definitions.yaml'
    output_path = repo_root / 'data' / 'simulation_results.csv'

    params = load_yaml(param_path)
    policies = load_yaml(policy_path)

    all_results = []
    print(f"Running {params['num_runs']} simulations per policy...")
    for name, policy in policies.items():
        print(f"Simulating policy: {name}")
        res = run_simulation(name, policy, params)
        all_results.append(res)
    df = pd.concat(all_results, ignore_index=True)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    # Summarise and display results
    baseline_policy = 'baseline_password_only'
    summary_df = summarise_results(df, baseline_policy)
    print("\nSFQ Summary (mean \u00b195% CI and effect size vs. baseline):")
    for _, row in summary_df.iterrows():
        print(
            f"{row['policy']:>25}: {row['mean_sfq']:.3f} "
            f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}] "
            f"d={row['effect_size_d']:.3f}"
        )


if __name__ == '__main__':
    main()
