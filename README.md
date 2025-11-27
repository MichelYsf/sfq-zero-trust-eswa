# Security Friction Quotient (SFQ) – Zero Trust Identity Policy

This repository contains all of the code, configuration and documentation needed to reproduce the results in the paper:

> **Security Friction Quotient for Zero Trust Identity Policy with Empirical Validation**  
> Michel Youssef, 2025 (submitted to *Expert Systems With Applications*)

The Security Friction Quotient (SFQ) is a simple yet powerful metric for comparing identity‑centric access control policies in Zero Trust architectures.  It combines five measurable dimensions—authentication latency, failure rate, multi‑factor authentication (MFA) prompt frequency, helpdesk workload and residual risk—into a single score in the range \([0,1]\).  Higher SFQ values correspond to greater overall “friction” (worse risk or usability), while lower values indicate smoother, safer authentication.

## Summary for Non‑specialists

*Zero Trust* is an approach to enterprise security in which every access request is continuously verified.  Policies such as “password only”, “risk‑based MFA”, “device compliance required” and “phishing‑resistant passkeys” offer different trade‑offs between security and user convenience.  SFQ condenses these trade‑offs into a single number by measuring how often users struggle to sign in, how often support teams need to intervene and how much risk remains against common attack scenarios.  A lower SFQ means the policy is both secure and user friendly.

## Summary for Researchers

The SFQ metric is defined as a weighted sum of normalized components:

\[\mathrm{SFQ} = w_L \hat{L} + w_F \hat{F} + w_P \hat{P} + w_H \hat{H} + w_R (1 - \hat{R}),\]

where \(\hat{L}, \hat{F}, \hat{P}, \hat{H}\) are the latency, failure, prompt and helpdesk metrics normalized to \([0,1]\), \(\hat{R}\) is the residual risk index, and the weights \(w_i\) sum to one.  Residual risk is calculated from attack scenarios (password spraying, credential theft, travel, legacy factors and adversary‑in‑the‑middle) with adjustable prevalence weights.  This repository contains a Monte Carlo simulation engine that synthesizes authentication traces for a cohort of 1 200 users over a 12‑week horizon, perturbs each component with noise and policy‑specific deltas, and computes SFQ with confidence intervals.  The provided notebooks reproduce the bar charts, rank‑stability and tornado diagrams in the manuscript.

## Repository Structure

| Path | Purpose |
|---|---|
| `src/` | Python source code implementing SFQ computation (`sfq_calculator.py`), the Monte Carlo simulation (`monte_carlo_sim.py`) and machine‑readable policy definitions (`policy_definitions.yaml`). |
| `data/` | Configuration for the simulation (`simulation_parameters.yaml`) and (optionally) aggregated simulation outputs. |
| `notebooks/` | Jupyter notebooks for analysing simulation output and regenerating Figures 1–3. |
| `figures/` | Static images used in the manuscript (included here for reference). |
| `docs/` | Manuscript source files (e.g., LaTeX) or additional documentation. |
| `LICENSE` | MIT licence for this repository. |
| `CITATION.cff` | Citation metadata describing how to cite this work. |
| `requirements.txt` | Python package dependencies. |
| `.gitignore` | Standard Git ignore patterns for Python projects. |

## Installation

1. Ensure you have **Python 3.9** or later installed.
2. Clone this repository:

   ```bash
   git clone https://github.com/USER/sfq-zero-trust-eswa.git
   cd sfq-zero-trust-eswa
   ```

3. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

## How to Reproduce the Results

1. **Configure the simulation.**  The file `data/simulation_parameters.yaml` defines the number of users, simulation horizon, baseline distributions, clamping ranges, noise parameters and scenario weights.  Policy deltas and scenario effectiveness values are defined in `src/policy_definitions.yaml`.  You can adjust these YAML files to explore different settings.

2. **Run the Monte Carlo simulation.**  Execute the simulation script from the command line:

   ```bash
   python -m src.monte_carlo_sim
   ```

   This script will load the parameters, run the specified number of simulations for each policy and save a CSV file (e.g., `data/simulation_results.csv`) containing per‑run metrics and SFQ scores.  It will also print a summary table with mean SFQ and confidence intervals.

3. **Reproduce figures and tables.**  Start a Jupyter notebook server and open the notebook provided in `notebooks/reproduce_figures.ipynb`.  The notebook contains code cells to:

   - Read the simulation results CSV.
   - Compute aggregated statistics and effect sizes.
   - Plot **Figure 1** (SFQ by policy with 95 % CI), **Figure 2** (rank stability under Dirichlet weight draws) and **Figure 3** (tornado analysis of component sensitivity).

   Alternatively, you can adapt the code in the notebook to your own analysis scripts.

## Limitations and Notes

- The synthetic data generated here reflects general enterprise patterns (e.g., sign‑in frequency, latency distribution) and relies on assumed noise parameters.  Actual environments may differ substantially.
- Residual risk estimates are based on high‑level effectiveness assumptions; they should be refined when better empirical data is available.
- Equal weights (\(w_i = 0.2\)) are used by default.  Sensitivity analysis under varying weights is included in the notebook.

## Citation

If you use this code or reproduce results from the paper, please cite it as follows:

```
@article{youssef2025sfq,
  author  = {Michel Youssef},
  title   = {Security Friction Quotient for Zero Trust Identity Policy with Empirical Validation},
  year    = {2025},
  journal = {Expert Systems With Applications},
  note    = {Submitted},
  url     = {https://github.com/USER/sfq-zero-trust-eswa}
}
```

More formal citation metadata is available in the `CITATION.cff` file.