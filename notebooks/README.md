# Notebooks

This directory is intended to contain Jupyter notebooks that demonstrate how to
load the simulation results and recreate the main figures and tables from the
SFQ manuscript.  A recommended notebook, `reproduce_figures.ipynb`, should
include the following cells:

1. **Imports and configuration**
   ```python
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   import seaborn as sns

   sns.set(style="whitegrid")
   ```

2. **Load results**
   ```python
   results = pd.read_csv('../data/simulation_results.csv')
   # Ensure policies are categorical with desired order
   policy_order = [
       'baseline_password_only',
       'risk_based_mfa',
       'device_compliance',
       'phishing_resistant_mfa',
       'combined_controls',
   ]
   results['policy'] = pd.Categorical(results['policy'], categories=policy_order, ordered=True)
   ```

3. **Figure 1: SFQ by policy (95 % CI)**
   ```python
   # Compute mean and quantiles
   summary = results.groupby('policy')['sfq'].agg(['mean', lambda x: x.quantile(0.025), lambda x: x.quantile(0.975)])
   summary.columns = ['mean', 'ci_lower', 'ci_upper']
   summary = summary.reset_index()

   plt.figure(figsize=(8, 5))
   plt.bar(summary['policy'], summary['mean'], color='darkorange', yerr=[
       summary['mean'] - summary['ci_lower'],
       summary['ci_upper'] - summary['mean']],
       capsize=4)
   plt.ylabel('SFQ')
   plt.xlabel('Policy')
   plt.title('SFQ by Policy (95% CI)')
   plt.xticks(rotation=30, ha='right')
   plt.tight_layout()
   ```

4. **Figure 2: Rank stability under Dirichlet weight draws**
   ```python
   # Draw random weight vectors from a symmetric Dirichlet distribution
   n_draws = 10000
   alpha = np.ones(5)  # equal concentration
   rng = np.random.default_rng()
   weight_draws = rng.dirichlet(alpha, size=n_draws)

   # Compute SFQ ranks for each draw
   policies = summary['policy'].tolist()
   ranks_preserved = np.zeros(len(policies))
   for w in weight_draws:
       # Map order: latency, failure, prompts, helpdesk, residual
       comp_keys = ['latency_norm', 'failure_norm', 'prompts_norm', 'helpdesk_norm', 'residual_risk']
       # For each policy compute weighted SFQ with random weights
       scores = []
       for pol in policies:
           subset = results[results['policy'] == pol]
           # Use mean components as representative
           mean_vals = subset[comp_keys].mean()
           # SFQ under this draw (note residual enters as 1 - R)
           score = (
               w[0] * mean_vals['latency_norm'] +
               w[1] * mean_vals['failure_norm'] +
               w[2] * mean_vals['prompts_norm'] +
               w[3] * mean_vals['helpdesk_norm'] +
               w[4] * (1.0 - mean_vals['residual_risk'])
           )
           scores.append(score)
       # Determine ordering
       order = np.argsort(scores)
       # Count if ordering matches equal‑weight ordering
       if list(order) == list(range(len(policies))):
           ranks_preserved += 1
   stability = ranks_preserved / n_draws

   plt.figure(figsize=(8, 5))
   plt.bar(policies, stability, color='darkorange')
   plt.ylabel('Probability of Rank Preserved')
   plt.xlabel('Policy')
   plt.title('Rank Stability under Random Weights')
   plt.xticks(rotation=30, ha='right')
   plt.tight_layout()
   ```

5. **Figure 3: Tornado analysis**
   ```python
   # Perturb each component across its range and measure impact on ranking
   components = ['latency_norm', 'failure_norm', 'prompts_norm', 'helpdesk_norm', 'residual_risk']
   baseline_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
   impact = []
   for comp in components:
       # vary component up and down by 10 % of its range
       delta = 0.1
       scores_plus = []
       scores_minus = []
       for pol in policies:
           vals = results[results['policy'] == pol][components].mean()
           # plus perturbation (except residual, which is reversed)
           perturbed = vals.copy()
           if comp == 'residual_risk':
               # residual enters as (1-R), so increasing R lowers score
               perturbed[comp] = max(0.0, vals[comp] - delta)
           else:
               perturbed[comp] = min(1.0, vals[comp] + delta)
           score_plus = (
               baseline_weights[0] * perturbed['latency_norm'] +
               baseline_weights[1] * perturbed['failure_norm'] +
               baseline_weights[2] * perturbed['prompts_norm'] +
               baseline_weights[3] * perturbed['helpdesk_norm'] +
               baseline_weights[4] * (1.0 - perturbed['residual_risk'])
           )
           # minus perturbation
           perturbed2 = vals.copy()
           if comp == 'residual_risk':
               perturbed2[comp] = min(1.0, vals[comp] + delta)
           else:
               perturbed2[comp] = max(0.0, vals[comp] - delta)
           score_minus = (
               baseline_weights[0] * perturbed2['latency_norm'] +
               baseline_weights[1] * perturbed2['failure_norm'] +
               baseline_weights[2] * perturbed2['prompts_norm'] +
               baseline_weights[3] * perturbed2['helpdesk_norm'] +
               baseline_weights[4] * (1.0 - perturbed2['residual_risk'])
           )
           scores_plus.append(score_plus)
           scores_minus.append(score_minus)
       # Measure average absolute change in ranking position
       rank_change = np.abs(np.array(scores_plus).argsort() - np.array(scores_minus).argsort()).mean()
       impact.append(rank_change)

   plt.figure(figsize=(8, 5))
   plt.bar(components, impact, color='darkorange')
   plt.ylabel('Impact on Ranking (a.u.)')
   plt.title('Tornado Analysis of SFQ Components')
   plt.xticks(rotation=30, ha='right')
   plt.tight_layout()
   ```

The notebook should conclude with a discussion of the results and optional export of the figures to the `figures/` directory.
