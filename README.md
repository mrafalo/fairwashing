## Adversarial Attacks on Feature Attribution in Malware Classification

This repository accompanies the paper:

> **Breaking the Explanation Layer: Adversarial Attacks in Malware Classification**

This project investigates whether malware samples can **preserve their classification outcome while manipulating reported feature importance**, thereby misleading human analysts — a phenomenon closely related to explanation-layer fairwashing.

---

## 🔍 Research Objective

We evaluate whether adversaries can:

- Maintain classifier predictions
- Substantially alter feature-attribution explanations
- Suppress security-relevant indicators
- Preserve structural validity constraints

Formally, given a classifier \( f\_\theta \) and explainer \( \mathcal{E} \), we seek:

\[
f*\theta(\tilde{x}) = f*\theta(x)
\]

while maximizing explanation drift:

\[
D(\mathcal{E}(f*\theta,\tilde{x}), \mathcal{E}(f*\theta,x))
\]

where \( D(\cdot,\cdot) \) measures attribution divergence.

---

## 🧠 Attack Surfaces Implemented

- Model-Level (Surrogate-Based) Attack
- Explanation-Level Attack
- Input-Level Manifold-Constrained Attack
