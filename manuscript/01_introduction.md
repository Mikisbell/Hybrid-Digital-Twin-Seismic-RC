# 1. Introduction

<!-- HRPUB Section: Introduction -->
<!-- Include: Background, literature review, research gap, contribution -->

## 1.1 Background

Reinforced concrete (RC) buildings exhibit complex nonlinear behavior under seismic excitation. While high-fidelity Finite Element Models (FEM) using frameworks like OpenSeesPy can accurately predict this response, the computational cost of Nonlinear Time History Analysis (NLTHA) becomes prohibitive for Probabilistic Seismic Demand Analysis (PSDA), which often requires thousands of simulations [1].

## 1.2 Literature Review

<!-- Review of:
- Computational cost of NLTHA for large-scale parametric studies
- Surrogate modeling (metamodeling) in structural engineering
- Physics-Informed Neural Networks (PINNs) as next-gen surrogates
- The gap between "Toy Problems" and "High-Fidelity FEM Emulation"
-->

## 1.3 Research Gap

Despite advances in machine learning, most surrogate models for seismic response are:
1.  **Black-box**: Lacking physical interpretability and adherence to laws of motion
2.  **Scalar-only**: Predicting only peak quantities (e.g., max drift) rather than full time histories
3.  **Non-Parametric**: Fixed to a specific building geometry, unable to generalize to N-story variations

## 1.4 Contribution

This paper presents a **Hybrid Digital Twin of the FEM**—a parametric, physics-informed surrogate model that:
1.  Emulates the full nonlinear displacement history $u(t)$ of N-story RC frames
2.  Reduces computational time by orders of magnitude compared to direct integration
3.  Enforces kinematic consistency via a novel "Kinematic-Informed Loss" function

---

<!-- References will use numeric correlative format: [1], [2], ... -->
