# 2. Objectives

The overarching goal of this work is to develop and validate a
Hybrid Digital Twin that couples high-fidelity NLTHA with a PgNN surrogate for
real-time seismic damage prediction. Specifically, the study aims to:

1. Construct a parametric OpenSeesPy RC frame model that scales from $N=3$ to
   $N=10$ stories without manual reconfiguration.

2. Train a PgNN using FEM-derived physics tensors as training signal, targeting
   $R^2 \geq 0.70$ and RMSE below 1\% across both building heights.

3. Characterize the per-story accuracy profile and its relationship to
   higher-mode participation (the whiplash effect).

4. Demonstrate CPU inference latency compatible with real-time structural
   health monitoring ($\leq 100$ ms).
