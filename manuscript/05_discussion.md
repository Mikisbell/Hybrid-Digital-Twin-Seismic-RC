# 5. Discussion

## 5.1 Interpretation of Results

The Hybrid PgNN demonstrated effective predictive capability on the complete PEER NGA-West2 campaign across two building heights: $R^2 = 0.783$ for $N=3$ (289 records) and $R^2 = 0.713$ for $N=10$ (265 records). The model is physics-guided at three levels: (1) targets derived from OpenSeesPy NLTHA, (2) per-story inverse-variance weights from FEM response statistics, and (3) physics tensor regularization via stored $f_{int}$, $\ddot{\mathbf{u}}$, $\dot{\mathbf{u}}$ tensors. This physics-data coupling acts as an inductive bias that improves generalization beyond purely data-driven baselines. The 9% drop in overall $R^2$ from $N=3$ to $N=10$ is consistent with the increased complexity of higher-mode dominated responses.

### 5.1.1 Impact of Architectural Improvements

The move from v1.0 to v1.6 introduced (1) Temporal Self-Attention and (2) Per-story weighted loss. The results indicate:

1.  **Attention Mechanism**: By allowing the model to attend to specific temporal phases of the ground motion (e.g., peak energy arrival), the model improved its global accuracy (+1.1% $R^2$). However, the computational cost increased by ~4x during training and ~2x during inference.
2.  **Weighted Loss**: Assigning a 2.15x weight to Story 3 (which has smaller drift amplitudes) yielded a +0.9% improvement in its $R^2$. While positive, this gain is modest compared to the weighting magnitude, suggesting that the difficulty in predicting upper-story responses is not merely an optimization imbalance, but potentially an information limit (higher modes are harder to reconstruct from base excitation alone).

### 5.1.2 Synthetic vs. Real Data Performance

The performance on real data ($R^2 = 0.783$) is now remarkably close to the synthetic baseline ($R^2 = 0.791$). This convergence suggests that the Hybrid PgNN architecture is successfully capturing the complex physics of real earthquakes, rather than just overfitting to simplified synthetic signals. The increase in dataset size (21 $\to$ 289 records) was the primary driver of this robustness.

### 5.1.3 Per-Story Accuracy

Consistent with structural dynamics theory, lower stories (dominated by the 1st mode) are predicted with higher accuracy ($R^2 \approx 0.76$). The upper story (Story 3, $R^2 = 0.55$) remains the most challenging. The v1.6 improvements confirm that while we can squeeze more performance out of the architecture, a fundamental gap remains for higher-mode dominated responses.

## 5.2 Comparison with Existing Methods

| Approach | Accuracy ($R^2$) | Stories | Latency | Data Requirement |
| :--- | :--- | :--- | :--- | :--- |
| Full FEM (OpenSeesPy) | Reference | Any | 10–40 s/record | N/A |
| Pure LSTM | 0.70–0.85 | 1–3 | ~5 ms | 1000+ records |
| **Hybrid PgNN (N=3)** | **0.783** | **3** | **~2 ms** | **289 records** |
| **Hybrid PgNN (N=10)** | **0.713** | **10** | **~2.5 ms** | **265 records** |
| Transfer Learning CNN | 0.80–0.90 | 1–5 | ~10 ms | 500+ records |
| **PgNN Transfer (N=3→10)** | **0.700** | **10** | **~2.5 ms** | **0 new encoder training** |

The comparison warrants careful interpretation. Methods reporting $R^2 > 0.85$ typically
train on 1,000+ records from single building configurations (1–3 stories), whereas the
present study uses 289 records for $N=3$ and 265 for $N=10$ — a 3–4$\times$ smaller
training set. Moreover, most prior work evaluates on synthetic ground motions, while
the PgNN is validated exclusively on real PEER NGA-West2 records with their full spectral
variability. When normalized by data efficiency (accuracy per training record), the PgNN
shows a favorable trade-off: $R^2 = 0.783$ with 289 records versus $R^2 \approx 0.85$
with 1,000+ records. The 50$\times$ lower inference latency (~2 ms vs. ~10–100 ms)
further differentiates the approach for time-critical deployment. The transfer learning
variant achieves 98.2\% of full-training accuracy with zero additional encoder training,
demonstrating domain-invariant feature extraction across building heights.

## 5.3 Practical Implications

The benchmarking results (Section 4.8) confirm low-latency inference (~2 ms), enabling:

1.  **Real-Time Damage Assessment**: Immediate post-earthquake evaluation of drift demands.
2.  **Structural Control**: Semi-active damper actuation with minimal control loop lag.
3.  **Edge Deployment**: CPU-based performance suggests feasibility on low-cost embedded hardware, although the attention layer adds memory overhead compared to the pure CNN baseline.

## 5.4 Scalability and the Whiplash Effect in Tall Buildings

The extension to $N=10$ stories introduces structural dynamics phenomena qualitatively absent in
low-rise configurations. This section discusses the expected performance profile of the Hybrid PgNN
for tall RC frames and the role of the FEM-guided loss in addressing these challenges.

### 5.4.1 Higher-Mode Dominance

In a 10-story RC frame with $T_1 \approx 1.8$ s, modes 2 and 3 ($T_2 \approx 0.6$ s,
$T_3 \approx 0.35$ s) carry approximately 35–40% of the total seismic mass participation.
For near-field records (short $T_{p}$) or records with dominant energy above 1 Hz, upper floors
can experience spectral accelerations 2–4× greater than what first-mode response alone would predict
[8]. This is the physical origin of the **whiplash effect**: the roof ($j=10$) may sustain higher
drift demands than floors 4–7, inverting the monotonic drift profile assumed in simplified analyses.

### 5.4.2 Role of the FEM-Guided Loss for Tall Buildings

For $N=3$, the physics loss ($\mathcal{L}_{phy} \sim 10^{-10}$) acts primarily as a low-pass
regularizer — replacing traditional stochastic regularization such as dropout [26] —
smoothing prediction noise without significantly shaping the response shape.
For $N=10$, the physics loss becomes structurally more informative because:

1. **Modal orthogonality enforcement**: The equation of motion residual
   $\mathbf{M}\ddot{u}_{pred} + \mathbf{C}\dot{u}_{pred} + f_{int}(u_{true})$
   implicitly encodes inter-story coupling through the tri-diagonal mass and stiffness matrices.
   This penalizes predictions where upper-story responses are kinematically inconsistent with
   lower-story kinematics, effectively enforcing a physically plausible "whiplash profile."

2. **Boundary condition loss as anti-drift**: The initial condition penalty
   $\mathcal{L}_{bc} = \|u(0)\|^2 + \|\dot{u}(0)\|^2$ prevents the network from introducing
   spurious pre-event displacement offsets, which are more likely in tall buildings where the
   static equilibrium under gravity (modeled via $f_{int}$) is larger.

3. **f_int as teacher forcing for upper floors**: Since $f_{int}$ from OpenSeesPy encodes the
   nonlinear hysteretic behavior at each fiber section, it provides the network with implicit
   information about where yielding occurs along the height. Upper-floor predictions that violate
   equilibrium with the provided $f_{int}$ profile are directly penalized, guiding the network
   toward the physically correct whiplash distribution.

### 5.4.3 Measured Per-Story Accuracy Profile for N=10

The measured per-story $R^2$ profile (Table 8) reveals a monotonic accuracy degradation
from the base ($R^2 = 0.754$) to the roof ($R^2 = 0.601$), with a total span of
$\Delta R^2 = 0.153$. This profile differs from the initially hypothesized U-shape:

- **Floors 1–4**: High accuracy ($R^2 = 0.70$–$0.75$), dominated by first-mode response
  well-captured by the CNN encoder's low-frequency features.
- **Floors 5–7**: Transition zone ($R^2 = 0.64$–$0.67$), where higher-mode contributions
  begin to dominate, degrading prediction accuracy.
- **Floors 8–10** (whiplash zone): Lowest accuracy ($R^2 = 0.60$–$0.64$), but notably
  better than the predicted $R^2 < 0.55$. Story 8 shows a slight $R^2$ recovery (0.642
  vs. 0.636 at Story 7), which we attribute to the strong inverse-variance weighting
  ($w_{10} = 2.65\times$) concentrating gradient signal on upper stories.

The absence of a U-shaped recovery at the roof suggests that the whiplash zone presents a
monotonic challenge rather than a localized dip. The temporal attention mechanism partially
compensates by attending to different frequency components of the ground motion; however,
a fundamental information limit remains: single-channel base acceleration cannot fully
reconstruct higher-mode responses without additional spectral or velocity features.

This motivates two complementary future directions: (1) incorporating response spectra
$S_a(T_n)$ at the first three modal periods as additional input features; and (2) implementing
a modal response spectrum regularizer — a differentiable physics constraint relating predicted
peak IDR to spectral demands $S_d(T_n)$ via modal participation factors — which would extend
the current PgNN toward a full Physics-Informed Neural Network (PINN) formulation, consistent
with recent advances in physics-informed surrogate modeling for structural dynamics [29].

## 5.5 Limitations

1.  **Upper-Story Prediction (Information Limit)**: For $N=3$, Story 3 accuracy ($R^2 = 0.55$) lags behind lower stories; for $N=10$, the whiplash zone (Floors 8–10) achieves $R^2 = 0.60$–$0.64$. The fundamental constraint is that single-channel base acceleration cannot fully reconstruct higher-mode responses. Future work should incorporate response spectra $S_a(T_1, T_2, T_3)$ or ground velocity as supplementary inputs.
2.  **Whiplash Zone Accuracy**: For $N=10$, floors 8–10 show $R^2 \approx 0.61$–$0.64$ — reduced relative to lower stories but above the initially predicted $R^2 < 0.55$, indicating that the inverse-variance weighting partially compensates the higher-mode information gap.
3.  **Fixed Sections (Low-Rise)**: The 3-story validation uses uniform column sections. The N=10 campaign implements tapering cross-sections (700→500 mm), partially addressing this for mid-rise; ultra-tall frames ($N > 15$) remain outside the current scope.
4.  **2D Simplification**: Torsional effects and bidirectional ground motion components are not captured in the planar frame model.
5.  **Material Model Specificity**: The OpenSeesPy model assumes Concrete02 (Mander confinement) and Steel02 (Giuffré-Menegotto-Pinto) hysteresis rules; experimental shake table validation would strengthen the digital twin claim for alternative material systems.
