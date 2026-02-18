# 5. Discussion

## 5.1 Interpretation of Results

The Hybrid PINN (v1.6) demonstrated effective predictive capability on the complete PEER NGA-West2 campaign ($R^2 = 0.783$), improving upon the baseline v1.0 model ($R^2 = 0.772$) through architectural enhancements. The physics-based regularization term ($L_{phy} \sim 10^{-10}$) constrained predictions to physically valid solutions, acting as an inductive bias that improves generalization.

### Impact of Architectural Improvements

The move from v1.0 to v1.6 introduced (1) Temporal Self-Attention and (2) Per-story weighted loss. The results indicate:

1.  **Attention Mechanism**: By allowing the model to attend to specific temporal phases of the ground motion (e.g., peak energy arrival), the model improved its global accuracy (+1.1% $R^2$). However, the computational cost increased by ~4x during training and ~2x during inference.
2.  **Weighted Loss**: Assigning a 2.15x weight to Story 3 (which has smaller drift amplitudes) yielded a +0.9% improvement in its $R^2$. While positive, this gain is modest compared to the weighting magnitude, suggesting that the difficulty in predicting upper-story responses is not merely an optimization imbalance, but potentially an information limit (higher modes are harder to reconstruct from base excitation alone).

### Synthetic vs. Real Data Performance

The performance on real data ($R^2 = 0.783$) is now remarkably close to the synthetic baseline ($R^2 = 0.791$). This convergence suggests that the Hybrid PINN architecture is successfully capturing the complex physics of real earthquakes, rather than just overfitting to simplified synthetic signals. The increase in dataset size (21 $\to$ 289 records) was the primary driver of this robustness.

### Per-Story Accuracy

Consistent with structural dynamics theory, lower stories (dominated by the 1st mode) are predicted with higher accuracy ($R^2 \approx 0.76$). The upper story (Story 3, $R^2 = 0.55$) remains the most challenging. The v1.6 improvements confirm that while we can squeeze more performance out of the architecture, a fundamental gap remains for higher-mode dominated responses.

## 5.2 Comparison with Existing Methods

| Approach | Accuracy ($R^2$) | Latency | Data Requirement |
| :--- | :--- | :--- | :--- |
| Full FEM (OpenSeesPy) | Reference | 10–40 s/record | N/A |
| Pure LSTM | 0.70–0.85 | ~5 ms | 1000+ records |
| **Hybrid PINN (v1.6)** | **0.78** | **~2 ms** | **289 records** |
| Transfer Learning CNN | 0.80–0.90 | ~10 ms | 500+ records |

The Hybrid PINN achieves competitive accuracy with significantly fewer training samples and lower latency, enabled by the physics-informed loss function. The latency increase to 2ms (due to attention) is a negligible trade-off for the accuracy gain in real-time control contexts.

## 5.3 Practical Implications

The benchmarking results (Section 4.5) confirm low-latency inference (~2 ms), enabling:

1.  **Real-Time Damage Assessment**: Immediate post-earthquake evaluation of drift demands.
2.  **Structural Control**: Semi-active damper actuation with minimal control loop lag.
3.  **Edge Deployment**: CPU-based performance suggests feasibility on low-cost embedded hardware, although the attention layer adds memory overhead compared to the pure CNN baseline.

## 5.4 Limitations

1.  **Upper-Story Prediction**: Even with attention and heavy weighting, Story 3 accuracy ($R^2 = 0.55$) lags behind lower stories. Future work should explore adding input features like response spectra or ground velocity.
2.  **Fixed Sections**: The current 3-story model uses uniform column sections. For taller buildings ($N > 8$), variable cross-sections would be required.
3.  **2D Simplification**: Torsional effects and bidirectional ground motion components are not captured in the planar frame model.
4.  **Material Models**: The OpenSeesPy model assumes specific hysteresis rules; experimental shake table validation would strengthen the digital twin claim.
