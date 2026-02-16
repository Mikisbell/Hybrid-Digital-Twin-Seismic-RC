# 3. Methods

<!-- HRPUB Section: Methods -->
<!-- CRITICAL: HRPUB requires complete methodology description including
     data processing, statistical tests, and computational procedures. -->

## 3.1 Structural Model (OpenSeesPy)

### 3.1.1 Geometry and Loading

- 5-story, 3-bay RC frame
- Story height: 3.0 m (typical), 3.5 m (first story)
- Bay width: 5.0 m
- Dead load: 25 kN/m², Live load: 2.0 kN/m² (per ACI 318-19)

### 3.1.2 Material Models

| Material | Model | Key Parameters |
|----------|-------|----------------|
| Concrete (confined) | Concrete02 | f'c = 28 MPa, εc0 = 0.002, εcu = 0.006 |
| Concrete (unconfined) | Concrete02 | f'c = 28 MPa, εc0 = 0.002, εcu = 0.004 |
| Reinforcing steel | Steel02 | fy = 420 MPa, E = 200 GPa, b = 0.01 |

### 3.1.3 Element Formulation

- `forceBeamColumn` with fiber-section discretization
- 5 Gauss-Lobatto integration points per element
- P-Delta geometric transformation

### 3.1.4 Damping

- Rayleigh damping: ξ = 5% for modes 1 and 3
- Mass-proportional and stiffness-proportional coefficients

## 3.2 Ground Motion Selection (PEER NGA-West2)

- Database: PEER NGA-West2 [2]
- Selection criteria: Magnitude 6.0–7.5, Rjb 10–50 km
- Number of records: ≥200 (two horizontal components each, yielding 400+ time histories)
- Scaling: Spectrum-compatible to design spectrum (ASCE 7-22)
- Rationale: 200+ records provide statistical robustness for ML training and capture the aleatory variability inherent in seismic ground motions

## 3.3 Data Processing Pipeline

```
Raw NLTHA Output → Feature Extraction → Normalization → Train/Val/Test Split
      │                    │                  │                │
  Time series         IDR, PFA,          Min-Max or        70/15/15
  (disp, accel,       Sa, Sd,           StandardScaler
   force, drift)      Arias intensity
```

### 3.3.1 Feature Engineering

- Input features: Ground motion intensity measures (PGA, PGV, Sa(T1), Arias)
- Output targets: Maximum inter-story drift ratio per story
- Temporal features: Acceleration time series (windowed)

### 3.3.2 Statistical Validation

- Kolmogorov-Smirnov test for distribution normality
- Pearson correlation matrix for feature selection
- Cross-validation: 5-fold stratified by intensity level

## 3.4 Physics-Informed Neural Network (PINN)

### 3.4.1 Architecture

- Encoder: 1D-CNN for time series feature extraction
- Core: Fully connected layers (256-128-64-32)
- Activation: Swish (β=1)
- Output: IDR predictions per story (5 outputs)

### 3.4.2 Loss Function

The total loss embeds the equation of motion as a physics constraint:

$$\mathcal{L}_{total} = \mathcal{L}_{data} + \lambda_p \left\| M\ddot{u} + C\dot{u} + f_{int}(u, \dot{u}) + M\iota\ddot{u}_g \right\|^2 + \lambda_b \mathcal{L}_{bc}$$

Where:
- $\mathcal{L}_{data}$: Mean squared error between predicted and simulated IDR
- $M$, $C$: Mass and damping matrices (constant)
- $f_{int}(u, \dot{u})$: **Nonlinear restoring force** vector from OpenSeesPy, replacing the constant stiffness $K$. For RC elements with Concrete02/Steel02, this captures cracking, yielding, and cyclic degradation — essential for the "Hybrid" nature of the Digital Twin
- $\ddot{u}$, $\dot{u}$, $u$: Acceleration, velocity, displacement vectors
- $\ddot{u}_g$: Ground acceleration input
- $\iota$: Influence vector
- $\lambda_p$: Physics loss weight (tunable hyperparameter)
- $\lambda_b$: Boundary condition loss weight
- $\mathcal{L}_{bc}$: Boundary/initial conditions: $u(0)=0$, $\dot{u}(0)=0$

> **Note**: The use of $f_{int}(u, \dot{u})$ instead of $Ku$ is critical. For nonlinear RC structures, the tangent stiffness varies with displacement history. The restoring force is computed by OpenSeesPy at each time step, providing the ground truth that the PINN must learn to approximate.

### 3.4.3 Training Protocol

- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- Scheduler: Cosine annealing with warm restarts
- Epochs: 500 (early stopping patience=50)
- Batch size: 64

## 3.5 Benchmarking Protocol

Real-time applicability requires inference latency ≤ 100 ms. The benchmarking
script (`src/pinn/benchmark_latency.py`) measures:

1. **Cold start**: First inference after model load
2. **Warm inference**: Mean of 1000 consecutive predictions
3. **Batch throughput**: Predictions per second at batch sizes 1, 8, 32, 128
4. **Hardware**: CPU-only (deployment target) and GPU (training)
