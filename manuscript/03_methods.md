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

The Hybrid-PINN architecture is designed as a temporal encoder followed by a
regression head.  Rather than embedding physics directly into the network
topology, the physics constraint is enforced through a composite loss function
that penalizes violations of the equation of motion — an approach consistent
with the foundational PINN framework of Raissi et al. [3].  The complete model
contains **603,653 trainable parameters**, implemented in PyTorch [13] and
available in the project repository (`src/pinn/model.py`).

### 3.4.1 Architecture

The network comprises two sequential stages: (i) a one-dimensional
convolutional neural network (1D-CNN) encoder that extracts temporal features
from raw ground-acceleration time series, and (ii) a fully connected (FC)
regression head that maps extracted features to inter-story drift ratio (IDR)
predictions for each of the five stories.

**Temporal Encoder (1D-CNN).**  Three convolutional blocks progressively
increase the channel depth while halving the temporal resolution through
strided convolutions.  Each block consists of a `Conv1d` layer, batch
normalization (BN), and the SiLU activation function (Swish with $\beta=1$)
[9].  The encoder terminates with an adaptive average pooling layer that
produces a fixed-length feature vector regardless of input sequence length.

**Regression Head (FC).**  Four hidden layers with decreasing dimensionality
($256 \to 128 \to 64 \to 32$) map the encoded features to the five-story IDR
output.  SiLU activation is applied after each hidden layer.  Dropout ($p=0.05$)
is applied after the first two wider layers to mitigate overfitting.

**Weight Initialization.**  All convolutional and linear layers use Kaiming
(He) normal initialization [14], which accounts for the nonlinear activation and
prevents gradient vanishing in early training epochs.  Batch normalization
parameters are initialized to unity (weight) and zero (bias).

The complete layer-by-layer specification is presented in Table 1.

**Table 1.** Hybrid-PINN architecture specification.  $B$ denotes the batch
dimension; $k$, $s$, $p$ denote kernel size, stride, and padding, respectively.

| # | Layer | Configuration | Output Shape |
|---|-------|---------------|--------------|
| — | *Input* | Ground acceleration time series | $(B, 1, 2048)$ |
| 1 | Conv1d + BN + SiLU | $1 \to 32$ channels, $k{=}7$, $s{=}2$, $p{=}3$ | $(B, 32, 1024)$ |
| 2 | Conv1d + BN + SiLU | $32 \to 64$ channels, $k{=}5$, $s{=}2$, $p{=}2$ | $(B, 64, 512)$ |
| 3 | Conv1d + BN + SiLU | $64 \to 128$ channels, $k{=}3$, $s{=}2$, $p{=}1$ | $(B, 128, 256)$ |
| 4 | AdaptiveAvgPool1d | Output length = 16 | $(B, 128, 16)$ |
| — | *Flatten* | $128 \times 16 = 2048$ | $(B, 2048)$ |
| 5 | Linear + SiLU + Dropout(0.05) | $2048 \to 256$ | $(B, 256)$ |
| 6 | Linear + SiLU + Dropout(0.05) | $256 \to 128$ | $(B, 128)$ |
| 7 | Linear + SiLU | $128 \to 64$ | $(B, 64)$ |
| 8 | Linear + SiLU | $64 \to 32$ | $(B, 32)$ |
| 9 | Linear (output) | $32 \to 5$ | $(B, 5)$ |
| — | **Total trainable parameters** | | **603,653** |

The choice of SiLU (Swish) over ReLU is deliberate: as a $C^1$-continuous
function, SiLU produces well-defined higher-order gradients required by the
physics-loss term, where the equation of motion involves second-order
derivatives ($\ddot{u}$) [9].

### 3.4.2 Hybrid Loss Function

The total loss function embeds the equation of motion as a physics constraint,
following the PINN paradigm [3]:

$$\mathcal{L}_{total} = \lambda_d \, \mathcal{L}_{data} + \lambda_p \, \mathcal{L}_{physics} + \lambda_b \, \mathcal{L}_{bc} \tag{1}$$

Each component is defined below.

**Data fidelity loss ($\mathcal{L}_{data}$).**  The mean squared error (MSE)
between the network-predicted IDR ($\hat{\theta}_i$) and the OpenSeesPy-simulated
IDR ($\theta_i$) across all stories and samples:

$$\mathcal{L}_{data} = \frac{1}{N \cdot n_s} \sum_{j=1}^{N} \sum_{i=1}^{n_s} \left( \hat{\theta}_{i}^{(j)} - \theta_{i}^{(j)} \right)^2 \tag{2}$$

where $N$ is the number of samples and $n_s = 5$ is the number of stories.

**Physics loss ($\mathcal{L}_{physics}$).**  The $L_2$-norm residual of the
multi-degree-of-freedom equation of motion:

$$\mathcal{L}_{physics} = \frac{1}{N \cdot n_s \cdot T} \sum \left\| \mathbf{M}\ddot{\mathbf{u}} + \mathbf{C}\dot{\mathbf{u}} + \mathbf{f}_{int}(\mathbf{u}, \dot{\mathbf{u}}) + \mathbf{M}\boldsymbol{\iota}\ddot{u}_g \right\|^2 \tag{3}$$

where $\mathbf{M}$ and $\mathbf{C}$ are the lumped mass and Rayleigh damping
matrices (constant throughout the analysis), $\boldsymbol{\iota}$ is the
rigid-diaphragm influence vector, $\ddot{u}_g$ is the ground acceleration, and
$T$ represents the number of time steps.

**Critical distinction: nonlinear restoring force.**  The term
$\mathbf{f}_{int}(\mathbf{u}, \dot{\mathbf{u}})$ represents the *nonlinear*
restoring force vector recorded by OpenSeesPy at every time step during the
NLTHA.  Unlike a linear stiffness formulation ($\mathbf{K}\mathbf{u}$), this
quantity implicitly encodes the full hysteretic behavior of the fiber-section
elements: concrete cracking and crushing (Concrete02, Mander et al. [7]),
steel yielding and strain hardening (Steel02, Menegotto-Pinto [8]), and cyclic
degradation under repeated loading.  The use of $\mathbf{f}_{int}$ rather than
$\mathbf{K}\mathbf{u}$ is essential for capturing the nonlinear response of RC
structures and constitutes the **"Hybrid"** nature of the proposed Digital Twin —
the neural network learns from data while being constrained by the true
nonlinear physics of the finite-element model.

**Boundary/initial condition loss ($\mathcal{L}_{bc}$).**  Enforces that the
building starts from rest:

$$\mathcal{L}_{bc} = \frac{1}{N \cdot n_s} \sum_{j=1}^{N} \sum_{i=1}^{n_s} \left[ u_i^{(j)}(0)^2 + \dot{u}_i^{(j)}(0)^2 \right] \tag{4}$$

**Loss weights.**  The default weights ($\lambda_d = 1.0$, $\lambda_p = 0.1$,
$\lambda_b = 0.01$) were selected to balance the relative magnitudes of each
component.  Additionally, an *adaptive weight scheduling* mechanism based on
gradient-norm balancing [15] is available: at each epoch, the weights $\lambda_p$
and $\lambda_b$ are adjusted via exponential moving average (EMA, $\alpha = 0.9$)
so that the gradient contributions from all three loss components remain
comparable in magnitude, preventing any single term from dominating the
optimization landscape.

### 3.4.3 Training Protocol

All training hyperparameters are summarized in Table 2 and are fixed throughout
this study to ensure full reproducibility per HRPUB requirements.

**Table 2.** Training hyperparameters for the Hybrid-PINN model.

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Optimizer | AdamW [11] | Decoupled weight decay; superior generalization |
| Learning rate ($\eta$) | $1 \times 10^{-3}$ | Standard for Adam-family optimizers |
| Weight decay | $1 \times 10^{-4}$ | L2 regularization for generalization |
| LR scheduler | CosineAnnealingWarmRestarts [12] | Avoids premature convergence to local minima |
| $T_0$ (initial period) | 50 epochs | First cosine cycle length |
| $T_{mult}$ | 2 | Doubling period after each restart |
| $\eta_{min}$ | $1 \times 10^{-6}$ | Minimum learning rate floor |
| Maximum epochs | 500 | Upper bound (early stopping typically triggers earlier) |
| Batch size | 64 | Balances gradient estimation and GPU memory |
| Gradient clipping | Max norm = 1.0 | Prevents gradient explosion in physics loss |
| Early stopping patience | 50 epochs | Monitors validation loss; prevents overfitting |
| Random seed | 42 | Ensures deterministic initialization |
| Data split | 70% / 15% / 15% | Train / Validation / Test (stratified) |

**Training procedure.**  The model is trained on 200+ ground-motion records
generated by the Data Factory (Section 3.2) and processed through the NLTHA
pipeline.  Each record produces a pair of (acceleration time series,
per-story IDR) used as training input and target, respectively.  The physics
loss additionally receives the structural matrices ($\mathbf{M}$, $\mathbf{C}$)
and OpenSeesPy-recorded kinematics ($\ddot{\mathbf{u}}$, $\dot{\mathbf{u}}$,
$\mathbf{f}_{int}$) at every time step.  The dataset of 200+ records ensures
statistical robustness by capturing the aleatory variability inherent in seismic
ground motions across magnitudes ($M_w$ 6.0–7.5), distances ($R_{jb}$ 10–50 km),
and site conditions ($V_{s30}$ 180–760 m/s, NEHRP classes C–D).

**Early stopping** monitors the validation loss (computed on 15% of the dataset)
and halts training when no improvement exceeding $\delta = 10^{-6}$ is observed
for 50 consecutive epochs.  The best model checkpoint (lowest validation loss)
is saved to `data/models/pinn_best.pt` and used for all subsequent evaluation
and benchmarking.

The three training modes supported by the framework are:
1. **Data-only**: $\mathcal{L} = \mathcal{L}_{data}$ (pure supervised baseline).
2. **Hybrid**: $\mathcal{L} = \lambda_d \mathcal{L}_{data} + \lambda_p \mathcal{L}_{physics} + \lambda_b \mathcal{L}_{bc}$ (default).
3. **Adaptive**: Same as Hybrid with self-adaptive weight balancing [15].

## 3.5 Benchmarking Protocol

Real-time applicability requires inference latency ≤ 100 ms. The benchmarking
script (`src/pinn/benchmark_latency.py`) measures:

1. **Cold start**: First inference after model load
2. **Warm inference**: Mean of 1000 consecutive predictions
3. **Batch throughput**: Predictions per second at batch sizes 1, 8, 32, 128
4. **Hardware**: CPU-only (deployment target) and GPU (training)
