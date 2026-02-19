# 3. Methods

<!-- HRPUB Section: Methods -->
<!-- CRITICAL: HRPUB requires complete methodology description including
     data processing, statistical tests, and computational procedures. -->

## 3.1 Parametric FEM Reference Model

The proposed framework is built upon a high-fidelity nonlinear finite element model (FEM) developed in OpenSeesPy. Unlike traditional surrogate models, this system is fully parametric, allowing the generation of training data for $N$-story reinforced concrete (RC) frames by varying geometric and material properties. For the final validation, a 10-story frame ($N=10$) is utilized to assess the model's capability to capture higher-mode effects under real seismic excitations from the PEER NGA-West2 database.

The reference model specifically employs:
1.  **Fiber Sections**: Distributed plasticity with `Concrete02` (Mander confinement) and `Steel02` (Giuffré-Menegotto-Pinto).
2.  **Element Formulation**: Displacement-based beam-column elements with P-Delta geometric nonlinearity.
3.  **Variable Geometry**: Column cross-sections ($700^2 \to 500^2$ mm) and reinforcement ratios taper along the height to simulate realistic design practices.

## 3.2 Ground Motion Selection (PEER NGA-West2)

Ground motion records were obtained from the PEER NGA-West2 database [2], the
most comprehensive publicly available collection of processed strong-motion
recordings from shallow crustal earthquakes in active tectonic regimes.

### 3.2.1 Selection Criteria

Records were selected using the following engineering criteria, consistent with
ASCE 7-22 §16.2 requirements for nonlinear response-history analysis:

| Parameter | Range | Rationale |
|-----------|-------|-----------|
| Moment magnitude ($M_w$) | 6.0 – 7.5 | Destructive events governing code-level design |
| Joyner-Boore distance ($R_{jb}$) | 10 – 50 km | Near-to-moderate field; avoids near-fault directivity and far-field attenuation artifacts |
| Shear-wave velocity ($V_{s30}$) | 180 – 760 m/s | NEHRP site classes C and D (stiff soil to soft rock) |
| Fault mechanism | All types | Captures variability across strike-slip, reverse, and normal faulting |

### 3.2.2 Downloaded Dataset

The search yielded **100 unique seismic events** (identified by Record Sequence
Number, RSN), providing **299 three-component time histories** (two horizontal
and one vertical per station).  Only the horizontal components are used for
the 2D frame analysis, yielding approximately **200 input time series**.  All
records were downloaded in unscaled AT2 format (acceleration in units of *g*),
and subsequently scaled to match the site-specific ASCE 7-22 design spectrum
using the automated scaling procedure implemented in `data_factory.py`, with
a maximum allowable scale factor of 5.0.

### 3.2.3 Data Augmentation Strategy

To expand the effective training set beyond the 100 base events, three
augmentation techniques are applied by the preprocessing pipeline
(`src/preprocessing/pipeline.py`):

1. **Temporal windowing**: overlapping sub-windows of the strong-motion
   duration, increasing temporal diversity.
2. **Amplitude scaling**: random scaling within ±20% of the original PGA,
   simulating intensity variability.
3. **Gaussian noise injection**: additive noise ($\sigma = 0.01g$) to improve
   model robustness against measurement uncertainty.

These techniques transform the 100 base records into approximately
**1,000–1,500 training samples**, providing sufficient statistical robustness
for the PINN while capturing the aleatory variability inherent in seismic
ground motions across magnitudes, distances, and site conditions.

## 3.3 Seq2Seq PINN Architecture (v2.0)

To overcome the limitations of scalar regression (v1.0) in predicting upper-story responses, we transition to a **Sequence-to-Sequence (Seq2Seq)** architecture. The model consists of:

1.  **Temporal Encoder**: A 1D-CNN backbone coupled with Multi-Head Self-Attention layers to extract deep temporal features from the input ground acceleration $\ddot{u}_g(t)$.
2.  **History Decoder**: A dense output head that reconstructs the full displacement time-history $\mathbf{u}(t) \in \mathbb{R}^{N \times T}$ for all stories simultaneously.

By predicting the complete time-series, the network is forced to maintain temporal phase consistency, which is critical for resolving high-frequency oscillations in upper levels (e.g., Story 3 and above).

**Table 1.** Hybrid-PINN v2.0 architecture specification. $B$: Batch size, $N$: Stories (5).

| Layer | Configuration | Output Shape |
|-------|---------------|--------------|
| *Input* | Ground acceleration $\ddot{u}_g(t)$ | $(B, 1, 2048)$ |
| Conv Block 1 | $1 \to 32$ channels, $k{=}7$, $s{=}2$ | $(B, 32, 1024)$ |
| Conv Block 2 | $32 \to 64$ channels, $k{=}5$, $s{=}2$ | $(B, 64, 512)$ |
| Conv Block 3 | $64 \to 128$ channels, $k{=}3$, $s{=}2$ | $(B, 128, 256)$ |
| Upsample | Interpolate to original length $T$ | $(B, 128, 2048)$ |
| Projection | Linear $128 \to 32$ | $(B, 32, 2048)$ |
| **Output Head** | **Linear $32 \to N$ (applied per step)** | **$(B, 5, 2048)$** |

## 3.4 Kinematic-Informed Regularization

The "Hybrid" nature of the Digital Twin arises from a custom loss function that enforces the Equation of Motion (EoM) during training. We adopt a **Teacher Forcing** strategy for the internal forces $\mathbf{f}_{int}$, regularizing the predicted kinematics:

$$\mathcal{L}_{reg} = \| \mathbf{M}\ddot{\mathbf{u}}_{pred} + \mathbf{C}\dot{\mathbf{u}}_{pred} + \mathbf{f}_{int}(\mathbf{u}_{true}) + \mathbf{M}\boldsymbol{\iota}\ddot{u}_g \|^2$$

Where $\ddot{\mathbf{u}}_{pred}$ and $\dot{\mathbf{u}}_{pred}$ are computed via **differentiable finite differences** within the PyTorch computational graph. This formulation acts as a physics-based low-pass filter, penalizing non-physical smoothing and ensuring the Digital Twin remains a faithful emulator of the FEM's dynamic behavior.

### 3.4.1 Loss Components
The total loss $\mathcal{L}_{total}$ integrates data fidelity, physics regularization, and boundary conditions:

$$\mathcal{L}_{total} = \lambda_d \mathcal{L}_{data} + \lambda_p \mathcal{L}_{reg} + \lambda_b \mathcal{L}_{bc}$$

1.  **Data Loss ($\mathcal{L}_{data}$)**: MSE of displacement histories $\mathbf{u}(t)$.
2.  **Physics Loss ($\mathcal{L}_{reg}$)**: The kinematic residual defined above.
3.  **Boundary Loss ($\mathcal{L}_{bc}$)**: Enforcing zero initial displacement and velocity.

## 3.5 Training Protocol

All training hyperparameters are summarized in Table 2 and are fixed throughout
this study to ensure full reproducibility per HRPUB requirements.

**Table 2.** Training hyperparameters for the Hybrid-PINN model.

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Optimizer | AdamW [11] | Decoupled weight decay; superior generalization |
| Learning rate ($\eta$) | $1 \times 10^{-3}$ | Standard for Adam-family optimizers |
| Weight decay | $1 \times 10^{-4}$ | L2 regularization for generalization |
| Scheduler | CosineAnnealingWarmRestarts | Avoids premature convergence |
| Max epochs | 500 | Upper bound (early stopping) |
| Batch size | 64 | Balances gradient variance and memory |
| Physics Weight $\lambda_p$ | 0.1 | Tuned to balance MSE and EoM residual |

## 3.6 Benchmarking Protocol

Real-time applicability requires inference latency ≤ 100 ms. The benchmarking
script (`src/pinn/benchmark_latency.py`) measures:

1. **Cold start**: First inference after model load
2. **Warm inference**: Mean of 1000 consecutive predictions
3. **Batch throughput**: Predictions per second at batch sizes 1, 8, 32, 128
4. **Hardware**: CPU-only (deployment target) and GPU (training)
