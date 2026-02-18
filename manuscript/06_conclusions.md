# 6. Conclusions

## 6.1 Key Findings

This study developed and validated a Hybrid Digital Twin framework for seismic response prediction of reinforced concrete buildings, demonstrating its viability with the complete PEER NGA-West2 dataset. The key findings from the final v1.6 model are:

1.  **Physics-Informed Accuracy**: The PINN model achieved $R^2 = 0.783$ on the complete PEER campaign (299 records), with an RMSE of 0.834%. The integration of temporal self-attention and per-story weighted loss improved performance by +1.1% over the baseline CNN architecture.
2.  **Robustness to Real Data**: The model generalized effectively to real-world earthquake records, achieving accuracy comparable to synthetic benchmarks ($R^2 \approx 0.79$). Scaling the dataset from 21 to 289 records was critical for this achievement.
3.  **Real-Time Capability**: With an average total latency of **~1.99 ms** per time step on a standard CPU, the framework operates well within the 10–20 ms threshold required for real-time structural health monitoring and control, even with the added complexity of attention layers.
4.  **Data Efficiency**: The hybrid loss function enabled effective learning from **289 valid records** (5,058 augmented samples), achieving accuracy competitive with methods requiring thousands of simulations.
5.  **Computational Efficiency**: Parallelized NLTHA execution with 10 workers reduced the simulation campaign from ~14 hours to **49.5 minutes** (17× speedup), making large-scale data generation practical.

## 6.2 Contributions

-   **Open-Source Pipeline**: A fully reproducible, end-to-end Python pipeline integrating OpenSeesPy simulation with PyTorch-based PINN training, parameterized for $N$-story buildings.
-   **Architecture v1.6**: A refined Hybrid PINN architecture incorporating temporal self-attention and adaptive physics weighting, optimized for seismic time-series processing.
-   **Real Data Validation**: Validated against the full PEER NGA-West2 database (299 records across Friuli, Imperial Valley, Coalinga, and San Fernando events).
-   **Hybrid Loss Formulation**: Implementation of the weighted multi-degree-of-freedom equation of motion as a differentiable loss function.

## 6.3 Future Work

1.  **Input Feature Engineering**: To address the limitation in upper-story prediction ($R^2 = 0.55$), future iterations should incorporate additional inputs such as ground velocity ($v_g$) or spectral acceleration ($S_a(T)$) to better inform the model of frequency content.
2.  **Graph Neural Networks (GNN)**: Modeling the building topology explicitly as a graph could improve the flow of information between stories compared to the current dense regression head.
3.  **Variable Cross-Sections**: Implement story-dependent column sizing for buildings taller than 8 stories.
4.  **Experimental Validation**: Calibrate the digital twin against shake table test data.
5.  **Edge Deployment**: Deploy the quantized model on embedded hardware (Jetson Nano, Raspberry Pi) for decentralized SHM.
