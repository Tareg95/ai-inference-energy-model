# GPU Cluster Energy Simulation for AI Inference

> Simulation model for AI inference workloads — modeling request arrivals, GPU power states, provisioning strategies, energy use, and SLA risk.
> KTH Bachelor's Thesis · Electronics and Computer Engineering

---

## Overview

This project simulates the energy consumption of a GPU cluster serving AI inference workloads over a 24-hour period.

The model compares how different provisioning strategies affect:

* total daily energy consumption,
* useful active work,
* idle and overhead energy,
* GPU utilization,
* SLA violation risk during demand spikes.

The simulation is built around a reusable Python implementation in `src/` and an analysis notebook in `notebooks/`.

---

## Model Architecture

```text
Workload Generation
(NHPP, NHPP + Pareto bursts, or Gamma renewal process)
        ↓
Provisioning Strategy
(Static / Conservative / Moderate / Tight / Aggressive)
        ↓
Cold-Start Lag and Capacity Model
        ↓
Utilization Analysis
        ↓
GPU State Distribution
(Active / Execution-idle / Deep-idle)
        ↓
Energy Accounting
(kWh per simulation step and daily totals)
        ↓
Strategy Comparison and Headroom Sweep
```

---

## Repository Structure

```text
ai-inference-energy-model/
│
├── src/
│   ├── config.py          # Shared parameters and configuration
│   └── Simulation.py      # Core reusable simulation functions
│
├── notebooks/
│   └── ai_inference_energy_model.ipynb
│
├── figures/              # Generated figures
├── requirements.txt
└── README.md
```

---

## Simulation Blocks

The notebook is organized into sequential blocks:

| Block       | Description                                                         |
| ----------- | ------------------------------------------------------------------- |
| **Block 0** | Imports, reproducibility seed, system constants, and configuration  |
| **Block 1** | Workload generation using diurnal demand and burst/renewal arrivals |
| **Block 2** | Capacity, utilization, and cold-start lag model                     |
| **Block 3** | GPU state distribution: active, execution-idle, and deep-idle       |
| **Block 4** | Energy accounting and daily energy breakdown                        |
| **Block 5** | Strategy × arrival-process comparison table                         |
| **Block 6** | Headroom sweep showing energy savings versus SLA violations         |

---

## Key Parameters

### Cluster

| Parameter      | Value     | Description                              |
| -------------- | --------- | ---------------------------------------- |
| `total_gpus`   | 256       | Physical GPU cluster size                |
| `mu`           | 0.2 req/s | Service rate per GPU                     |
| `pod_size`     | 8 GPUs    | Scaling granularity                      |
| `c_peak`       | 200 GPUs  | Peak reference capacity                  |
| `step_minutes` | 3 min     | Simulation time resolution               |
| `num_steps`    | 480       | Number of simulation steps over 24 hours |

The peak request rate is derived from the conservative target utilization:

```python
lambda_peak = target_rho_conservative * mu * c_peak
```

With the default values:

```python
lambda_peak = 0.78 * 0.2 * 200 = 31.2 req/s
```

---

## Power Model

The simulation uses three GPU power states.

| State              | Power |
| ------------------ | ----- |
| Deep-idle GPU      | 140 W |
| Execution-idle GPU | 220 W |
| Active GPU average | 450 W |

The model also includes an execution-idle fraction:

```python
execution_idle_fraction = 0.197
```

This represents the part of provisioned GPU time that remains in an execution-idle state even during active serving periods.

---

## Arrival Processes

The model supports three arrival-process variants in the simulation code:

### 1. NHPP only

A smooth Non-Homogeneous Poisson Process shaped by a 24-hour diurnal demand curve.

### 2. Hybrid

An NHPP base workload combined with Pareto-distributed burst events.

Default burst parameters include:

```python
pareto_shape = 2.0
burst_rate_per_hour = 10.0
burst_window_hours = 0.01
size_multiplier_x_lambda_peak = 0.5
```

This mode is used as a stress-test workload with bursty request spikes.

### 3. Gamma renewal

A non-homogeneous Gamma renewal process used to model more variable inter-arrival behavior than a standard Poisson process.

The default Gamma renewal shape is:

```python
gamma_renewal_shape = 0.5
```

This gives a coefficient of variation above 1, producing more burst-like arrival spacing than a standard NHPP.

---

## Provisioning Strategies

The simulation compares five provisioning strategies:

| Strategy     | Target utilization | Description                                     |
| ------------ | ------------------ | ----------------------------------------------- |
| Static       | N/A                | Keeps all 256 GPUs provisioned                  |
| Conservative | 0.78               | Leaves larger headroom to reduce SLA risk       |
| Moderate     | 0.85               | Balances energy savings and risk                |
| Tight        | 0.90               | Reduces idle capacity further                   |
| Aggressive   | 0.95               | Maximizes energy savings but increases SLA risk |

Provisioned capacity is rounded up to the nearest pod size of 8 GPUs.

A one-step cold-start lag is applied when scaling up. With the default 3-minute resolution, this corresponds to a 3-minute scale-up delay. Scale-down is treated as immediate.

---

## GPU State Model

At every simulation step, GPUs are divided into three categories:

### Active inference

GPUs actively serving inference requests.

### Execution-idle

Provisioned GPUs that are not doing useful inference work but still keep the model loaded and consume execution-idle power.

### Deep-idle

Unprovisioned GPUs outside the active serving pool.

The energy model separates daily energy into:

```text
Total energy =
    Pure active work
  + Active task overhead
  + Inactive idle energy
  + Deep-idle energy
```

---

## Output Metrics

The main outputs are:

| Metric         | Meaning                                                            |
| -------------- | ------------------------------------------------------------------ |
| `total_kwh`    | Total daily energy consumption                                     |
| `work_kwh`     | Useful active inference energy                                     |
| `overhead_kwh` | Active execution-idle overhead                                     |
| `idle_kwh`     | Energy from provisioned but inactive GPUs                          |
| `deep_kwh`     | Energy from deep-idle GPUs                                         |
| `mean_rho`     | Mean effective utilization                                         |
| `max_rho`      | Maximum effective utilization                                      |
| `sla_viol`     | Number of 3-minute intervals where effective utilization exceeds 1 |

SLA violations are counted as simulation steps where:

```python
rho_effective > 1.0
```

With the default 3-minute resolution, one day contains 480 possible violation intervals.

---

## Requirements

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

The current requirements are:

```text
numpy
matplotlib
pandas
jupyter
ipywidgets
```

Python 3.8+ is recommended.

---

## Usage

1. Clone the repository.
2. Install the requirements.
3. Open the notebook:

```text
notebooks/ai_inference_energy_model.ipynb
```

4. Run the notebook from top to bottom.

The notebook imports shared configuration from `src/config.py` and simulation functions from `src/Simulation.py`.

To change the experiment, edit the configuration values in `src/config.py`, such as:

```python
total_gpus
mu
pod_size
step_minutes
target_rho_conservative
target_rho_moderate
target_rho_tight
target_rho_aggressive
```

---

## Reproducing the Thesis Results

To reproduce the main comparison table, run the notebook without changing the default parameters.

The main strategy comparison is produced in Block 5. It compares the provisioning strategies across the selected arrival-process models and reports mean and standard deviation over multiple random seeds.

Block 6 performs a headroom sweep to show the trade-off between reduced energy use and increased SLA risk.

---

## Project Context

This simulation was developed as part of a Bachelor's thesis at KTH Royal Institute of Technology.

The thesis investigates how workload characteristics and provisioning decisions affect the energy consumption of GPU clusters used for AI inference.

The model draws on:

* queueing-inspired utilization analysis,
* GPU power-state modeling,
* bursty AI inference workload behavior,
* autoscaling and provisioning trade-offs,
* energy versus service-quality analysis.

---

## Important Note

This repository currently focuses on GPU-side IT energy modeling.

It does not currently implement full datacenter carbon accounting, live grid-intensity fetching, or full facility-level energy modeling with PUE. Those can be added later as extensions, but they are not part of the current core implementation.

---

## License

This project is part of an academic thesis. Please cite appropriately if you build on this work.
