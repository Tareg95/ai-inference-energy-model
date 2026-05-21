"""Generate every figure the notebook produces, for every strategy / workload combo.

Sweeps (arrival_mode, burst_distribution) workload combos and (for the
per-strategy plots) every strategy in ``STRATEGIES``. Each image is written
to ``figures/all_runs/`` with a unique name encoding the combo, e.g.

    capacity_utilisation_cold_start__hybrid_pareto__Conservative.png
    workload_pattern__gamma_renewal.png
    comparison_summary__nhpp_only.png

Run from the repo root:  python -m src.generate_all_figures
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import (
    ArrivalParams,
    BurstParams,
    ClusterParams,
    DIURNAL_SHAPE,
    PowerParams,
    STRATEGIES,
)
from src.Simulation import (
    apply_cold_start_lag,
    bin_to_15min_rate,
    build_lambda_15min,
    energy_arrays,
    gpu_state_arrays,
    provisioning_arrays,
    scale_in_pods,
    simulate_bursts,
    simulate_gamma_renewal_nhpp,
    simulate_nhpp,
    utilisation_arrays,
)

OUT_DIR = _REPO_ROOT / "figures" / "all_runs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 67

STRATEGY_COLORS = {
    "Static":       "tab:blue",
    "Conservative": "tab:green",
    "Moderate":     "tab:olive",
    "Tight":        "tab:orange",
    "Aggressive":   "tab:red",
}

# (arrival_mode, burst_distribution_or_None) -> short tag for filenames
WORKLOADS: list[tuple[str, str | None, str]] = [
    ("nhpp_only",     None,     "nhpp_only"),
    ("hybrid",        "pareto", "hybrid_pareto"),
    ("hybrid",        "gamma",  "hybrid_gamma"),
    ("gamma_renewal", None,     "gamma_renewal"),
]


def workload_label(arrival_mode: str, burst_dist: str | None,
                   arrival_params: ArrivalParams) -> str:
    if arrival_mode == "nhpp_only":
        return "NHPP only [M/M/c]"
    if arrival_mode == "hybrid":
        return f"NHPP + {burst_dist.capitalize()} bursts [M/M/c + {burst_dist} overlay]"
    if arrival_mode == "gamma_renewal":
        cv = 1.0 / np.sqrt(arrival_params.gamma_renewal_shape)
        return (f"Gamma renewal (shape={arrival_params.gamma_renewal_shape}, "
                f"CV~={cv:.2f}) [G/M/c]")
    return arrival_mode


def build_workload(arrival_mode: str, burst_dist: str | None,
                   cluster: ClusterParams, arrival_params: ArrivalParams,
                   lambda_hourly: np.ndarray, lambda_15min_nhpp: np.ndarray,
                   ) -> tuple[np.ndarray, BurstParams, list[float], list[float], list[int]]:
    """Return (lambda_arr at step resolution, burst_params used, base NHPP arrivals,
    burst_times, burst_sizes) for one workload combo, reproducibly seeded."""

    np.random.seed(SEED)
    burst_params = BurstParams(distribution=burst_dist) if burst_dist else BurstParams()

    def lambda_func(t: float) -> float:
        return float(np.interp(t % 24, np.arange(24), lambda_hourly))

    base_arrivals = simulate_nhpp(lambda_func, float(lambda_hourly.max()),
                                  cluster.T_day_hours)

    burst_times: list[float] = []
    burst_sizes: list[int] = []

    if arrival_mode == "nhpp_only":
        active_arrivals = list(base_arrivals)
    elif arrival_mode == "hybrid":
        bursts, burst_times, burst_sizes = simulate_bursts(
            burst_params, T=cluster.T_day_hours, lambda_peak=cluster.lambda_peak,
        )
        active_arrivals = sorted(base_arrivals + bursts)
    elif arrival_mode == "gamma_renewal":
        active_arrivals = simulate_gamma_renewal_nhpp(
            arrival_params.gamma_renewal_shape,
            arrival_params.gamma_renewal_scale,
            lambda_func, cluster.T_day_hours,
        )
    else:
        raise ValueError(f"Unknown arrival_mode {arrival_mode!r}")

    lambda_arr = bin_to_15min_rate(
        active_arrivals,
        num_steps=cluster.num_steps,
        step_hours=cluster.step_hours,
    )
    return lambda_arr, burst_params, base_arrivals, burst_times, burst_sizes


def save_workload_pattern(tag: str, label: str, arrival_mode: str,
                          burst_dist: str | None, burst_params: BurstParams,
                          base_arrivals: list[float], active_arrivals_count: int,
                          burst_times: list[float], burst_sizes: list[int],
                          lambda_arr: np.ndarray,
                          cluster: ClusterParams,
                          lambda_hourly: np.ndarray) -> None:
    hours = np.arange(24)
    # Reconstruct active arrivals timestamps approximately via bins is lossy,
    # so for the per-hour count we recompute from a fresh draw using the same
    # seed by running the build twice would be wasteful; instead summarise via
    # lambda_arr's per-bin rates. For plotting fidelity match the notebook by
    # counting base + bursts when those are available.
    requests_nhpp = [sum(1 for t in base_arrivals if h <= t < h + 1) for h in hours]

    # Approximate active counts from the binned lambda: requests per hour
    # = sum over bins of lambda * step_seconds.
    step_seconds = cluster.step_hours * 3600.0
    requests_per_bin = lambda_arr * step_seconds
    bins_per_hour = int(round(1.0 / cluster.step_hours))
    requests_active = requests_per_bin.reshape(-1, bins_per_hour).sum(axis=1)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(hours, requests_nhpp, label="NHPP base reference", alpha=0.7)
    ax.plot(hours, requests_active, label=label, color="red", linewidth=1.5)

    if arrival_mode == "hybrid" and burst_sizes:
        max_bs = max(burst_sizes)
        for bt, bs in zip(burst_times, burst_sizes):
            ax.axvline(x=bt, color="orange", alpha=0.6,
                       linewidth=0.5 + (bs / max_bs) * 3)
        ax.axvline(x=-1, color="orange", alpha=0.6, linewidth=1.5,
                   label=f"{burst_dist.capitalize()} burst event")

    ax.set_title(f"Workload pattern: {label}")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Requests per hour")
    ax.set_xlim(0, 24)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"workload_pattern__{tag}.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_per_strategy_figures(workload_tag: str, workload_label_str: str,
                              strategy: str, lambda_arr: np.ndarray,
                              cluster: ClusterParams, power: PowerParams,
                              hours_axis: np.ndarray) -> None:
    prov = provisioning_arrays(lambda_arr, cluster.mu, cluster.total_gpus,
                               cluster.pod_size)
    c_active = prov[strategy]
    c_ready = apply_cold_start_lag(c_active)
    util = utilisation_arrays(lambda_arr, c_active, c_ready,
                              cluster.mu, cluster.total_gpus)
    states = gpu_state_arrays(c_ready, util["rho_effective"],
                              cluster.total_gpus, power.execution_idle_fraction)
    energy = energy_arrays(c_ready, util["rho_effective"],
                           states["gpus_active"], states["gpus_deep_idle"],
                           power, step_hours=cluster.step_hours)

    # --- Capacity / utilisation / cold-start ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax1.set_ylabel("System throughput (queries/sec)", fontsize=11, fontweight="bold")
    ax1.step(hours_axis, util["capacity_target_qps"], where="post",
             color="tab:blue", linewidth=2, alpha=0.5,
             label=f"Target capacity ({strategy})")
    ax1.step(hours_axis, util["capacity_ready_qps"], where="post",
             color="darkblue", linewidth=2.5, label="Ready capacity (post-lag)")
    ax1.plot(hours_axis, util["demand_qps"], color="tab:red",
             linestyle="-", linewidth=2.5,
             label=f"User demand λ(t)  [{workload_label_str}]")
    ax1.set_ylim(0, max(util["capacity_target_qps"].max(),
                        util["demand_qps"].max()) + 10)
    ax1.set_title(f"Hardware scaling vs demand: {strategy}, {workload_label_str}",
                  fontsize=13)
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.set_xlabel("Hour of day", fontsize=12)
    ax2.set_ylabel("Utilisation (ρ)", fontsize=11, fontweight="bold")
    ax2.plot(hours_axis, util["global_rho"], color="grey", linestyle=":",
             linewidth=1.5, label="Global cluster utilisation")
    ax2.plot(hours_axis, util["rho_theoretical"], color="lightgreen",
             marker=".", linestyle="-", linewidth=1.5,
             label="Theoretical utilisation")
    ax2.plot(hours_axis, util["rho_effective"], color="darkgreen",
             marker="x", markersize=4, linestyle="-", linewidth=2,
             label="Effective utilisation (with lag)")
    ax2.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5,
                label="SLA threshold (ρ > 1.0)")
    ax2.set_ylim(0, max(1.3, util["rho_effective"].max() * 1.05))
    ax2.set_xticks(np.arange(0, 25, 2))
    ax2.legend(loc="upper left", framealpha=0.9)
    ax2.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"capacity_utilisation_cold_start__{workload_tag}__{strategy}.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- GPU state distribution ---
    fig = plt.figure(figsize=(14, 6))
    plt.bar(hours_axis, states["active_ratio"],
            width=cluster.step_hours, label="Active inference",
            color="#2ca02c", align="edge")
    plt.bar(hours_axis, states["execution_idle_ratio"],
            bottom=states["active_ratio"], width=cluster.step_hours,
            label="Execution-idle (model loaded)", color="#ff7f0e", align="edge")
    plt.bar(hours_axis, states["deep_idle_ratio"],
            bottom=states["active_ratio"] + states["execution_idle_ratio"],
            width=cluster.step_hours,
            label="Deep-idle (HBM flushed)", color="#7f7f7f", align="edge")
    plt.title(f"Cluster state distribution: {strategy}, {workload_label_str}")
    plt.xlabel("Hour of day")
    plt.ylabel(f"Proportion of total cluster ({cluster.total_gpus} GPUs)")
    plt.xticks(np.arange(0, 25, 2))
    plt.ylim(0, 1.05)
    plt.legend(loc="lower right", framealpha=0.9)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    fig.savefig(OUT_DIR / f"gpu_state_distribution__{workload_tag}__{strategy}.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Energy breakdown by state ---
    fig = plt.figure(figsize=(12, 6))
    plt.stackplot(hours_axis,
                  energy["energy_pure_work"],
                  energy["energy_active_overhead"],
                  energy["energy_inactive_idle"],
                  energy["energy_deep_idle"],
                  labels=["Pure active work", "Active task overhead",
                          "Inactive idle (exec-idle)", "Deep idle"],
                  colors=["#2ca02c", "#ff7f0e", "#d62728", "#7f7f7f"],
                  alpha=0.7)
    plt.title(f"Energy breakdown (kWh per {cluster.step_minutes:.0f}-min step): "
              f"{strategy}, {workload_label_str}")
    plt.xlabel("Hour of day")
    plt.ylabel("Energy per step (kWh)")
    plt.legend(loc="upper right")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    fig.savefig(OUT_DIR / f"energy_breakdown_by_state__{workload_tag}__{strategy}.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_headroom_sweep(workload_tag: str, workload_label_str: str,
                        lambda_arr: np.ndarray,
                        cluster: ClusterParams) -> None:
    target_rhos = np.linspace(0.75, 1.00, 26)
    overhead_pct = (1.0 - target_rhos) * 100.0
    violations = np.zeros_like(target_rhos, dtype=int)
    for i, rho_star in enumerate(target_rhos):
        c_target = np.minimum(
            scale_in_pods(lambda_arr / (rho_star * cluster.mu), cluster.pod_size),
            cluster.total_gpus,
        )
        c_ready_i = apply_cold_start_lag(c_target)
        util_i = utilisation_arrays(lambda_arr, c_target, c_ready_i,
                                    cluster.mu, cluster.total_gpus)
        violations[i] = int(np.sum(util_i["rho_effective"] > 1.0))

    refs = [
        ("Conservative", cluster.target_rho_conservative, "tab:green"),
        ("Moderate",     cluster.target_rho_moderate,     "tab:olive"),
        ("Tight",        cluster.target_rho_tight,        "tab:orange"),
        ("Aggressive",   cluster.target_rho_aggressive,   "tab:red"),
    ]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(overhead_pct, violations, marker="o", linewidth=2,
            color="tab:red", label="SLA violations")
    for name, rho_star, color in refs:
        pct = (1.0 - rho_star) * 100.0
        ax.axvline(pct, color=color, linestyle="--", linewidth=1.5,
                   label=f"{name} ({pct:.0f}% overhead)")
    ax.invert_xaxis()
    ax.set_xlabel("Provisioning overhead (1 - target rho), %")
    ax.set_ylabel(f"SLA violations (out of {cluster.num_steps} intervals)")
    ax.set_title(f"SLA violations vs overhead - workload: {workload_label_str}")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"headroom_sweep_violations__{workload_tag}.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_strategy_comparison(workload_tag: str, workload_label_str: str,
                             lambda_arr: np.ndarray,
                             cluster: ClusterParams, power: PowerParams,
                             hours_axis: np.ndarray) -> None:
    prov = provisioning_arrays(lambda_arr, cluster.mu, cluster.total_gpus,
                               cluster.pod_size)
    scenarios: list[dict] = []
    for s in STRATEGIES:
        c_active = prov[s]
        c_ready = apply_cold_start_lag(c_active)
        util = utilisation_arrays(lambda_arr, c_active, c_ready,
                                  cluster.mu, cluster.total_gpus)
        states = gpu_state_arrays(c_ready, util["rho_effective"],
                                  cluster.total_gpus, power.execution_idle_fraction)
        energy = energy_arrays(c_ready, util["rho_effective"],
                               states["gpus_active"], states["gpus_deep_idle"],
                               power, step_hours=cluster.step_hours)
        scenarios.append({"strategy": s, "util": util, "states": states,
                          "energy": energy})

    labels = [sc["strategy"] for sc in scenarios]
    colors = [STRATEGY_COLORS[s] for s in labels]

    # --- 2x2 summary bar charts ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    ax = axes[0, 0]
    ax.bar(labels, [sc["energy"]["total_kwh"] for sc in scenarios], color=colors)
    ax.set_title("Total daily energy (kWh)")
    ax.set_ylabel("kWh")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax = axes[0, 1]
    ax.bar(labels,
           [100 * sc["energy"]["work_kwh"] / sc["energy"]["total_kwh"]
            for sc in scenarios], color=colors)
    ax.set_title("Work fraction (% of total energy)")
    ax.set_ylabel("%")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax = axes[1, 0]
    ax.bar(labels, [float(sc["util"]["rho_effective"].max()) for sc in scenarios],
           color=colors)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="SLA threshold")
    ax.set_title("Max effective utilisation rho")
    ax.set_ylabel("rho")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")

    ax = axes[1, 1]
    ax.bar(labels, [int(np.sum(sc["util"]["rho_effective"] > 1.0))
                    for sc in scenarios], color=colors)
    ax.set_title(f"SLA violations (out of {cluster.num_steps} intervals)")
    ax.set_ylabel("count")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.suptitle(f"Strategy comparison - workload: {workload_label_str}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"comparison_summary__{workload_tag}.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Side-by-side rho(t) grid ---
    fig, axes = plt.subplots(1, len(STRATEGIES), figsize=(20, 4.5),
                             sharex=True, sharey=True)
    for c, sc in enumerate(scenarios):
        ax = axes[c]
        ax.plot(hours_axis, sc["util"]["rho_effective"],
                color=STRATEGY_COLORS[sc["strategy"]], linewidth=1.5)
        ax.axhline(1.0, color="red", linestyle="--", linewidth=1)
        ax.set_title(sc["strategy"], fontsize=11, fontweight="bold")
        ax.set_xlabel("Hour of day")
        ax.grid(True, linestyle=":", alpha=0.4)
        if c == 0:
            ax.set_ylabel("rho_eff")
    plt.suptitle(f"Effective utilisation rho(t) per strategy - workload: {workload_label_str}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"comparison_utilisation_grid__{workload_tag}.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Stacked energy components per strategy ---
    work_vals     = np.array([sc["energy"]["work_kwh"]     for sc in scenarios])
    overhead_vals = np.array([sc["energy"]["overhead_kwh"] for sc in scenarios])
    inactive_vals = np.array([sc["energy"]["idle_kwh"]     for sc in scenarios])
    deep_vals     = np.array([sc["energy"]["deep_kwh"]     for sc in scenarios])
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x, work_vals, label="Pure active work", color="#2ca02c")
    ax.bar(x, overhead_vals, bottom=work_vals,
           label="Active task overhead", color="#ff7f0e")
    ax.bar(x, inactive_vals, bottom=work_vals + overhead_vals,
           label="Inactive idle (exec-idle)", color="#d62728")
    ax.bar(x, deep_vals, bottom=work_vals + overhead_vals + inactive_vals,
           label="Deep idle", color="#7f7f7f")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Daily energy (kWh)")
    ax.set_title(f"Energy breakdown by GPU state - workload: {workload_label_str}")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"comparison_energy_breakdown__{workload_tag}.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cluster = ClusterParams()
    power = PowerParams()
    arrival_params = ArrivalParams()

    lambda_hourly = DIURNAL_SHAPE * cluster.lambda_peak
    hours_24 = np.arange(int(cluster.T_day_hours))
    hours_axis = np.arange(0.0, cluster.T_day_hours, cluster.step_hours)
    lambda_15min_nhpp = np.interp(hours_axis, hours_24, lambda_hourly)

    written: list[str] = []
    for arrival_mode, burst_dist, tag in WORKLOADS:
        label = workload_label(arrival_mode, burst_dist, arrival_params)
        print(f"\n=== {tag}: {label} ===")

        lambda_arr, burst_params, base_arrivals, burst_times, burst_sizes = build_workload(
            arrival_mode, burst_dist, cluster, arrival_params,
            lambda_hourly, lambda_15min_nhpp,
        )

        # Workload pattern (one per workload combo)
        save_workload_pattern(tag, label, arrival_mode, burst_dist, burst_params,
                              base_arrivals, len(base_arrivals),
                              burst_times, burst_sizes, lambda_arr,
                              cluster, lambda_hourly)
        written.append(f"workload_pattern__{tag}.png")

        # Per-strategy figures
        for strategy in STRATEGIES:
            save_per_strategy_figures(tag, label, strategy, lambda_arr,
                                      cluster, power, hours_axis)
            written.extend([
                f"capacity_utilisation_cold_start__{tag}__{strategy}.png",
                f"gpu_state_distribution__{tag}__{strategy}.png",
                f"energy_breakdown_by_state__{tag}__{strategy}.png",
            ])
            print(f"  [{strategy}] per-strategy figures saved")

        # Headroom sweep (per workload combo)
        save_headroom_sweep(tag, label, lambda_arr, cluster)
        written.append(f"headroom_sweep_violations__{tag}.png")

        # Block-7 strategy comparison (per workload combo)
        save_strategy_comparison(tag, label, lambda_arr, cluster, power, hours_axis)
        written.extend([
            f"comparison_summary__{tag}.png",
            f"comparison_utilisation_grid__{tag}.png",
            f"comparison_energy_breakdown__{tag}.png",
        ])
        print(f"  workload-level figures saved")

    print(f"\nWrote {len(written)} figures to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
