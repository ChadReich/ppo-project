import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

scratch_results_file = "results/halfcheetah_scratch.csv"
sb3_results_file = "results/halfcheetah_sb3.csv"

scratch_diagnostics_file = "results/halfcheetah_scratch_diagnostics.csv"
sb3_diagnostics_file = "results/halfcheetah_sb3_diagnostics.csv"

# Load evaluation results from CSV files
def load_results(
    results_file
):
    """
    Load evaluation results from a CSV file.
    """

    return pd.read_csv(
        results_file
    )

# Load results for both implementations
scratch_results = load_results(
    scratch_results_file
)

sb3_results = load_results(
    sb3_results_file
)       

# Load diagnostics from CSV files
def load_diagnostics(
    diagnostics_file
):
    """
    Load PPO diagnostic results from a CSV file.
    """

    return pd.read_csv(
        diagnostics_file
    )

# Load diagnostics for both implementations
scratch_diagnostics = load_diagnostics(
    scratch_diagnostics_file
)

sb3_diagnostics = load_diagnostics(
    sb3_diagnostics_file
)

print("Scratch evaluation rows:", len(scratch_results))
print("SB3 evaluation rows:", len(sb3_results))

print("Scratch diagnostic rows:", len(scratch_diagnostics))
print("SB3 diagnostic rows:", len(sb3_diagnostics))

print()
print(
    "Scratch training seeds:",
    sorted(scratch_results["training_seed"].unique())
)

print(
    "SB3 training seeds:",
    sorted(sb3_results["training_seed"].unique())
)

print()
print(
    "Scratch final timestep:",
    scratch_results["timestep"].max()
)

print(
    "SB3 final timestep:",
    sb3_results["timestep"].max()
)

# Evaluation summary

scratch_summary = (
    scratch_results
    .groupby("timestep")["mean_return"]
    .agg(["mean", "std"])
    .reset_index()
)

sb3_summary = (
    sb3_results
    .groupby("timestep")["mean_return"]
    .agg(["mean", "std"])
    .reset_index()
)

print()
print("Scratch evaluation checkpoints:", len(scratch_summary))
print("SB3 evaluation checkpoints:", len(sb3_summary))


# Learning curve

os.makedirs(
    "figures",
    exist_ok=True
)

fig, ax = plt.subplots(
    figsize=(9, 5.5)
)

scratch_colour = "tab:blue"
sb3_colour = "tab:orange"


# Plot each scratch training seed
for seed in sorted(
    scratch_results["training_seed"].unique()
):

    seed_results = scratch_results[
        scratch_results["training_seed"] == seed
    ]

    ax.plot(
        seed_results["timestep"] / 1_000_000,
        seed_results["mean_return"],
        color=scratch_colour,
        alpha=0.15,
        linewidth=1
    )


# Plot each SB3 training seed
for seed in sorted(
    sb3_results["training_seed"].unique()
):

    seed_results = sb3_results[
        sb3_results["training_seed"] == seed
    ]

    ax.plot(
        seed_results["timestep"] / 1_000_000,
        seed_results["mean_return"],
        color=sb3_colour,
        alpha=0.15,
        linewidth=1
    )


# Plot scratch mean
ax.plot(
    scratch_summary["timestep"] / 1_000_000,
    scratch_summary["mean"],
    color=scratch_colour,
    linewidth=2.5,
    label="Scratch PPO"
)


# Plot scratch mean +/- one standard deviation
ax.fill_between(
    scratch_summary["timestep"] / 1_000_000,
    scratch_summary["mean"] - scratch_summary["std"],
    scratch_summary["mean"] + scratch_summary["std"],
    color=scratch_colour,
    alpha=0.15
)


# Plot SB3 mean
ax.plot(
    sb3_summary["timestep"] / 1_000_000,
    sb3_summary["mean"],
    color=sb3_colour,
    linewidth=2.5,
    label="Stable-Baselines3"
)


# Plot SB3 mean +/- one standard deviation
ax.fill_between(
    sb3_summary["timestep"] / 1_000_000,
    sb3_summary["mean"] - sb3_summary["std"],
    sb3_summary["mean"] + sb3_summary["std"],
    color=sb3_colour,
    alpha=0.15
)


ax.set_xlabel(
    "Environment interactions (millions)"
)

ax.set_ylabel(
    "Mean deterministic evaluation return"
)

ax.legend()

ax.grid(
    alpha=0.2
)

fig.tight_layout()


# Save learning curve
fig.savefig(
    "figures/halfcheetah_learning_curve.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    "figures/halfcheetah_learning_curve.pdf",
    bbox_inches="tight"
)

plt.show()


# Diagnostic summaries

diagnostic_names = [
    "explained_variance",
    "approx_kl",
    "clip_fraction",
    "policy_std"
]

diagnostic_titles = [
    "Explained variance",
    "Approximate KL divergence",
    "Clip fraction",
    "Policy standard deviation"
]

scratch_diagnostic_summaries = {}
sb3_diagnostic_summaries = {}

for diagnostic in diagnostic_names:

    scratch_diagnostic_summaries[diagnostic] = (
        scratch_diagnostics
        .groupby("timestep")[diagnostic]
        .agg(["mean", "std"])
        .reset_index()
    )

    sb3_diagnostic_summaries[diagnostic] = (
        sb3_diagnostics
        .groupby("timestep")[diagnostic]
        .agg(["mean", "std"])
        .reset_index()
    )


# Diagnostic plots

fig, axes = plt.subplots(
    2,
    2,
    figsize=(10, 7)
)

axes = axes.flatten()

for index in range(len(diagnostic_names)):

    diagnostic = diagnostic_names[index]
    title = diagnostic_titles[index]
    ax = axes[index]

    scratch_summary_diagnostic = (
        scratch_diagnostic_summaries[diagnostic]
    )

    sb3_summary_diagnostic = (
        sb3_diagnostic_summaries[diagnostic]
    )


    # Plot scratch mean
    ax.plot(
        scratch_summary_diagnostic["timestep"] / 1_000_000,
        scratch_summary_diagnostic["mean"],
        color=scratch_colour,
        linewidth=2,
        label="Scratch PPO"
    )


    # Plot scratch mean +/- one standard deviation
    ax.fill_between(
        scratch_summary_diagnostic["timestep"] / 1_000_000,
        scratch_summary_diagnostic["mean"]
        - scratch_summary_diagnostic["std"],
        scratch_summary_diagnostic["mean"]
        + scratch_summary_diagnostic["std"],
        color=scratch_colour,
        alpha=0.15
    )


    # Plot SB3 mean
    ax.plot(
        sb3_summary_diagnostic["timestep"] / 1_000_000,
        sb3_summary_diagnostic["mean"],
        color=sb3_colour,
        linewidth=2,
        label="Stable-Baselines3"
    )


    # Plot SB3 mean +/- one standard deviation
    ax.fill_between(
        sb3_summary_diagnostic["timestep"] / 1_000_000,
        sb3_summary_diagnostic["mean"]
        - sb3_summary_diagnostic["std"],
        sb3_summary_diagnostic["mean"]
        + sb3_summary_diagnostic["std"],
        color=sb3_colour,
        alpha=0.15
    )


    ax.set_title(
        title
    )

    ax.grid(
        alpha=0.2
    )


# Axis labels
axes[0].set_ylabel(
    "Explained variance"
)

axes[1].set_ylabel(
    "Approximate KL"
)

axes[2].set_ylabel(
    "Clip fraction"
)

axes[3].set_ylabel(
    "Policy standard deviation"
)

axes[2].set_xlabel(
    "Environment interactions (millions)"
)

axes[3].set_xlabel(
    "Environment interactions (millions)"
)


# Shared legend below the plots
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=2,
    bbox_to_anchor=(0.5, -0.01)
)

fig.tight_layout(
    rect=[0, 0.06, 1, 1]
)


# Save diagnostic figure
fig.savefig(
    "figures/halfcheetah_diagnostics.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    "figures/halfcheetah_diagnostics.pdf",
    bbox_inches="tight"
)

plt.show()

# Final performance summary

final_timestep = 999_424

scratch_final = scratch_results[
    scratch_results["timestep"] == final_timestep
]["mean_return"]

sb3_final = sb3_results[
    sb3_results["timestep"] == final_timestep
]["mean_return"]


scratch_final_summary = {
    "implementation": "Scratch PPO",
    "mean": scratch_final.mean(),
    "median": scratch_final.median(),
    "std": scratch_final.std(),
    "min": scratch_final.min(),
    "max": scratch_final.max()
}

sb3_final_summary = {
    "implementation": "Stable-Baselines3",
    "mean": sb3_final.mean(),
    "median": sb3_final.median(),
    "std": sb3_final.std(),
    "min": sb3_final.min(),
    "max": sb3_final.max()
}


final_summary = pd.DataFrame(
    [
        scratch_final_summary,
        sb3_final_summary
    ]
)


print()
print("Final performance summary:")
print(
    final_summary.to_string(
        index=False
    )
)


# Final return for each training seed

scratch_final_seeds = scratch_results[
    scratch_results["timestep"] == final_timestep
][
    [
        "training_seed",
        "mean_return"
    ]
].copy()

scratch_final_seeds["implementation"] = "Scratch PPO"


sb3_final_seeds = sb3_results[
    sb3_results["timestep"] == final_timestep
][
    [
        "training_seed",
        "mean_return"
    ]
].copy()

sb3_final_seeds["implementation"] = "Stable-Baselines3"


final_seed_results = pd.concat(
    [
        scratch_final_seeds,
        sb3_final_seeds
    ],
    ignore_index=True
)


# Save numerical summaries

final_summary.to_csv(
    "results/halfcheetah_final_summary.csv",
    index=False
)

final_seed_results.to_csv(
    "results/halfcheetah_final_seed_results.csv",
    index=False
)

# Late-training diagnostic summary

late_updates = 50

diagnostic_names = [
    "explained_variance",
    "approx_kl",
    "clip_fraction",
    "policy_std"
]


# Calculate the mean diagnostic value over the final
# 50 updates separately for each training seed
scratch_late_diagnostics = (
    scratch_diagnostics
    .groupby("training_seed")
    .tail(late_updates)
    .groupby("training_seed")[diagnostic_names]
    .mean()
    .reset_index()
)

sb3_late_diagnostics = (
    sb3_diagnostics
    .groupby("training_seed")
    .tail(late_updates)
    .groupby("training_seed")[diagnostic_names]
    .mean()
    .reset_index()
)


# Summarise the five seed-level means
late_diagnostic_summary = []

for diagnostic in diagnostic_names:

    late_diagnostic_summary.append(
        {
            "implementation": "Scratch PPO",
            "diagnostic": diagnostic,
            "mean": scratch_late_diagnostics[diagnostic].mean(),
            "std": scratch_late_diagnostics[diagnostic].std()
        }
    )

    late_diagnostic_summary.append(
        {
            "implementation": "Stable-Baselines3",
            "diagnostic": diagnostic,
            "mean": sb3_late_diagnostics[diagnostic].mean(),
            "std": sb3_late_diagnostics[diagnostic].std()
        }
    )


late_diagnostic_summary = pd.DataFrame(
    late_diagnostic_summary
)


print()
print("Late-training diagnostic summary:")
print(
    late_diagnostic_summary.to_string(
        index=False
    )
)


# Save the diagnostic summaries
late_diagnostic_summary.to_csv(
    "results/halfcheetah_late_diagnostic_summary.csv",
    index=False
)

scratch_late_diagnostics.to_csv(
    "results/halfcheetah_scratch_late_diagnostics_by_seed.csv",
    index=False
)

sb3_late_diagnostics.to_csv(
    "results/halfcheetah_sb3_late_diagnostics_by_seed.csv",
    index=False
)