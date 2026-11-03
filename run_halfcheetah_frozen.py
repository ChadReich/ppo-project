import argparse
import csv
import os
import subprocess
import sys


training_seeds = [0, 1, 2, 3, 4]

result_files = [
    "results/halfcheetah_scratch_frozen.csv",
    "results/halfcheetah_scratch_frozen_diagnostics.csv",
    "results/halfcheetah_scratch_frozen_runtime.csv",
    "results/halfcheetah_sb3_frozen.csv",
    "results/halfcheetah_sb3_frozen_diagnostics.csv",
    "results/halfcheetah_sb3_frozen_runtime.csv"
]


def completed_seeds(
    path
):
    """
    Read the runtime CSV and return the seeds already completed.
    """

    if not os.path.exists(
        path
    ):
        return set()

    seeds = set()

    with open(
        path,
        newline=""
    ) as csv_file:

        reader = csv.DictReader(
            csv_file
        )

        for row in reader:

            seeds.add(
                int(
                    row["training_seed"]
                )
            )

    return seeds


def run_script(
    script,
    seed
):
    """
    Run one implementation for one seed using the active Python environment.
    """

    subprocess.run(
        [
            sys.executable,
            script,
            "--seed",
            str(seed)
        ],
        check=True
    )


parser = argparse.ArgumentParser()

parser.add_argument(
    "--restart",
    action="store_true",
    help="Delete existing HalfCheetah frozen result files and restart from seed 0."
)

args = parser.parse_args()

os.makedirs(
    "results",
    exist_ok=True
)

if args.restart:

    for path in result_files:

        if os.path.exists(
            path
        ):
            os.remove(
                path
            )

scratch_completed = completed_seeds(
    "results/halfcheetah_scratch_frozen_runtime.csv"
)

sb3_completed = completed_seeds(
    "results/halfcheetah_sb3_frozen_runtime.csv"
)

for seed in training_seeds:

    if seed not in scratch_completed:

        print()
        print("=" * 60)
        print("Seed", seed, "- scratch PPO")
        print("=" * 60)

        run_script(
            "train_halfcheetah_frozen_scratch_seed.py",
            seed
        )

    else:

        print(
            "Skipping completed scratch seed:",
            seed
        )

    if seed not in sb3_completed:

        print()
        print("=" * 60)
        print("Seed", seed, "- Stable-Baselines3 PPO")
        print("=" * 60)

        run_script(
            "train_halfcheetah_frozen_sb3_seed.py",
            seed
        )

    else:

        print(
            "Skipping completed SB3 seed:",
            seed
        )

print()
print("HalfCheetah frozen efficiency comparison complete.")
print("Results are in the results/ directory.")
