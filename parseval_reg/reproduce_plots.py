"""Driver to generate paper-style plots from saved results.

Usage:
    python reproduce_plots.py --env gridworld --algs base parseval --num_repeats 6
    python reproduce_plots.py --env metaworld --algs base parseval --num_repeats 3

Writes PNGs into ./plots/.  Uses the Agg backend so it runs headless.
"""

import os
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plotting


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="gridworld", choices=["gridworld", "metaworld", "carl_lunarlander", "carl_dmcquadruped"])
    ap.add_argument("--algs", nargs="+", default=["base", "parseval"])
    ap.add_argument("--num_repeats", type=int, default=None)
    ap.add_argument("--num_task_sequences", type=int, default=None)
    ap.add_argument("--running_mean_window", type=int, default=5)
    ap.add_argument("--save_freq", type=int, default=None)
    ap.add_argument("--num_steps", type=int, default=None)
    ap.add_argument("--change_freq", type=int, default=None)
    ap.add_argument("--load_path", default="results/")
    ap.add_argument("--plot_dir", default="plots/")
    args = ap.parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)

    # Defaults per env (mirroring plotting.py / main.py)
    defaults = {
        "gridworld": dict(num_repeats=6, num_task_sequences=1, save_freq=5000, num_steps=800000, change_freq=40000),
        "metaworld": dict(num_repeats=3, num_task_sequences=20, save_freq=25000, num_steps=10_000_000, change_freq=1_000_000),
        "carl_lunarlander": dict(num_repeats=3, num_task_sequences=20, save_freq=25000, num_steps=10_000_000, change_freq=500000),
        "carl_dmcquadruped": dict(num_repeats=3, num_task_sequences=20, save_freq=25000, num_steps=12_000_000, change_freq=1_500_000),
    }[args.env]

    num_repeats = args.num_repeats or defaults["num_repeats"]
    num_task_sequences = args.num_task_sequences or defaults["num_task_sequences"]
    save_freq = args.save_freq or defaults["save_freq"]
    num_steps = args.num_steps or defaults["num_steps"]
    change_freq = args.change_freq or defaults["change_freq"]

    # plot_performance_profile reads these two as module globals (upstream bug);
    # inject them so we can call it without editing plotting.py.
    plotting.num_repeats = num_repeats
    plotting.num_task_sequences = num_task_sequences

    print(f"Plotting learning curves for {args.env} ({args.algs}, repeats={num_repeats})")
    plotting.plot_learning_curves(
        args.load_path, args.algs, args.env,
        num_repeats=num_repeats, num_task_sequences=num_task_sequences,
        save_freq=save_freq, running_mean_window=args.running_mean_window,
        plot_save_path=args.plot_dir,
    )
    plt.close("all")

    print(f"Plotting performance profile for {args.env}")
    plotting.plot_performance_profile(
        args.load_path, args.algs, args.env,
        save_freq=save_freq, change_freq=change_freq, num_steps=num_steps,
        plot_save_path=args.plot_dir,
    )
    plt.close("all")

    print("Done. Plots written to", args.plot_dir)


if __name__ == "__main__":
    main()
