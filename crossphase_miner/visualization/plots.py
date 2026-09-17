"""
Visualization utilities for CrossPhaseMiner simulation results.

Provides a unified Visualizer class for generating plots of signal
observations, TOD comparisons, decision timelines, and cycle time
distributions.  All plots use a consistent color scheme and support
saving to PNG files.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from crossphase_miner.core.models import (
    PhaseTransition,
    SignalColor,
    SignalObservation,
)
from crossphase_miner.core.tod import TODManager

# ---------------------------------------------------------------------------
# Color scheme
# ---------------------------------------------------------------------------

_COLOR_MAP = {
    SignalColor.RED: "red",
    SignalColor.GREEN: "green",
    SignalColor.FLASHING_GREEN: "gold",
    SignalColor.UNKNOWN: "gray",
}

_NUMERIC_COLOR = {
    SignalColor.RED: 0.0,
    SignalColor.GREEN: 1.0,
    SignalColor.FLASHING_GREEN: 0.5,
    SignalColor.UNKNOWN: -0.5,
}


# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------


class Visualizer:
    """Unified visualization interface for CrossPhaseMiner results.

    Wraps matplotlib to provide a consistent look-and-feel across all
    diagnostic plots.  Supports interactive display and file export (PNG).

    Attributes:
        figsize: Default figure size (width, height) in inches.
        _figures: List of active figure handles for lifecycle management.
    """

    def __init__(self, figsize: Tuple[int, int] = (14, 8)) -> None:
        """Initialize the Visualizer.

        Args:
            figsize: Default figure size in inches.
        """
        self.figsize = figsize
        self._figures: List[plt.Figure] = []

    # -- Lifecycle helpers ------------------------------------------------

    def _new_figure(self, title: str) -> Tuple[plt.Figure, plt.Axes]:
        """Create a new figure and axes with the default styling.

        Args:
            title: Plot title.

        Returns:
            Tuple of (Figure, Axes).
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#f8f9fa")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.grid(True, linestyle="--", alpha=0.5)
        self._figures.append(fig)
        return fig, ax

    def _save_or_show(self, fig: plt.Figure, save_path: Optional[str] = None) -> None:
        """Save figure to disk or prepare for display.

        Args:
            fig: Matplotlib Figure handle.
            save_path: If provided, save the figure to this path (PNG).
        """
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight", format="png")
            print(f"    Saved plot to: {save_path}")

    def show(self) -> None:
        """Display all pending figures."""
        plt.show()

    def close(self) -> None:
        """Close all figures created by this Visualizer."""
        for fig in self._figures:
            plt.close(fig)
        self._figures.clear()

    # -- Plot 1: Observations ---------------------------------------------

    def plot_observations(
        self,
        observations: List[SignalObservation],
        title: str = "Signal Observations",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot a time-series of signal color observations.

        Each observation is rendered as a colored scatter point.  The
        numeric encoding (RED=0, FLASHING_GREEN=0.5, GREEN=1) makes
        phase patterns easy to spot at a glance.

        Args:
            observations: List of SignalObservation to plot.
            title: Plot title.
            save_path: Optional PNG file path to save the figure.
        """
        if not observations:
            print("[Visualizer] No observations to plot.")
            return

        fig, ax = self._new_figure(title)

        # Sort by timestamp
        sorted_obs = sorted(observations, key=lambda o: o.timestamp)
        t0 = sorted_obs[0].timestamp
        times = [obs.timestamp - t0 for obs in sorted_obs]
        colors = [_NUMERIC_COLOR[obs.color] for obs in sorted_obs]
        point_colors = [_COLOR_MAP[obs.color] for obs in sorted_obs]
        confidences = [obs.confidence for obs in sorted_obs]

        # Size points by confidence
        sizes = [20 + 80 * c for c in confidences]

        ax.scatter(
            times,
            colors,
            c=point_colors,
            s=sizes,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.3,
        )
        ax.axhline(0.0, color="red", linestyle="-", alpha=0.3, linewidth=1)
        ax.axhline(1.0, color="green", linestyle="-", alpha=0.3, linewidth=1)
        ax.set_yticks([-0.5, 0.0, 0.5, 1.0])
        ax.set_yticklabels(["UNKNOWN", "RED", "FLASHING_GREEN", "GREEN"])
        ax.set_xlabel("Time (seconds since first observation)", fontsize=11)
        ax.set_ylabel("Signal Color", fontsize=11)
        ax.set_ylim(-0.8, 1.3)

        self._save_or_show(fig, save_path)

    def plot_tod_comparison(
        self,
        tod_manager: "TODManager",
        intersection_id: str,
        title: str = "TOD Models",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot a comparison of TOD period models for an intersection.

        Displays a grouped bar chart of T_cycle, T_red, and T_green for
        each TOD period that has a learned model.

        Args:
            tod_manager: The TODManager containing learned models.
            intersection_id: Intersection to visualize.
            title: Plot title.
            save_path: Optional PNG file path to save the figure.
        """
        profile = tod_manager.get_profile(intersection_id)
        if profile is None or not profile.tod_models:
            print(f"[Visualizer] No TOD models for '{intersection_id}' to plot.")
            return

        fig, ax = self._new_figure(f"{title} – {intersection_id}")

        tod_labels = list(profile.tod_models.keys())
        t_cycles = [profile.tod_models[label].T_cycle for label in tod_labels]
        t_reds = [profile.tod_models[label].T_red for label in tod_labels]
        t_greens = [profile.tod_models[label].T_green for label in tod_labels]

        x = np.arange(len(tod_labels))
        width = 0.25

        bars1 = ax.bar(x - width, t_cycles, width, label="T_cycle", color="steelblue")
        bars2 = ax.bar(x, t_reds, width, label="T_red", color="coral")
        bars3 = ax.bar(x + width, t_greens, width, label="T_green", color="mediumseagreen")

        ax.set_ylabel("Duration (seconds)", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(tod_labels, rotation=15, ha="right")
        ax.legend(loc="upper right")

        # Add value labels on bars
        for bars in (bars1, bars2, bars3):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.0f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

        plt.tight_layout()
        self._save_or_show(fig, save_path)

    # -- Plot 5: Cycle Distribution ---------------------------------------

    def _compute_cycle_intervals(self, transitions: List[PhaseTransition]) -> List[float]:
        """Compute full-cycle intervals from a list of transitions."""
        red_to_green: List[float] = []
        green_to_red: List[float] = []
        for t in transitions:
            if t.from_color == SignalColor.RED and t.to_color == SignalColor.GREEN:
                red_to_green.append(t.timestamp)
            elif t.from_color == SignalColor.GREEN and t.to_color == SignalColor.RED:
                green_to_red.append(t.timestamp)

        intervals: List[float] = []
        for ts in (red_to_green, green_to_red):
            ts_sorted = sorted(ts)
            for i in range(1, len(ts_sorted)):
                interval = ts_sorted[i] - ts_sorted[i - 1]
                if 30 <= interval <= 600:
                    intervals.append(interval)
        return intervals

    def plot_cycle_distribution(
        self,
        transitions: List[PhaseTransition],
        title: str = "Cycle Time Distribution",
        save_path: Optional[str] = None,
        tod_manager: Optional["TODManager"] = None,
        ground_truth: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    ) -> None:
        """Plot per-intersection / per-TOD cycle-time distributions.

        Transitions are grouped by intersection (and by TOD period when a
        TODManager is supplied).  Each panel shows the histogram of inferred
        full-cycle intervals with ground-truth cycle lengths as reference
        lines.

        Args:
            transitions: List of PhaseTransition events.
            title: Plot title.
            save_path: Optional PNG file path to save the figure.
            tod_manager: Optional TODManager used to assign transitions to
                TOD periods.
            ground_truth: Optional mapping ``{intersection_id: {period:
                {"T_cycle": float, ...}}}`` for reference lines.
        """
        if not transitions:
            print("[Visualizer] No transitions to plot.")
            return

        # Group transitions by intersection, then optionally by TOD period.
        grouped: Dict[str, Dict[str, List[PhaseTransition]]] = {}
        for t in transitions:
            iid = t.intersection_id
            period = "all"
            if tod_manager is not None:
                period = tod_manager.classify_tod(t.timestamp)
            grouped.setdefault(iid, {}).setdefault(period, []).append(t)

        intersections = sorted(grouped.keys())
        if not intersections:
            print("[Visualizer] No intersections to plot.")
            return

        cols = min(2, len(intersections))
        rows = (len(intersections) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 4.5), squeeze=False)
        fig.patch.set_facecolor("white")
        fig.suptitle(title, fontsize=14, fontweight="bold")
        self._figures.append(fig)

        for idx, iid in enumerate(intersections):
            ax = axes[idx // cols, idx % cols]
            period_map = grouped[iid]
            periods = sorted(period_map.keys())

            # Use different colors for TOD periods.
            colors = plt.cm.tab10(np.linspace(0, 0.45, max(len(periods), 1)))
            for p_idx, (period, trans) in enumerate(period_map.items()):
                intervals = self._compute_cycle_intervals(trans)
                if not intervals:
                    continue
                label = f"{period} (n={len(intervals)}, μ={np.mean(intervals):.1f}s)"
                n_bins = min(15, max(5, len(intervals) // 2))
                ax.hist(
                    intervals,
                    bins=n_bins,
                    color=colors[p_idx],
                    edgecolor="black",
                    alpha=0.6,
                    label=label,
                )

            # Ground-truth reference lines.
            gt_periods = ground_truth.get(iid, {}) if ground_truth else {}
            for period, params in sorted(gt_periods.items()):
                if "T_cycle" in params:
                    linestyle = "--" if period == "all" else ":"
                    ax.axvline(
                        params["T_cycle"],
                        color="red",
                        linestyle=linestyle,
                        linewidth=2,
                        label=f"GT {period}: {params['T_cycle']:.0f}s",
                    )

            ax.set_xlabel("Cycle Time (seconds)", fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            ax.set_title(iid)
            ax.legend(loc="upper right", fontsize=7)
            ax.grid(True, linestyle="--", alpha=0.4)

        for idx in range(len(intersections), rows * cols):
            axes[idx // cols, idx % cols].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        self._save_or_show(fig, save_path)

    # -- Plot 6: Ground-Truth Convergence -----------------------------------

    def plot_ground_truth_convergence(
        self,
        convergence_data: Dict[str, Dict[str, Dict[str, List[float]]]],
        title: str = "Ground-Truth Estimation Error vs. Observed Transitions",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot how ground-truth estimation error decreases as more transitions
        are observed per intersection and per TOD period.

        Args:
            convergence_data: Mapping from intersection_id to TOD period to a
                dict with keys ``steps``, ``color_error``, ``cycle_rel_error``,
                ``time_to_green_mae``.
            title: Figure title.
            save_path: Optional PNG file path. If provided, one file per
                intersection is written by appending ``_<intersection_id>``.
        """
        if not convergence_data:
            print("[Visualizer] No convergence data to plot.")
            return

        for iid, period_map in sorted(convergence_data.items()):
            periods = sorted(period_map.keys())
            if not periods:
                continue

            cols = min(2, len(periods))
            rows = (len(periods) + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 4.5), squeeze=False)
            fig.patch.set_facecolor("white")
            fig.suptitle(f"{title} – {iid}", fontsize=14, fontweight="bold")
            self._figures.append(fig)

            for idx, period in enumerate(periods):
                ax = axes[idx // cols, idx % cols]
                data = period_map[period]
                steps = data["steps"]
                color_err = data["color_error"]
                cycle_err = data["cycle_rel_error"]
                ttg_mae = data.get("time_to_green_mae", [])

                ax.plot(
                    steps,
                    color_err,
                    "o-",
                    color="coral",
                    markersize=3,
                    label="Color prediction error",
                )
                ax.plot(
                    steps,
                    cycle_err,
                    "s-",
                    color="steelblue",
                    markersize=3,
                    label="T_cycle relative error",
                )

                has_ttg = bool(ttg_mae) and any(np.isfinite(ttg_mae))
                if has_ttg:
                    ax2 = ax.twinx()
                    ax2.plot(
                        steps, ttg_mae, "^-", color="green", markersize=3, label="Time-to-green MAE"
                    )
                    ax2.set_ylabel("Time-to-green MAE (s)", color="green")
                    ax2.tick_params(axis="y", labelcolor="green")
                    ax2.legend(loc="upper right")

                ax.set_xlabel("Transitions observed")
                ax.set_ylabel("Error")
                ax.set_title(period)
                ax.set_ylim(-0.05, 1.05)
                ax.grid(True, linestyle="--", alpha=0.5)
                ax.legend(loc="center right")

            # Hide unused subplots
            for idx in range(len(periods), rows * cols):
                axes[idx // cols, idx % cols].axis("off")

            plt.tight_layout(rect=[0, 0, 1, 0.96])

            if save_path:
                base, ext = os.path.splitext(save_path)
                per_iid_path = f"{base}_{iid}{ext}"
                self._save_or_show(fig, per_iid_path)
            else:
                self._save_or_show(fig, None)

    # -- Plot 7: Countdown comparison across arrivals ----------------------

    def plot_countdown_examples(
        self,
        examples: Dict[int, Tuple[List[float], List[float], List[float]]],
        title: str = "Countdown to Green - Learning Progress",
        save_path: Optional[str] = None,
    ) -> None:
        """Compare countdown predictions across several arrivals.

        One row per example arrival (e.g. first predictable / middle /
        last): ground truth (dashed green) vs. model prediction (blue).
        A missing prediction line means the model was not reliable yet -
        the rows together show the learning progress.

        Args:
            examples: {arrival_number: (seconds, pred_ttg, gt_ttg)}.
            title: Figure title.
            save_path: Optional PNG file path.
        """
        if not examples:
            print("[Visualizer] No countdown examples to plot.")
            return

        n = len(examples)
        fig, axes = plt.subplots(
            n,
            1,
            figsize=(14, 3.2 * n),
            sharex=False,
        )
        if n == 1:
            axes = [axes]
        fig.patch.set_facecolor("white")
        fig.suptitle(title, fontsize=14, fontweight="bold")
        self._figures.append(fig)

        for ax, (arr_no, (secs, pred, gt)) in zip(axes, sorted(examples.items())):
            mae = np.nanmean(np.abs(np.array(pred) - np.array(gt)))
            ax.plot(
                secs,
                pred,
                "o-",
                color="#1f77b4",
                linewidth=1.2,
                markersize=2.5,
                label="Model prediction",
            )
            ax.plot(
                secs,
                gt,
                "--",
                color="darkgreen",
                linewidth=1.8,
                label="Ground truth",
            )
            ax.axhline(0, color="green", linestyle=":", alpha=0.4)
            # Per-row y-limit: examples have very different wait durations.
            row_max = max(
                (v for v in list(pred) + list(gt) if np.isfinite(v)),
                default=60.0,
            )
            ax.set_ylim(-0.02 * row_max, row_max * 1.1 + 1.0)
            ax.set_xlabel("Seconds since arrival", fontsize=10)
            ax.set_ylabel("Seconds to GREEN", fontsize=10)
            ax.set_title(f"Arrival #{arr_no} (MAE={mae:.1f}s)", fontsize=11, loc="left")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="upper right", fontsize=8)

        fig.subplots_adjust(top=0.93, hspace=0.45)
        self._save_or_show(fig, save_path)

    # -- Plot 8: Arrival-level countdown convergence ----------------------

    def plot_arrival_convergence(
        self,
        arrival_errors: List[Tuple[int, float]],
        example_arrivals: Optional[Dict[int, Tuple[List[int], List[float], List[float]]]] = None,
        title: str = "Countdown Prediction Error vs. Arrival Number",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot how countdown prediction error improves over repeated arrivals.

        Args:
            arrival_errors: List of (arrival_number, mean_absolute_error_seconds).
            example_arrivals: Optional mapping from arrival number to a tuple of
                (seconds, predicted_ttg, ground_truth_ttg) for drawing example
                countdown curves.
            title: Plot title.
            save_path: Optional PNG file path.
        """
        if not arrival_errors:
            print("[Visualizer] No arrival convergence data to plot.")
            return

        if example_arrivals:
            n_examples = len(example_arrivals)
            fig = plt.figure(figsize=(16, 4 + 3 * n_examples))
            gs = fig.add_gridspec(2 + n_examples, 1, height_ratios=[1] * (2 + n_examples))
            ax_main = fig.add_subplot(gs[0, :])
            ax_trend = fig.add_subplot(gs[1, :], sharex=ax_main)
            example_axes = [fig.add_subplot(gs[i + 2, :]) for i in range(n_examples)]
        else:
            fig = plt.figure(figsize=(14, 8))
            gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])
            ax_main = fig.add_subplot(gs[0, :])
            ax_trend = fig.add_subplot(gs[1, :], sharex=ax_main)
            example_axes = []

        fig.patch.set_facecolor("white")
        self._figures.append(fig)
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

        arrivals, errors = zip(*arrival_errors)
        arrivals_arr = np.array(arrivals, dtype=float)
        errors_arr = np.array(errors, dtype=float)

        # Main error scatter + running mean.
        ax_main.scatter(
            arrivals_arr,
            errors_arr,
            c="coral",
            s=80,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
            zorder=3,
        )

        # Running mean (cumulative).
        if len(errors_arr) >= 2:
            running_mean = np.cumsum(errors_arr) / np.arange(1, len(errors_arr) + 1)
            ax_main.plot(
                arrivals_arr,
                running_mean,
                "-",
                color="darkred",
                linewidth=2,
                label="Cumulative mean error",
            )

        ax_main.axhline(0, color="green", linestyle="--", alpha=0.4)
        ax_main.set_ylabel("Countdown MAE (seconds)", fontsize=11)
        ax_main.set_title(
            "Per-arrival countdown prediction error",
            fontsize=12,
            fontweight="bold",
            loc="left",
        )
        ax_main.grid(True, linestyle="--", alpha=0.4)
        ax_main.legend(loc="upper right")

        # Trend: best-fit exponential decay.
        if len(errors_arr) >= 3:
            # Fit a simple exponential decay: y = a * exp(-b * x) + c
            # Use log-linear fit for robustness.
            x = arrivals_arr
            y = errors_arr
            # Offset to avoid log(0) and model floor.
            floor = max(0.0, np.min(y) - 1.0)
            log_y = np.log(np.maximum(y - floor, 1e-6))
            A = np.vstack([x, np.ones_like(x)]).T
            m, c_log = np.linalg.lstsq(A, log_y, rcond=None)[0]
            a_fit = np.exp(c_log)
            b_fit = -m
            x_fit = np.linspace(x.min(), x.max(), 200)
            y_fit = a_fit * np.exp(-b_fit * x_fit) + floor
            ax_trend.plot(
                x_fit,
                y_fit,
                "--",
                color="steelblue",
                linewidth=2,
                label=f"Exp decay fit (a={a_fit:.1f}, b={b_fit:.3f})",
            )

        ax_trend.scatter(arrivals_arr, errors_arr, c="coral", s=50, alpha=0.5, zorder=3)
        ax_trend.set_xlabel("Arrival number", fontsize=11)
        ax_trend.set_ylabel("Countdown MAE (seconds)", fontsize=11)
        ax_trend.set_title(
            "Error trend (should decrease as more transitions are observed)",
            fontsize=12,
            fontweight="bold",
            loc="left",
        )
        ax_trend.grid(True, linestyle="--", alpha=0.4)
        ax_trend.legend(loc="upper right")

        # Example countdown curves for selected arrivals.
        for ax, (arr_no, (secs, pred, gt)) in zip(example_axes, sorted(example_arrivals.items())):
            ax.plot(
                secs,
                pred,
                "o-",
                color="coral",
                markersize=2,
                linewidth=1.2,
                label="Predicted seconds to GREEN",
            )
            ax.plot(
                secs,
                gt,
                "s--",
                color="darkgreen",
                markersize=2,
                linewidth=1.5,
                alpha=0.8,
                label="Ground-truth seconds to GREEN",
            )
            ax.set_xlabel("Seconds since arrival", fontsize=9)
            ax.set_ylabel("Seconds to GREEN", fontsize=9)
            ax.set_title(
                f"Arrival #{arr_no} countdown (MAE={np.nanmean(np.abs(np.array(pred) - np.array(gt))):.1f}s)",
                fontsize=10,
                fontweight="bold",
                loc="left",
            )
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend(loc="upper right", fontsize=7)

        fig.subplots_adjust(top=0.93, hspace=0.35)
        self._save_or_show(fig, save_path)

    # -- Summary plots (compact evaluation output) -------------------------

    def plot_learner_comparison(
        self,
        convergence: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]],
        period: str,
        title: str = "Learner Convergence Comparison",
        save_path: Optional[str] = None,
    ) -> None:
        """Compare Bayesian and UKF convergence in one figure.

        One panel per intersection.  Following scientific plotting
        convention, line COLOR encodes the learner (blue: Bayesian, red:
        UKF) and the MARKER encodes the metric (circle: color prediction
        error, star: T_cycle relative error).

        Args:
            convergence: {learner_name: {iid: {period: series}}} as produced
                by ``compute_convergence_curves``.
            period: TOD period to show (typically "day", which has the most
                data).
            title: Figure title.
            save_path: Optional PNG file path.
        """
        learners = sorted(convergence.keys())
        iids = sorted(
            {
                iid
                for data in convergence.values()
                for iid, pm in data.items()
                if period in pm and pm[period]["steps"]
            }
        )
        if not iids:
            print("[Visualizer] No convergence data to compare.")
            return

        cols = min(2, len(iids))
        rows = (len(iids) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 4.5), squeeze=False)
        fig.patch.set_facecolor("white")
        fig.suptitle(f"{title} ({period})", fontsize=14, fontweight="bold")
        self._figures.append(fig)

        learner_color = {"bayesian": "#1f77b4", "ukf": "#d62728"}

        for idx, iid in enumerate(iids):
            ax = axes[idx // cols, idx % cols]
            for learner in learners:
                series = convergence[learner].get(iid, {}).get(period)
                if not series:
                    continue
                color = learner_color.get(learner, "gray")
                steps = series["steps"]
                ax.plot(
                    steps,
                    series["color_error"],
                    "o-",
                    color=color,
                    markersize=4,
                    linewidth=1.5,
                    label=f"{learner} color error",
                )
                ax.plot(
                    steps,
                    series["cycle_rel_error"],
                    "*--",
                    color=color,
                    markersize=9,
                    linewidth=1.5,
                    markeredgecolor="black",
                    markeredgewidth=0.4,
                    label=f"{learner} T_cycle error",
                )
            ax.set_xlabel("Arrival number")
            ax.set_ylabel("Error")
            ax.set_title(iid)
            ax.set_ylim(-0.05, 1.05)
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="upper right", fontsize=8)

        for idx in range(len(iids), rows * cols):
            axes[idx // cols, idx % cols].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        self._save_or_show(fig, save_path)

    def plot_mae_summary(
        self,
        arrival_convergence: Dict[str, Dict[str, Tuple[List[Tuple[int, float]], dict]]],
        tod_manager: "TODManager",
        ground_truth: Dict[str, Dict[str, Dict[str, float]]],
        period: str,
        title: str = "Countdown Prediction Error vs. Arrival Number",
        save_path: Optional[str] = None,
    ) -> None:
        """One-figure summary of the end-to-end result.

        One panel per intersection: countdown MAE per arrival (with running
        mean), titled with the final learned parameters vs. ground truth.

        Args:
            arrival_convergence: {iid: {period: (arrival_errors, examples)}}
                as produced by ``compute_arrival_convergence``.
            tod_manager: Final learned models (for the panel titles).
            ground_truth: {iid: {period: {"T_cycle": ..., "T_red": ...}}}.
            period: TOD period to show.
            title: Figure title.
            save_path: Optional PNG file path.
        """
        iids = sorted(iid for iid, pm in arrival_convergence.items() if period in pm)
        if not iids:
            print("[Visualizer] No arrival convergence data to plot.")
            return

        cols = min(2, len(iids))
        rows = (len(iids) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 4.5), squeeze=False)
        fig.patch.set_facecolor("white")
        fig.suptitle(f"{title} ({period})", fontsize=14, fontweight="bold")
        self._figures.append(fig)

        # Shared y-axis across all panels so intersections are comparable.
        all_errors = [
            e for iid in iids for _, e in arrival_convergence[iid][period][0] if np.isfinite(e)
        ]
        y_max = max(all_errors) * 1.08 if all_errors else 1.0

        for idx, iid in enumerate(iids):
            ax = axes[idx // cols, idx % cols]
            arr_errors, _ = arrival_convergence[iid][period]
            arrivals = np.array([a for a, _ in arr_errors], dtype=float)
            errors = np.array([e for _, e in arr_errors], dtype=float)
            finite = np.isfinite(errors)

            ax.scatter(
                arrivals[~finite],
                np.zeros(int(np.sum(~finite))),
                c="gray",
                s=40,
                alpha=0.5,
                marker="x",
                label="No reliable model yet",
            )
            ax.scatter(
                arrivals[finite],
                errors[finite],
                c="coral",
                s=60,
                alpha=0.8,
                edgecolors="black",
                linewidth=0.5,
                label="Countdown MAE",
                zorder=3,
            )
            if np.sum(finite) >= 2:
                running = np.cumsum(errors[finite]) / np.arange(1, int(np.sum(finite)) + 1)
                ax.plot(
                    arrivals[finite],
                    running,
                    "-",
                    color="darkred",
                    linewidth=2,
                    label="Cumulative mean",
                )

            # Panel title: learned vs. true parameters.
            model = None
            profile = tod_manager.get_profile(iid)
            if profile is not None:
                model = profile.tod_models.get(period)
            true = ground_truth.get(iid, {}).get(period, {})
            if model is not None and true:
                ax.set_title(
                    f"{iid}\n"
                    f"T_cycle {model.T_cycle:.1f}/{true.get('T_cycle', 0):.0f}s, "
                    f"T_red {model.T_red:.1f}/{true.get('T_red', 0):.0f}s "
                    f"(learned/true)",
                    fontsize=10,
                )
            else:
                ax.set_title(iid)

            ax.set_xlabel("Arrival number")
            ax.set_ylabel("Countdown MAE (s)")
            ax.set_ylim(-0.02 * y_max, y_max)
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="upper right", fontsize=8)

        for idx in range(len(iids), rows * cols):
            axes[idx // cols, idx % cols].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        self._save_or_show(fig, save_path)
