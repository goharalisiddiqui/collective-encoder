import logging
import math
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for headless servers & SLURM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytorch_lightning.loggers import CSVLogger

_log = logging.getLogger(__name__)


def plot_metrics_csv(
    csv_path: str,
    output_dir: Optional[str] = None,
    dpi: int = 300,
    plot_format: str = "png",
    log_scale: bool = True,
    include_step_plots: bool = False,
    title_prefix: str = "Training Progress",
) -> List[str]:
    """Reads a PyTorch Lightning metrics.csv file and generates comprehensive metric plots.

    Args:
        csv_path: Path to the metrics.csv file.
        output_dir: Directory where plots will be saved. Defaults to a 'plots' folder next to csv_path.
        dpi: Resolution of the saved figures.
        plot_format: File format ('png', 'pdf', 'svg').
        log_scale: Whether to generate additional log-scale loss plots.
        include_step_plots: Whether to generate separate plots for raw step-level logs.
        title_prefix: Prefix for plot titles.

    Returns:
        List of generated plot file paths.
    """
    if not os.path.exists(csv_path):
        _log.warning("Metrics CSV file '%s' does not exist. Skipping plot generation.", csv_path)
        return []

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        _log.warning("Failed to read metrics CSV '%s': %s", csv_path, e)
        return []

    if df.empty:
        _log.warning("Metrics CSV '%s' is empty. Skipping plot generation.", csv_path)
        return []

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(csv_path), "plots")
    os.makedirs(output_dir, exist_ok=True)

    generated_files = []

    # Determine primary x-axis: use 'epoch' if available with multiple values, else 'step'
    has_epoch = "epoch" in df.columns and df["epoch"].dropna().nunique() > 1
    x_col = "epoch" if has_epoch else "step"
    if x_col not in df.columns:
        df["step"] = np.arange(len(df))
        x_col = "step"

    # Identify metric columns (excluding x-axis and book-keeping columns)
    ignored_cols = {"step", "epoch", "Unnamed: 0"}
    metric_cols = [c for c in df.columns if c not in ignored_cols and pd.api.types.is_numeric_dtype(df[c])]

    if not metric_cols:
        return []

    # Group metrics into train/val pairs, learning rates, and standalone metrics
    paired_metrics: Dict[str, Dict[str, str]] = {}
    lr_metrics: List[str] = []
    standalone_metrics: List[str] = []

    def _base_name(name: str) -> str:
        clean = name
        for prefix in ["train_", "val_", "test_"]:
            if clean.startswith(prefix):
                clean = clean[len(prefix):]
        for suffix in ["_epoch", "_step"]:
            if clean.endswith(suffix):
                clean = clean[:-len(suffix)]
        return clean

    all_base_names = sorted(list(set(_base_name(c) for c in metric_cols if not c.lower().startswith("lr"))))

    for base in all_base_names:
        train_cand = [c for c in metric_cols if c.startswith("train_") and _base_name(c) == base]
        val_cand = [c for c in metric_cols if c.startswith("val_") and _base_name(c) == base]
        if train_cand or val_cand:
            t_col = next((c for c in train_cand if "_epoch" in c), train_cand[0] if train_cand else None)
            v_col = next((c for c in val_cand if "_epoch" in c), val_cand[0] if val_cand else None)
            paired_metrics[base] = {"train": t_col, "val": v_col}

    handled_cols = set()
    for base, p in paired_metrics.items():
        for cand in [c for c in metric_cols if _base_name(c) == base]:
            handled_cols.add(cand)

    for col in metric_cols:
        if col.lower().startswith("lr") or "-lr" in col.lower() or "lr-" in col.lower():
            lr_metrics.append(col)
            handled_cols.add(col)
        elif col not in handled_cols:
            if not include_step_plots and col.endswith("_step"):
                continue
            standalone_metrics.append(col)

    # ------------------------------------------------------------------
    # 1. Individual Metric Plots
    # ------------------------------------------------------------------
    for base, pair in paired_metrics.items():
        t_col = pair.get("train")
        v_col = pair.get("val")

        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=dpi)
        has_data = False

        if t_col and t_col in df.columns:
            sub = df[[x_col, t_col]].dropna()
            if not sub.empty:
                grouped = sub.groupby(x_col)[t_col].mean().reset_index()
                final_val = grouped[t_col].iloc[-1]
                min_val = grouped[t_col].min()
                label = f"Train (Final: {final_val:.4g}, Min: {min_val:.4g})"
                ax.plot(grouped[x_col], grouped[t_col], label=label, color="#1f77b4", linewidth=2.0, marker="o", markersize=3, alpha=0.9)
                has_data = True

        if v_col and v_col in df.columns:
            sub = df[[x_col, v_col]].dropna()
            if not sub.empty:
                grouped = sub.groupby(x_col)[v_col].mean().reset_index()
                final_val = grouped[v_col].iloc[-1]
                min_val = grouped[v_col].min()
                label = f"Validation (Final: {final_val:.4g}, Min: {min_val:.4g})"
                ax.plot(grouped[x_col], grouped[v_col], label=label, color="#ff7f0e", linewidth=2.0, marker="s", markersize=3, alpha=0.9)
                has_data = True

        if has_data:
            metric_title = base.replace("_", " ").title()
            ax.set_title(f"{title_prefix}: {metric_title}", fontsize=13, fontweight="bold", pad=10)
            ax.set_xlabel(x_col.capitalize(), fontsize=11)
            ax.set_ylabel(metric_title, fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.5)
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend(frameon=True, fancybox=True, framealpha=0.9, fontsize=9)
            fig.tight_layout()

            out_file = os.path.join(output_dir, f"{base}.{plot_format}")
            fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            generated_files.append(out_file)

            # Optional log-scale plot for losses
            if log_scale and "loss" in base.lower():
                fig_log, ax_log = plt.subplots(figsize=(7, 4.5), dpi=dpi)
                has_log_data = False
                if t_col and t_col in df.columns:
                    sub = df[[x_col, t_col]].dropna()
                    sub = sub[sub[t_col] > 0]
                    if not sub.empty:
                        grouped = sub.groupby(x_col)[t_col].mean().reset_index()
                        ax_log.plot(grouped[x_col], grouped[t_col], label=f"Train {metric_title}", color="#1f77b4", linewidth=2.0, marker="o", markersize=3)
                        has_log_data = True
                if v_col and v_col in df.columns:
                    sub = df[[x_col, v_col]].dropna()
                    sub = sub[sub[v_col] > 0]
                    if not sub.empty:
                        grouped = sub.groupby(x_col)[v_col].mean().reset_index()
                        ax_log.plot(grouped[x_col], grouped[v_col], label=f"Val {metric_title}", color="#ff7f0e", linewidth=2.0, marker="s", markersize=3)
                        has_log_data = True
                if has_log_data:
                    ax_log.set_yscale("log")
                    ax_log.set_title(f"{title_prefix}: {metric_title} (Log Scale)", fontsize=13, fontweight="bold", pad=10)
                    ax_log.set_xlabel(x_col.capitalize(), fontsize=11)
                    ax_log.set_ylabel(f"{metric_title} (log)", fontsize=11)
                    ax_log.grid(True, which="both", linestyle="--", alpha=0.5)
                    handles_l, labels_l = ax_log.get_legend_handles_labels()
                    if labels_l:
                        ax_log.legend(frameon=True, fancybox=True, framealpha=0.9, fontsize=9)
                    fig_log.tight_layout()

                    out_log_file = os.path.join(output_dir, f"{base}_logscale.{plot_format}")
                    fig_log.savefig(out_log_file, dpi=dpi, bbox_inches="tight")
                    plt.close(fig_log)
                    generated_files.append(out_log_file)

    # ------------------------------------------------------------------
    # 2. Learning Rate Plot
    # ------------------------------------------------------------------
    if lr_metrics:
        fig_lr, ax_lr = plt.subplots(figsize=(7, 4.5), dpi=dpi)
        has_lr_data = False
        for lr_col in lr_metrics:
            sub = df[[x_col, lr_col]].dropna()
            if not sub.empty:
                grouped = sub.groupby(x_col)[lr_col].mean().reset_index()
                ax_lr.plot(grouped[x_col], grouped[lr_col], label=lr_col, color="#2ca02c", linewidth=2.0, marker="^", markersize=3)
                has_lr_data = True

        if has_lr_data:
            ax_lr.set_title(f"{title_prefix}: Learning Rate Schedule", fontsize=13, fontweight="bold", pad=10)
            ax_lr.set_xlabel(x_col.capitalize(), fontsize=11)
            ax_lr.set_ylabel("Learning Rate", fontsize=11)
            ax_lr.set_yscale("log")
            ax_lr.grid(True, which="both", linestyle="--", alpha=0.5)
            handles_lr, labels_lr = ax_lr.get_legend_handles_labels()
            if labels_lr:
                ax_lr.legend(frameon=True, fancybox=True, framealpha=0.9, fontsize=9)
            fig_lr.tight_layout()

            lr_file = os.path.join(output_dir, f"learning_rate.{plot_format}")
            fig_lr.savefig(lr_file, dpi=dpi, bbox_inches="tight")
            plt.close(fig_lr)
            generated_files.append(lr_file)
        else:
            plt.close(fig_lr)

    # ------------------------------------------------------------------
    # 3. Standalone Metrics Plots
    # ------------------------------------------------------------------
    for col in standalone_metrics:
        sub = df[[x_col, col]].dropna()
        if sub.empty:
            continue
        grouped = sub.groupby(x_col)[col].mean().reset_index()
        fig_s, ax_s = plt.subplots(figsize=(7, 4.5), dpi=dpi)
        final_val = grouped[col].iloc[-1]
        ax_s.plot(grouped[x_col], grouped[col], label=f"{col} (Final: {final_val:.4g})", color="#9467bd", linewidth=2.0, marker="d", markersize=3)
        clean_title = col.replace("_", " ").title()
        ax_s.set_title(f"{title_prefix}: {clean_title}", fontsize=13, fontweight="bold", pad=10)
        ax_s.set_xlabel(x_col.capitalize(), fontsize=11)
        ax_s.set_ylabel(clean_title, fontsize=11)
        ax_s.grid(True, linestyle="--", alpha=0.5)
        handles_s, labels_s = ax_s.get_legend_handles_labels()
        if labels_s:
            ax_s.legend(frameon=True, fancybox=True, framealpha=0.9, fontsize=9)
        fig_s.tight_layout()

        out_s_file = os.path.join(output_dir, f"{col}.{plot_format}")
        fig_s.savefig(out_s_file, dpi=dpi, bbox_inches="tight")
        plt.close(fig_s)
        generated_files.append(out_s_file)

    # ------------------------------------------------------------------
    # 4. Multi-Panel Summary Dashboard Figure
    # ------------------------------------------------------------------
    panels_to_plot = []
    for base, pair in paired_metrics.items():
        panels_to_plot.append(("pair", base, pair))
    if lr_metrics:
        # Check if there is actual data for lr
        has_any_lr = any(not df[[x_col, c]].dropna().empty for c in lr_metrics)
        if has_any_lr:
            panels_to_plot.append(("lr", "Learning Rate", lr_metrics))
    for col in standalone_metrics:
        panels_to_plot.append(("single", col, col))

    num_panels = len(panels_to_plot)
    if num_panels > 0:
        ncols = 2 if num_panels <= 4 else 3
        nrows = math.ceil(num_panels / ncols)

        fig_sum, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 4.0 * nrows), dpi=dpi)
        axes_flat = np.array(axes).flatten() if num_panels > 1 else [axes]

        for idx, item in enumerate(panels_to_plot):
            ax = axes_flat[idx]
            ptype = item[0]

            if ptype == "pair":
                base, pair = item[1], item[2]
                t_col, v_col = pair.get("train"), pair.get("val")
                if t_col and t_col in df.columns:
                    sub = df[[x_col, t_col]].dropna()
                    if not sub.empty:
                        g = sub.groupby(x_col)[t_col].mean().reset_index()
                        ax.plot(g[x_col], g[t_col], label="Train", color="#1f77b4", linewidth=1.8, marker="o", markersize=2.5)
                if v_col and v_col in df.columns:
                    sub = df[[x_col, v_col]].dropna()
                    if not sub.empty:
                        g = sub.groupby(x_col)[v_col].mean().reset_index()
                        ax.plot(g[x_col], g[v_col], label="Val", color="#ff7f0e", linewidth=1.8, marker="s", markersize=2.5)
                metric_title = base.replace("_", " ").title()
                ax.set_title(metric_title, fontsize=11, fontweight="bold")
                ax.set_xlabel(x_col.capitalize(), fontsize=9)
                ax.set_ylabel(metric_title, fontsize=9)
                ax.grid(True, linestyle="--", alpha=0.4)
                h, l = ax.get_legend_handles_labels()
                if l:
                    ax.legend(frameon=True, fontsize=8)

            elif ptype == "lr":
                for lr_col in item[2]:
                    sub = df[[x_col, lr_col]].dropna()
                    if not sub.empty:
                        g = sub.groupby(x_col)[lr_col].mean().reset_index()
                        ax.plot(g[x_col], g[lr_col], label=lr_col, color="#2ca02c", linewidth=1.8, marker="^", markersize=2.5)
                ax.set_title("Learning Rate", fontsize=11, fontweight="bold")
                ax.set_xlabel(x_col.capitalize(), fontsize=9)
                ax.set_ylabel("LR", fontsize=9)
                ax.set_yscale("log")
                ax.grid(True, which="both", linestyle="--", alpha=0.4)
                h, l = ax.get_legend_handles_labels()
                if l:
                    ax.legend(frameon=True, fontsize=8)

            elif ptype == "single":
                col = item[1]
                sub = df[[x_col, col]].dropna()
                if not sub.empty:
                    g = sub.groupby(x_col)[col].mean().reset_index()
                    ax.plot(g[x_col], g[col], label=col, color="#9467bd", linewidth=1.8, marker="d", markersize=2.5)
                clean_title = col.replace("_", " ").title()
                ax.set_title(clean_title, fontsize=11, fontweight="bold")
                ax.set_xlabel(x_col.capitalize(), fontsize=9)
                ax.set_ylabel(clean_title, fontsize=9)
                ax.grid(True, linestyle="--", alpha=0.4)
                h, l = ax.get_legend_handles_labels()
                if l:
                    ax.legend(frameon=True, fontsize=8)

        # Hide any unused subplots in the grid
        for idx in range(num_panels, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig_sum.suptitle(f"{title_prefix} Dashboard", fontsize=14, fontweight="bold", y=1.002)
        fig_sum.tight_layout()

        summary_file = os.path.join(output_dir, f"summary.{plot_format}")
        fig_sum.savefig(summary_file, dpi=dpi, bbox_inches="tight")
        plt.close(fig_sum)
        generated_files.append(summary_file)

    return generated_files


class CSVPlotLogger(CSVLogger):
    """Extended PyTorch Lightning CSVLogger that automatically renders training metric plots.

    Inherits from :class:`pytorch_lightning.loggers.CSVLogger`. Logs step and epoch metrics
    to ``metrics.csv``, and automatically parses and visualizes loss curves, component losses,
    learning rate schedules, and a multi-panel summary dashboard upon ``save()`` and ``finalize()``.

    Args:
        save_dir: Base directory for experiment logs.
        name: Experiment subfolder name (default: ``"csv_logs"``).
        version: Experiment version (default: ``"version_0"``).
        prefix: Metric name prefix.
        flush_logs_every_n_steps: Flush frequency.
        plot_each_epoch: If ``True``, regenerates plots at the end of each epoch (default: ``False``).
        plot_dpi: Resolution of saved plots (default: 300).
        plot_format: Plot file extension (default: ``"png"``).
        log_scale: Whether to produce log-scale loss plots.
        mirror_plots_to_savedir: Whether to copy generated plots directly to ``save_dir/training_plots``.
    """

    def __init__(
        self,
        save_dir: str,
        name: Optional[str] = "csv_logs",
        version: Optional[Union[int, str]] = "version_0",
        prefix: str = "",
        flush_logs_every_n_steps: int = 100,
        plot_each_epoch: bool = False,
        plot_on_save: Optional[bool] = None,  # Kept as alias to plot_each_epoch
        plot_dpi: int = 300,
        plot_format: str = "png",
        log_scale: bool = True,
        mirror_plots_to_savedir: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            save_dir=save_dir,
            name=name,
            version=version,
            prefix=prefix,
            flush_logs_every_n_steps=flush_logs_every_n_steps,
        )
        if plot_on_save is not None:
            plot_each_epoch = plot_on_save
        self.plot_each_epoch = bool(plot_each_epoch)
        self.plot_dpi = plot_dpi
        self.plot_format = plot_format
        self.log_scale = log_scale
        self.mirror_plots_to_savedir = mirror_plots_to_savedir
        self._last_plotted_epoch: int = -1

    @property
    def metrics_csv_path(self) -> str:
        return os.path.join(self.log_dir, "metrics.csv")

    @property
    def plots_dir(self) -> str:
        return os.path.join(self.log_dir, "plots")

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        super().log_metrics(metrics, step=step)
        if self.plot_each_epoch and "epoch" in metrics:
            epoch = metrics["epoch"]
            if isinstance(epoch, (int, float)) and int(epoch) > self._last_plotted_epoch:
                self._last_plotted_epoch = int(epoch)
                super().save()  # Flush CSV to disk before rendering
                try:
                    self.generate_plots()
                except Exception as e:
                    _log.debug("Could not generate plots for epoch %d: %s", int(epoch), e)

    def generate_plots(self) -> List[str]:
        """Renders metric plots from the current metrics.csv."""
        csv_file = self.metrics_csv_path
        if not os.path.exists(csv_file):
            return []

        plots = plot_metrics_csv(
            csv_path=csv_file,
            output_dir=self.plots_dir,
            dpi=self.plot_dpi,
            plot_format=self.plot_format,
            log_scale=self.log_scale,
            title_prefix=f"{os.path.basename(self.save_dir)}",
        )

        # Mirror plots to save_dir/training_plots for easy direct access
        if self.mirror_plots_to_savedir and plots:
            mirror_dir = os.path.join(self.save_dir, "training_plots")
            os.makedirs(mirror_dir, exist_ok=True)
            for p in plots:
                shutil.copy2(p, os.path.join(mirror_dir, os.path.basename(p)))

        return plots

    def finalize(self, status: str) -> None:
        """Finalize logger and render final plots."""
        super().finalize(status)
        try:
            self.generate_plots()
        except Exception as e:
            _log.warning("Could not generate plots during finalize(): %s", e)
