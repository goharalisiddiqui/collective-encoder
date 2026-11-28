import os
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from .base import BaseDataAnalyser


def _render_histogram_worker(
    task: Tuple[Dict[str, Any], np.ndarray, str, str]
) -> Optional[str]:
    """
    Worker function that computes histogram statistics and renders a 1-D histogram
    plot to disk.

    Parameters
    ----------
    task : tuple
        Packed ``(spec, raw_arr, outpath, fname)`` tuple.

    Returns
    -------
    str or None
        Saved file path, or ``None`` if no finite values were available.
    """
    spec, raw_arr, outpath, fname = task
    arr_1d = np.asarray(raw_arr, dtype=float).reshape(-1)
    finite_vals = arr_1d[np.isfinite(arr_1d)]
    if spec.get("ignore_zeros", False):
        finite_vals = finite_vals[finite_vals != 0.0]
    if len(finite_vals) == 0:
        return None

    var_key = spec.get("x") or spec.get("label") or spec.get("variable") or "value"
    bins = int(spec.get("bins", 50))
    xlabel = spec.get("xlabel", var_key)
    title = spec.get("title", f"Distribution of {xlabel}")

    plt.switch_backend("Agg")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(
        finite_vals,
        bins=bins,
        color="steelblue",
        edgecolor="black",
        alpha=0.75,
        density=True,
    )
    mean_val = float(np.mean(finite_vals))
    std_val = float(np.std(finite_vals))
    ax.axvline(
        mean_val,
        color="crimson",
        linestyle="--",
        linewidth=1.2,
        label=f"mean={mean_val:.3g} (std={std_val:.3g})",
    )
    if spec.get("angles", False):
        ax.set_xlim([-np.pi, np.pi])
        ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()

    fn = os.path.join(outpath, f"{fname}.png")
    fig.savefig(fn, dpi=150)
    plt.close(fig)
    return fn


class SimpleDataAnalyser(BaseDataAnalyser):
    """
    Data analyser for extracting and plotting label data.

    This analyser fetches properties designated as labels in datasets
    and can create 1D time-series style plots, 1D histograms, 2D scatter plots,
    and correlation matrices based on provided configurations.
    """

    _IDENTIFIER = "SIMPLE"
    _COMPATIBLE_DATASET_TYPES = ["DISTANCES", "POSITIONS"]
    _OPTIONAL_ARGS = BaseDataAnalyser._OPTIONAL_ARGS | {
        "plots_hist_cb": [],
        "plot_histograms": False,
        "parallel": True,
    }
    
    def after_2dscatter(self, 
                        plot : Dict[str, Any],
                        data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]], # x, y, xerr, yerr, labels
                        fig: plt.Figure,
                        axes: np.ndarray
                        ) -> None:
        plot.update({
            "tag":  self.label,
        })
        
    def parse_data(self, data) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Parse the data and labels from the dataset.

        Parameters
        ----------
        data : list
            Data points from the dataset.

        Returns
        -------
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]
            Parsed datapoints and labels as dictionaries.
        """
        
        return data.get_data()
        
    def write_data(self, data, label="") -> None:
        """
        Execute the plotting logic after all batches have been collected.

        Parameters
        ----------
        data : list
            Data points from the dataset.
        """
        self.label = label
        datapoints, labelpoints = self.parse_data(data)
        try:
            datapoints = self._parse_selection(self.data_selection, datapoints, "data")
            labels = self._parse_selection(self.labels_selection, labelpoints, "labels")
        except Exception as e:
            self.log_exception(f"Error occurred while parsing selections: {e}")
            return

        # All names must be unique
        all_names = list(labels.keys()) + list(datapoints.keys())
        if len(all_names) == 0:
            self.log_exception("No labels, data found after parsing selections. Nothing to plot.")
            return

        if len(all_names) != len(set(all_names)):
            raise ValueError(f"Duplicate names found in labels and data keys. "
                             f"All names must be unique. Found names: {all_names}")
        self.vals = {**labels, **datapoints}
        
        self.plot_vals()
        self._plot_configured_histograms(self.vals)

    def _plot_configured_histograms(self, vals: Dict[str, np.ndarray]) -> None:
        """
        Generate 1-D distribution histograms configured via ``plots_hist_cb`` or
        ``plot_histograms``, using parallel workers when ``self.parallel`` is True
        and displaying a ``tqdm`` progress bar.

        Parameters
        ----------
        vals : dict of str to numpy.ndarray
            Parsed dictionary of selected data and label arrays.
        """
        hist_specs = list(self.plots_hist_cb or [])
        if self.plot_histograms and not hist_specs:
            hist_specs = [{"name": f"hist_{k}", "x": k} for k in vals.keys()]

        if not hist_specs:
            return

        suffix = f"_{self.label}" if self.label else ""
        tasks: List[Tuple[Dict[str, Any], np.ndarray, str, str]] = []
        for spec in hist_specs:
            var_key = spec.get("x") or spec.get("label") or spec.get("variable")
            if not var_key or var_key not in vals:
                self.log_warn(
                    f"Histogram variable '{var_key}' not found in selected values "
                    f"{list(vals.keys())}. Skipping."
                )
                continue
            fname = spec.get("name", f"hist_{var_key}") + suffix
            tasks.append((dict(spec), np.asarray(vals[var_key]), self.outpath, fname))

        if not tasks:
            return

        desc = f"Plotting histograms ({self.label})" if self.label else "Plotting histograms"
        if self.parallel and len(tasks) > 1 and self.logger_type != "WandbLogger":
            n_workers = min(len(tasks), os.cpu_count() or 1)
            with Pool(processes=n_workers) as pool:
                saved_paths = list(
                    tqdm(
                        pool.imap(_render_histogram_worker, tasks),
                        total=len(tasks),
                        desc=desc,
                        disable=not self.verbose,
                        dynamic_ncols=True,
                    )
                )
            for fn in saved_paths:
                if fn is not None:
                    self.log_info(f"Saved plot to {fn}")
        else:
            for task in tqdm(
                tasks,
                desc=desc,
                disable=not self.verbose,
                dynamic_ncols=True,
            ):
                fn = _render_histogram_worker(task)
                if fn is not None:
                    self.log_info(f"Saved plot to {fn}")