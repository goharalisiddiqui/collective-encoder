"""Static visualization and diagnostic tools for Optuna studies in collective_encoder."""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

_log = logging.getLogger(__name__)


def study_to_dataframe(study: optuna.Study) -> pd.DataFrame:
    """Converts all completed trials of an Optuna study into a clean pandas DataFrame."""
    rows = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {
            "trial_number": trial.number,
            "state": trial.state.name,
            "duration": trial.duration.total_seconds() if trial.duration else 0.0,
        }
        # Hyperparameters
        for k, v in trial.params.items():
            row[f"param_{k}"] = v

        # User attributes / Metrics
        for k, v in trial.user_attrs.items():
            if isinstance(v, (int, float, str, bool)):
                row[f"attr_{k}"] = v

        # Primary values
        if len(study.directions) == 1:
            try:
                if isinstance(trial.value, (int, float)):
                    row["value"] = trial.value
            except Exception:
                pass
        if trial.values is not None:
            for i, val in enumerate(trial.values):
                row[f"objective_{i}"] = val

        rows.append(row)

    return pd.DataFrame(rows)


def _resolve_metric_columns(df: pd.DataFrame, metrics: Optional[Union[str, List[str]]]) -> List[str]:
    """Resolves explicitly requested metric names to matching numeric columns in the study DataFrame."""
    if metrics is None:
        if "value" in df.columns:
            return ["value"]
        if "attr_val_loss" in df.columns:
            return ["attr_val_loss"]
        return [c for c in df.columns if c.startswith("attr_") and pd.api.types.is_numeric_dtype(df[c])][:1]

    if isinstance(metrics, str):
        metrics_list = [metrics]
    else:
        metrics_list = list(metrics)

    matched_cols = []
    for m in metrics_list:
        m_str = str(m).strip()
        # 1. Exact match in df
        if m_str in df.columns and pd.api.types.is_numeric_dtype(df[m_str]):
            matched_cols.append(m_str)
            continue

        # 2. attr_{m_str} or attr_test_{m_str}
        cand1 = f"attr_{m_str}"
        cand2 = f"attr_test_{m_str}"
        if cand1 in df.columns and pd.api.types.is_numeric_dtype(df[cand1]):
            matched_cols.append(cand1)
            continue
        if cand2 in df.columns and pd.api.types.is_numeric_dtype(df[cand2]):
            matched_cols.append(cand2)
            continue

        # 3. Normalized / fuzzy match
        norm_target = m_str.lower().replace("_", "").replace("-", "")
        found = False
        for col in df.columns:
            if col.startswith("attr_") and pd.api.types.is_numeric_dtype(df[col]):
                norm_col = col.replace("attr_test_", "").replace("attr_", "").lower().replace("_", "").replace("-", "")
                if norm_col == norm_target:
                    matched_cols.append(col)
                    found = True
                    break
        if found:
            continue

        # 4. If m_str is val_loss and value is in df
        if m_str == "val_loss" and "value" in df.columns:
            matched_cols.append("value")
            continue

    # Remove duplicates preserving order
    seen = set()
    result = []
    for c in matched_cols:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result


def plot_latent_elbow_static(df: pd.DataFrame, output_dir: str) -> Optional[str]:
    """Plots Latent Space Dimension vs. Loss elbow curve."""
    if "param_latent_dim" not in df.columns:
        return None

    y_col = "value" if "value" in df.columns else ("attr_val_loss" if "attr_val_loss" in df.columns else None)
    if y_col is None:
        return None

    plt.figure(figsize=(8, 5.5))
    x = df["param_latent_dim"].values
    y = df[y_col].values

    # Scatter of all individual trials
    plt.scatter(x, y, color="steelblue", alpha=0.7, s=60, edgecolors="k", linewidths=0.5, label="Trials", zorder=4)

    # Best and mean curve per latent dimension
    unique_lds = sorted(df["param_latent_dim"].unique())
    min_vals = [df[df["param_latent_dim"] == ld][y_col].min() for ld in unique_lds]
    mean_vals = [df[df["param_latent_dim"] == ld][y_col].mean() for ld in unique_lds]

    plt.plot(unique_lds, min_vals, "r-o", linewidth=2.2, label="Best (Minimum Loss)", zorder=5)
    plt.plot(unique_lds, mean_vals, "b--s", linewidth=1.5, alpha=0.7, label="Mean Loss", zorder=4)

    plt.xlabel("Latent Space Dimension (CV Bottleneck)", fontsize=12)
    plt.ylabel("Validation Loss", fontsize=12)
    plt.title("Latent Bottleneck Dimension vs. Representation Fidelity", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=10)
    plt.xticks(unique_lds)
    plt.tight_layout()

    path = os.path.join(output_dir, "latent_elbow.png")
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_1d_trends_static(df: pd.DataFrame, param_names: List[str], metric_name: str, output_dir: str) -> List[str]:
    """Generates 1D trend curves for each hyperparameter against the target metric."""
    paths = []
    if metric_name not in df.columns or len(df) == 0:
        return paths

    clean_metric_name = metric_name.replace("attr_test_", "").replace("attr_", "")
    for p in param_names:
        p_col = f"param_{p}"
        if p_col not in df.columns:
            continue

        plt.figure(figsize=(7.5, 5))
        x = df[p_col]
        y = df[metric_name]

        # Check if parameter is numeric
        if pd.api.types.is_numeric_dtype(x):
            unique_x = sorted(x.unique())
            if len(unique_x) > 1:
                min_y = [df[df[p_col] == ux][metric_name].min() for ux in unique_x]
                plt.plot(unique_x, min_y, "r--", alpha=0.8, linewidth=1.8, label="Best Curve")
            plt.scatter(x, y, color="darkorange", alpha=0.8, s=50, edgecolors="k", zorder=4, label="Trials")
            if min(unique_x) > 0 and max(unique_x) / (min(unique_x) + 1e-12) > 50:
                plt.xscale("log")
        else:
            plt.scatter(x.astype(str), y, color="darkorange", alpha=0.8, s=50, edgecolors="k", zorder=4)

        plt.xlabel(p, fontsize=12)
        plt.ylabel(clean_metric_name, fontsize=12)
        plt.title(f"Trend: {clean_metric_name} vs. {p}", fontsize=13, fontweight="bold")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()

        out_name = f"trend_{p}_vs_{clean_metric_name}.png"
        path = os.path.join(output_dir, out_name)
        plt.savefig(path, dpi=300)
        plt.close()
        paths.append(path)

    return paths


def plot_2d_heatmaps_static(df: pd.DataFrame, param_names: List[str], metric_name: str, output_dir: str) -> List[str]:
    """Generates 2D heatmap matrices for all pairs of discrete hyperparameters."""
    paths = []
    if metric_name not in df.columns or len(param_names) < 2 or len(df) == 0:
        return paths

    clean_metric_name = metric_name.replace("attr_test_", "").replace("attr_", "")
    for i in range(len(param_names)):
        for j in range(i + 1, len(param_names)):
            p1 = param_names[i]
            p2 = param_names[j]
            p1_col = f"param_{p1}"
            p2_col = f"param_{p2}"
            if p1_col not in df.columns or p2_col not in df.columns:
                continue

            try:
                pivot = df.pivot_table(index=p1_col, columns=p2_col, values=metric_name, aggfunc="min")
                if pivot.shape[0] < 2 or pivot.shape[1] < 2:
                    continue

                plt.figure(figsize=(8, 6))
                im = plt.imshow(pivot.values, cmap="viridis_r", aspect="auto", origin="lower")
                plt.colorbar(im, label=clean_metric_name)

                plt.xticks(range(len(pivot.columns)), pivot.columns, fontsize=10)
                plt.yticks(range(len(pivot.index)), pivot.index, fontsize=10)
                plt.xlabel(p2, fontsize=12)
                plt.ylabel(p1, fontsize=12)
                plt.title(f"Heatmap: {clean_metric_name} ({p1} vs. {p2})", fontsize=13, fontweight="bold")

                # Annotate values inside cells
                for row_idx in range(pivot.shape[0]):
                    for col_idx in range(pivot.shape[1]):
                        val = pivot.values[row_idx, col_idx]
                        if not np.isnan(val):
                            plt.text(col_idx, row_idx, f"{val:.4g}", ha="center", va="center", color="white" if val > pivot.values.mean() else "black", fontsize=9, fontweight="bold")

                plt.tight_layout()
                out_name = f"heatmap_{p1}_vs_{p2}_{clean_metric_name}.png"
                path = os.path.join(output_dir, out_name)
                plt.savefig(path, dpi=300)
                plt.close()
                paths.append(path)
            except Exception as e:
                _log.debug(f"Could not generate heatmap for {p1} vs {p2}: {e}")

    return paths


def plot_multi_series_interaction_static(df: pd.DataFrame, param1: str, param2: str, metric_name: str, output_dir: str) -> Optional[str]:
    """Plots metric vs param1 with separate colored series for each value of param2."""
    p1_col = f"param_{param1}"
    p2_col = f"param_{param2}"
    if p1_col not in df.columns or p2_col not in df.columns or metric_name not in df.columns:
        return None

    clean_metric_name = metric_name.replace("attr_test_", "").replace("attr_", "")
    try:
        plt.figure(figsize=(8.5, 5.5))
        unique_p2 = sorted(df[p2_col].unique())
        cmap = plt.get_cmap("tab10")

        for idx, val2 in enumerate(unique_p2):
            sub_df = df[df[p2_col] == val2].sort_values(by=p1_col)
            plt.plot(sub_df[p1_col], sub_df[metric_name], marker="o", linewidth=2, color=cmap(idx % 10), label=f"{param2}={val2}")

        plt.xlabel(param1, fontsize=12)
        plt.ylabel(clean_metric_name, fontsize=12)
        plt.title(f"Interaction: {clean_metric_name} vs. {param1} by {param2}", fontsize=13, fontweight="bold")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(frameon=True, fontsize=9)
        plt.tight_layout()

        out_name = f"interaction_{param1}_by_{param2}_{clean_metric_name}.png"
        path = os.path.join(output_dir, out_name)
        plt.savefig(path, dpi=300)
        plt.close()
        return path
    except Exception as e:
        _log.debug(f"Could not plot interaction for {param1} by {param2}: {e}")
        return None


def plot_pareto_front_static(
    study: optuna.Study,
    df: pd.DataFrame,
    output_dir: str,
    metrics: Optional[List[str]] = None
) -> Optional[str]:
    """Plots Pareto front for 2-objective optimization studies."""
    if len(study.directions) != 2 or len(df) == 0:
        return None

    obj0_col = "objective_0" if "objective_0" in df.columns else None
    obj1_col = "objective_1" if "objective_1" in df.columns else None
    if not obj0_col or not obj1_col:
        return None

    try:
        plt.figure(figsize=(8, 6))
        plt.scatter(df[obj0_col], df[obj1_col], color="steelblue", alpha=0.7, s=60, edgecolors="k", label="All Trials", zorder=3)

        best_trials = study.best_trials
        if best_trials:
            pareto_x = [t.values[0] for t in best_trials]
            pareto_y = [t.values[1] for t in best_trials]
            plt.scatter(pareto_x, pareto_y, color="crimson", s=90, marker="*", edgecolors="black", label="Pareto Front", zorder=5)

        obj0_label = "Objective 0"
        obj1_label = "Objective 1"
        if metrics and len(metrics) >= 2:
            obj0_label = str(metrics[0])
            obj1_label = str(metrics[1])
        else:
            if "attr_val_loss" in df.columns:
                obj0_label = "Validation Loss"
            for col in df.columns:
                if "beta_score" in col or "mig_score" in col or "factor_score" in col or "LC_max" in col:
                    obj1_label = col.replace("attr_test_", "").replace("attr_", "")

        plt.xlabel(obj0_label, fontsize=12)
        plt.ylabel(obj1_label, fontsize=12)
        plt.title(f"Pareto Front: {obj0_label} vs. {obj1_label}", fontsize=13, fontweight="bold")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(frameon=True, fontsize=10)
        plt.tight_layout()

        path = os.path.join(output_dir, "pareto_front.png")
        plt.savefig(path, dpi=300)
        plt.close()
        return path
    except Exception as e:
        _log.debug(f"Could not plot Pareto front: {e}")
        return None


def export_study_csv_and_markdown(study: optuna.Study, df: pd.DataFrame, output_dir: str) -> Tuple[str, str]:
    """Exports a full CSV table and a formatted Markdown summary report."""
    csv_path = os.path.join(output_dir, "study_results.csv")
    df.to_csv(csv_path, index=False)

    md_path = os.path.join(output_dir, "summary.md")
    with open(md_path, "w") as f:
        f.write(f"# Study Summary: {study.study_name}\n\n")
        f.write(f"- **Total Trials**: {len(study.trials)}\n")
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        f.write(f"- **Complete Trials**: {len(complete_trials)}\n")

        if len(study.directions) == 1 and complete_trials:
            best_t = study.best_trial
            f.write(f"- **Best Trial ID**: #{best_t.number}\n")
            f.write(f"- **Best Metric Value**: `{best_t.value:.6f}`\n\n")
            f.write("### Best Hyperparameters:\n\n")
            for k, v in best_t.params.items():
                f.write(f"- **`{k}`**: `{v}`\n")
        elif len(study.directions) > 1 and complete_trials:
            f.write("### Best Pareto Front Trials:\n\n")
            for bt in study.best_trials:
                f.write(f"- **Trial #{bt.number}**: Objective Values=`{bt.values}` | Parameters: `{bt.params}`\n")

        f.write("\n### All Completed Trials:\n\n")
        # Format preview table in markdown
        preview_df = df.head(50)
        f.write(preview_df.to_markdown(index=False) if hasattr(preview_df, "to_markdown") else preview_df.to_string())
        f.write("\n")

    return csv_path, md_path


def export_study_visualizations(
    study: optuna.Study,
    output_dir: str,
    metrics: Optional[Union[str, List[str]]] = None
) -> List[str]:
    """Generates static publication-quality PNG plots, CSV tables, and Markdown reports for explicitly given metrics."""
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []

    df = study_to_dataframe(study)
    if len(df) == 0:
        _log.warning("No completed trials found in study. Skipping visualization export.")
        return generated_files

    # 1. Export CSV & Markdown reports (contains all metrics, attributes, and hyperparameters)
    csv_path, md_path = export_study_csv_and_markdown(study, df, output_dir)
    generated_files.extend([csv_path, md_path])

    # Extract parameter and explicitly requested metric columns
    param_names = [col.replace("param_", "") for col in df.columns if col.startswith("param_")]
    target_metric_cols = _resolve_metric_columns(df, metrics)
    primary_metric = target_metric_cols[0] if target_metric_cols else ("value" if "value" in df.columns else None)

    # 2. Latent Bottleneck Elbow curve (if latent dimension parameter is swept)
    elbow_path = plot_latent_elbow_static(df, output_dir)
    if elbow_path:
        generated_files.append(elbow_path)

    # 3. Pareto front plot for multi-objective studies
    metrics_list = [metrics] if isinstance(metrics, str) else (list(metrics) if metrics is not None else None)
    pareto_path = plot_pareto_front_static(study, df, output_dir, metrics=metrics_list)
    if pareto_path:
        generated_files.append(pareto_path)

    # 4. 1D Trends ONLY for explicitly specified metric(s)
    for m in target_metric_cols:
        trend_paths = plot_1d_trends_static(df, param_names, m, output_dir)
        generated_files.extend(trend_paths)

    # 5. 2D Heatmaps ONLY for explicitly specified metric(s)
    if len(param_names) >= 2:
        for m in target_metric_cols:
            heatmap_paths = plot_2d_heatmaps_static(df, param_names, m, output_dir)
            generated_files.extend(heatmap_paths)

    # 6. Multi-series interaction plots ONLY for primary explicit metric
    if len(param_names) >= 2 and primary_metric:
        inter_path = plot_multi_series_interaction_static(df, param_names[0], param_names[1], primary_metric, output_dir)
        if inter_path:
            generated_files.append(inter_path)

    # 7. Optuna Interactive Visualizations (as standalone HTML backups)
    try:
        from optuna.visualization import plot_param_importances
        fig = plot_param_importances(study)
        p = os.path.join(output_dir, "param_importances.html")
        fig.write_html(p)
        generated_files.append(p)
    except Exception:
        pass

    try:
        from optuna.visualization import plot_parallel_coordinate
        fig = plot_parallel_coordinate(study)
        p = os.path.join(output_dir, "parallel_coordinate.html")
        fig.write_html(p)
        generated_files.append(p)
    except Exception:
        pass

    _log.info(f"Generated {len(generated_files)} diagnostic plots and reports in {output_dir}")
    return generated_files


def launch_dashboard(storage: str, study_name: Optional[str] = None, port: int = 8080) -> None:
    """Launches the interactive optuna-dashboard web server (optional)."""
    import subprocess
    cmd = ["optuna-dashboard", storage, "--port", str(port)]
    _log.info(f"Starting optuna-dashboard at http://localhost:{port} with storage {storage}")
    print(f"\n==> Optuna Dashboard running at http://localhost:{port} (Ctrl+C to stop)\n")
    subprocess.run(cmd)
