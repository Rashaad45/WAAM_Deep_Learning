"""Materials-style vector plots for the manuscript's zone-level LOTO results."""
from pathlib import Path
import argparse
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.manifold import TSNE
from utils_signal import load_numeric_csv, welch_psd
from config import CONFIG


def set_materials_style() -> None:
    sns.set_theme(style="ticks", context="paper", font="DejaVu Sans", palette=["#0072B2", "#D55E00"])
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 8})


def save_signal_comparison(signals: dict[str, Path], output: Path, fs: float = 25_000.0) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for material, path in signals.items():
        values, _ = load_numeric_csv(path)
        freq, psd = welch_psd(values[:, -1], fs)
        ax.semilogy(freq, psd, label=material)
    for mode in (3500, 4300):
        ax.axvline(mode, color="0.35", ls="--", lw=0.8)
        ax.text(mode + 35, ax.get_ylim()[0] * 2, f"{mode / 1000:.1f} kHz", rotation=90, va="bottom", fontsize=7)
    ax.set(xlabel="Frequency (Hz)", ylabel="PSD (unit²/Hz)", xlim=(0, 8000), title="Welch power spectra")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "psd_comparison.svg")
    fig.savefig(output / "psd_comparison.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, len(signals), figsize=(6.5, 3.2), sharey=True, squeeze=False)
    for ax, (material, path) in zip(axes[0], signals.items()):
        values, _ = load_numeric_csv(path)
        f, t, z = signal.stft(values[:, -1], fs, nperseg=1024, noverlap=768)
        mask = f <= 8000
        mesh = ax.pcolormesh(t, f[mask], 20 * np.log10(np.maximum(np.abs(z[mask]), 1e-12)), shading="auto", cmap="magma")
        ax.set(title=material, xlabel="Time (s)", ylim=(0, 8000))
    axes[0, 0].set_ylabel("Frequency (Hz)")
    fig.colorbar(mesh, ax=axes.ravel().tolist(), label="Magnitude (dB)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(output / "stft_comparison.svg")
    fig.savefig(output / "stft_comparison.pdf")
    plt.close(fig)


def collapse_seed_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    grouping = [c for c in ("trial_id", "material_type", "zone_id", "model", "variant", "y_class", "y_reg") if c in frame.columns]
    aggregations = {"probability": ("probability", "mean"), "predicted_Ra_um": ("predicted_Ra_um", "mean"),
                    "embedding": ("embedding", "first")}
    for column in ("kinematic_attention_mean", "thermal_attention_mean"):
        if column in frame.columns:
            aggregations[column] = (column, "mean")
    return frame.groupby(grouping, dropna=False, as_index=False).agg(**aggregations)


def save_edge_attention_plot(predictions: pd.DataFrame, output: Path) -> None:
    names = ("kinematic_attention_mean", "thermal_attention_mean")
    if not all(name in predictions and predictions[name].notna().any() for name in names):
        return
    by_trial = predictions.groupby(["trial_id", "material_type"], as_index=False)[list(names)].mean()
    long = by_trial.melt(id_vars=["trial_id", "material_type"], value_vars=list(names), var_name="edge_type", value_name="attention")
    long["edge_type"] = long.edge_type.map({names[0]: "Kinematic", names[1]: "Thermal"})
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    sns.barplot(data=long, x="material_type", y="attention", hue="edge_type", errorbar="sd", ax=ax)
    ax.set(xlabel="Material", ylabel="Mean GAT attention", title="Attention by physical edge type")
    ax.legend(frameon=False, title="Edge type")
    fig.tight_layout()
    fig.savefig(output / "gat_edge_attention.svg")
    fig.savefig(output / "gat_edge_attention.pdf")
    plt.close(fig)


def save_classification_plots(predictions: pd.DataFrame, output: Path) -> None:
    valid = predictions[np.isfinite(predictions.y_class)].copy()
    y = valid.y_class.astype(int).to_numpy()
    p = valid.probability.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))
    if len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, lw=1.8, color="#0072B2", label="LOTO out-of-fold")
        axes[0].legend(frameon=False)
    axes[0].plot([0, 1], [0, 1], "--", color="0.5", lw=0.8)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="Instability ROC", xlim=(0, 1), ylim=(0, 1))
    matrix = confusion_matrix(y, p >= 0.5, labels=[0, 1])
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[1], xticklabels=["Stable", "Unstable"], yticklabels=["Stable", "Unstable"])
    axes[1].set(xlabel="Predicted", ylabel="Observed", title="Confusion matrix")
    fig.tight_layout()
    fig.savefig(output / "instability_roc_confusion.svg")
    fig.savefig(output / "instability_roc_confusion.pdf")
    plt.close(fig)


def save_roughness_plots(predictions: pd.DataFrame, output: Path) -> None:
    valid = predictions[np.isfinite(predictions.y_reg)].copy()
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    for material, group in valid.groupby("material_type"):
        ax.scatter(group.y_reg, group.predicted_Ra_um, label=material, alpha=0.8, s=24, edgecolor="white", linewidth=0.4)
    low = float(min(valid.y_reg.min(), valid.predicted_Ra_um.min()))
    high = float(max(valid.y_reg.max(), valid.predicted_Ra_um.max()))
    ax.plot([low, high], [low, high], "--", color="0.35", lw=0.9)
    ax.set(xlabel="Measured Ra (µm)", ylabel="Predicted Ra (µm)", title="Zone-level out-of-fold parity")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "roughness_parity.svg")
    fig.savefig(output / "roughness_parity.pdf")
    plt.close(fig)
    selected = valid[valid.material_type.str.lower().str.contains("waam")].sort_values(["trial_id", "zone_id"])
    if not selected.empty:
        trial = selected.trial_id.iloc[0]
        sample = selected[selected.trial_id == trial]
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        ax.plot(sample.zone_id, sample.y_reg, "o-", label="Measured")
        ax.plot(sample.zone_id, sample.predicted_Ra_um, "s--", label="Predicted")
        ax.set(xlabel="Cutting zone (5 mm increments)", ylabel="Ra (µm)", title=f"Along-wall roughness: trial {trial}")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output / "roughness_along_wall.svg")
        fig.savefig(output / "roughness_along_wall.pdf")
        plt.close(fig)


def save_tsne(predictions: pd.DataFrame, output: Path) -> None:
    if "embedding" not in predictions:
        raise ValueError("Prediction table contains no model embeddings")
    vectors = []
    for value in predictions.embedding:
        vectors.append(np.asarray(ast.literal_eval(value) if isinstance(value, str) else value, dtype=float))
    embeddings = np.stack(vectors)
    if len(embeddings) < 3:
        raise ValueError("t-SNE requires at least three zone embeddings")
    perplexity = min(30.0, max(2.0, (len(embeddings) - 1) / 3))
    coords = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=CONFIG.train.seeds[0]).fit_transform(embeddings)
    labels = predictions.y_class.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(5, 4))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="coolwarm", s=22, alpha=0.85)
    ax.set(xlabel="t-SNE 1", ylabel="t-SNE 2", title="Zone graph embeddings")
    fig.colorbar(scatter, ax=ax, label="Instability label")
    fig.tight_layout()
    fig.savefig(output / "graph_tsne.svg")
    fig.savefig(output / "graph_tsne.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CONFIG.output_dir)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--waam-signal", type=Path)
    parser.add_argument("--wrought-signal", type=Path)
    parser.add_argument("--sampling-rate", type=float, default=25_000.0)
    parser.add_argument("--model", default="gat")
    parser.add_argument("--variant", default="full")
    args = parser.parse_args()
    set_materials_style()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.waam_signal and args.wrought_signal:
        save_signal_comparison({"WAAM Al 5356": args.waam_signal, "Wrought AA 5052": args.wrought_signal}, args.output, args.sampling_rate)
    if args.predictions:
        all_rows = pd.read_csv(args.predictions)
        selected = all_rows[(all_rows.model == args.model) & (all_rows.variant == args.variant)]
        rows = collapse_seed_predictions(selected)
        save_classification_plots(rows, args.output)
        save_roughness_plots(rows, args.output)
        save_edge_attention_plot(rows, args.output)
        save_tsne(rows, args.output)


if __name__ == "__main__":
    main()
