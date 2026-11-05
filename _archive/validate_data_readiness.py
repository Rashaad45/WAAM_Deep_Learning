"""Fail-fast readiness gate: checks whether this repository currently has what
`dataset_graph.py` and `train_evaluate.py` need to run a scientifically valid
training job, WITHOUT fabricating anything to make the check pass.

This script only reads raw_trial_manifest.csv, zone_labels.csv, and the files
they point to. It never writes labels or invents values. Run it before
feature preparation / training (run_training.bat does this automatically) and
read the printed report: a "NOT READY" verdict means the missing items listed
are real gaps in the source data, not a bug to patch around.
"""
from pathlib import Path
import argparse
import sys
import pandas as pd

PIPELINE_DIR = Path(__file__).resolve().parent


def _filled_mask(series: pd.Series) -> pd.Series:
    """Return true only for non-null, non-blank values (including literal NaN text)."""
    text = series.astype("string").str.strip()
    return (series.notna() & text.ne("") & text.str.casefold().ne("nan")).fillna(False)


def _resolve(path_str: str, base: Path) -> Path:
    path = Path(str(path_str))
    return path if path.is_absolute() else (base / path)


def check_manifest(manifest_path: Path, base: Path) -> tuple[list[str], list[str], pd.DataFrame | None]:
    errors, warnings = [], []
    if not manifest_path.exists():
        errors.append(f"Manifest not found: {manifest_path}")
        return errors, warnings, None
    frame = pd.read_csv(manifest_path)
    required = {"trial_id", "material_type", "spindle_speed_rpm", "depth_of_cut_mm", "force_csv", "acceleration_csv"}
    missing_cols = required - set(frame.columns)
    if missing_cols:
        errors.append(f"{manifest_path.name} is missing required columns: {sorted(missing_cols)}")
        return errors, warnings, frame

    n = len(frame)
    with_material = int(_filled_mask(frame["material_type"]).sum())
    with_force = int(_filled_mask(frame["force_csv"]).sum())
    with_cut_start = int(_filled_mask(frame["cut_start_time_s"]).sum()) if "cut_start_time_s" in frame else 0

    warnings.append(f"{n} trial rows; material_type filled for {with_material}/{n}; "
                     f"force_csv filled for {with_force}/{n}; cut_start_time_s filled for {with_cut_start}/{n}.")
    if with_material < n:
        warnings.append(f"{n - with_material} trial(s) have no verified material_type (WAAM vs wrought) -- "
                         "they cannot be used for any material-conditioned analysis until confirmed.")
    if with_cut_start == 0:
        errors.append("cut_start_time_s is blank for every trial: dataset_graph.py's prepare_raw_trial_cache "
                       "cannot locate the cutting window inside any force/acceleration recording without it. "
                       "This must come from an acquisition log, not be guessed.")
    elif with_cut_start < n:
        errors.append(f"cut_start_time_s is filled for only {with_cut_start}/{n} trials; remove unresolved trials from the training cohort or supply verified cutting-window starts.")
    if with_force < n:
        errors.append(f"force_csv is filled for only {with_force}/{n} trials; remove trials without a verified force recording from the training cohort.")

    missing_files = []
    for col in ("acceleration_csv", "force_csv"):
        if col not in frame:
            continue
        for _, row in frame.iterrows():
            value = str(row[col]).strip()
            if not value or value == "nan":
                continue
            if not _resolve(value, base).exists():
                missing_files.append(f"{row.get('trial_id', '?')}: {col}={value}")
    if missing_files:
        errors.append(f"{len(missing_files)} referenced source file(s) do not exist on disk, e.g. {missing_files[:5]}")

    return errors, warnings, frame


def check_labels(labels_path: Path) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    if not labels_path.exists():
        errors.append(f"Labels file not found: {labels_path}")
        return errors, warnings
    frame = pd.read_csv(labels_path)
    required = {"trial_id", "zone_id", "chatter_label", "measured_Ra_um"}
    missing_cols = required - set(frame.columns)
    if missing_cols:
        errors.append(f"{labels_path.name} is missing required columns: {sorted(missing_cols)}")
        return errors, warnings

    n = len(frame)
    chatter_filled = frame["chatter_label"].notna().sum()
    ra_filled = frame["measured_Ra_um"].notna().sum()
    warnings.append(f"{n} zone-label rows; chatter_label filled for {chatter_filled}/{n}; "
                     f"measured_Ra_um filled for {ra_filled}/{n}.")
    if "ra_label_status" in frame.columns:
        imputed = frame["ra_label_status"].astype("string").str.contains("IMPUTED", case=False, na=False)
        imputed_count = int((imputed & frame["measured_Ra_um"].notna()).sum())
        if imputed_count:
            warnings.append(f"{imputed_count} Ra targets are literature-imputed material means, not observed zone measurements.")

    if chatter_filled == 0 and ra_filled == 0:
        errors.append(
            "zone_labels.csv has ZERO chatter_label and ZERO measured_Ra_um entries. There is no supervised "
            "target for either task. Per DATA_READINESS_REPORT.md, the source paper and raw WAAM/ archive only "
            "contain material-level aggregate Ra (two numbers) and a qualitative chatter discussion -- neither "
            "is a valid per-trial/per-zone label, so none was copied in. Training cannot proceed until real "
            "zone-level chatter annotations and/or Ra measurements are supplied."
        )
    else:
        if chatter_filled == 0:
            warnings.append("No chatter_label values are present; only the roughness-regression task could run "
                             "(train_evaluate.py --ablations roughness_only).")
        if ra_filled == 0:
            warnings.append("No measured_Ra_um values are present; only the chatter-classification task could run "
                             "(train_evaluate.py --ablations classification_only).")
        if 0 < chatter_filled < n:
            classes = frame["chatter_label"].dropna().unique()
            if len(classes) < 2:
                errors.append(f"chatter_label has only {len(classes)} distinct class(es) ({list(classes)}); "
                               "binary classification requires both 0 and 1 to appear at least once.")

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PIPELINE_DIR / "raw_trial_manifest.csv")
    parser.add_argument("--labels", type=Path, default=PIPELINE_DIR / "zone_labels.csv")
    parser.add_argument("--base", type=Path, default=PIPELINE_DIR,
                         help="Directory that relative paths inside the manifest are resolved against")
    args = parser.parse_args()

    print("=" * 72)
    print("WAAM GNN pipeline -- data readiness check")
    print("=" * 72)

    manifest_errors, manifest_warnings, _ = check_manifest(args.manifest, args.base)
    label_errors, label_warnings = check_labels(args.labels)

    print(f"\n[manifest] {args.manifest}")
    for w in manifest_warnings:
        print(f"  - {w}")
    for e in manifest_errors:
        print(f"  ERROR: {e}")

    print(f"\n[labels] {args.labels}")
    for w in label_warnings:
        print(f"  - {w}")
    for e in label_errors:
        print(f"  ERROR: {e}")

    all_errors = manifest_errors + label_errors
    print("\n" + "=" * 72)
    if all_errors:
        print(f"NOT READY -- {len(all_errors)} blocking issue(s) found. Training must not proceed.")
        print("See DATA_READINESS_REPORT.md for exactly what source records are needed to resolve each item.")
        print("=" * 72)
        return 1

    labels_frame = pd.read_csv(args.labels)
    has_imputed_targets = (
        "ra_label_status" in labels_frame.columns
        and labels_frame["ra_label_status"].astype("string").str.contains("IMPUTED", case=False, na=False).any()
    )
    if has_imputed_targets:
        print("PILOT INPUTS READY -- literature-imputed material means are repeated as zone targets. "
              "This is only a code-path pilot, NOT valid per-zone Ra prediction or publishable "
              "predictive performance.")
    else:
        print("READY -- required columns are present and referenced files resolve. This does NOT "
              "certify scientific validity of the DOE/material mapping (see trial_mapping_evidence.csv "
              "for confidence on every assignment).")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
