import csv
import json
from pathlib import Path

import numpy as np


WALK_COLUMNS = [
    "timestamp",
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "mag_x",
    "mag_y",
    "mag_z",
    "external_e",
    "external_n",
    "true_e",
    "true_n",
    "true_heading",
    "is_step",
]

TRAINING_COLUMNS = ["step_frequency", "acc_variance", "true_stride"]


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_stride_model(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_stride_model(path, model):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)


def _parse_float(value, column_name, allow_nan=False):
    if value is None or value == "":
        if allow_nan:
            return np.nan
        raise ValueError(f"Missing value in required column '{column_name}'")
    return float(value)


def load_walk_csv(path):
    path = Path(path)
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} does not contain a CSV header")
        missing = [column for column in WALK_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")

        rows = []
        for row in reader:
            parsed = {}
            for column in WALK_COLUMNS:
                parsed[column] = _parse_float(
                    row.get(column),
                    column,
                    allow_nan=column in {"external_e", "external_n"},
                )
            parsed["is_step"] = bool(int(parsed["is_step"]))
            rows.append(parsed)

    if not rows:
        raise ValueError(f"{path} does not contain data rows")

    timestamp = np.array([row["timestamp"] for row in rows], dtype=float)
    if np.any(np.diff(timestamp) <= 0.0):
        raise ValueError("timestamp values must be strictly increasing")

    acc = np.column_stack([[row[f"acc_{axis}"] for row in rows] for axis in "xyz"])
    gyro = np.column_stack([[row[f"gyro_{axis}"] for row in rows] for axis in "xyz"])
    mag = np.column_stack([[row[f"mag_{axis}"] for row in rows] for axis in "xyz"])
    external_position = np.column_stack(
        [[row["external_e"] for row in rows], [row["external_n"] for row in rows]]
    )
    true_position = np.column_stack(
        [[row["true_e"] for row in rows], [row["true_n"] for row in rows]]
    )

    return {
        "timestamp": timestamp,
        "acc": acc.astype(float),
        "gyro": gyro.astype(float),
        "mag": mag.astype(float),
        "external_position": external_position.astype(float),
        "true_position": true_position.astype(float),
        "true_heading": np.array([row["true_heading"] for row in rows], dtype=float),
        "is_step": np.array([row["is_step"] for row in rows], dtype=bool),
    }


def load_stride_training_csv(path):
    path = Path(path)
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} does not contain a CSV header")
        missing = [column for column in TRAINING_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")

        rows = []
        for row in reader:
            rows.append(
                {
                    column: _parse_float(row.get(column), column)
                    for column in TRAINING_COLUMNS
                }
            )

    if not rows:
        raise ValueError(f"{path} does not contain data rows")

    return {
        "step_frequency": np.array([row["step_frequency"] for row in rows], dtype=float),
        "acc_variance": np.array([row["acc_variance"] for row in rows], dtype=float),
        "true_stride": np.array([row["true_stride"] for row in rows], dtype=float),
    }


def _format_optional(value):
    if np.isfinite(value):
        return f"{value:.8f}"
    return ""


def save_trajectory_csv(path, data, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    states = result["states"]
    step_mask = np.zeros(len(data["timestamp"]), dtype=bool)
    for record in result["step_records"]:
        step_mask[record["index"]] = True

    fieldnames = [
        "timestamp",
        "est_e",
        "est_n",
        "est_heading",
        "est_gyro_bias",
        "external_e",
        "external_n",
        "true_e",
        "true_n",
        "true_heading",
        "detected_step",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, timestamp in enumerate(data["timestamp"]):
            writer.writerow(
                {
                    "timestamp": f"{timestamp:.4f}",
                    "est_e": f"{states[i, 0]:.8f}",
                    "est_n": f"{states[i, 1]:.8f}",
                    "est_heading": f"{states[i, 2]:.8f}",
                    "est_gyro_bias": f"{states[i, 3]:.8f}",
                    "external_e": _format_optional(data["external_position"][i, 0]),
                    "external_n": _format_optional(data["external_position"][i, 1]),
                    "true_e": f"{data['true_position'][i, 0]:.8f}",
                    "true_n": f"{data['true_position'][i, 1]:.8f}",
                    "true_heading": f"{data['true_heading'][i]:.8f}",
                    "detected_step": int(step_mask[i]),
                }
            )


def save_metrics_json(path, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
