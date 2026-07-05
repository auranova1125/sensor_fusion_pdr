import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sensor_fusion_pdr_mpl")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .preprocessing import wrap_angle
from .stride_model import predict_stride


def _ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def plot_trajectory(data, result, path):
    _ensure_parent(path)
    states = result["states"]
    external = data["external_position"]
    ext_mask = np.all(np.isfinite(external), axis=1)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(data["true_position"][:, 0], data["true_position"][:, 1], label="true")
    ax.plot(states[:, 0], states[:, 1], label="estimated")
    if ext_mask.any():
        ax.scatter(
            external[ext_mask, 0],
            external[ext_mask, 1],
            s=18,
            label="external obs",
            alpha=0.7,
        )
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_title("East-North Trajectory")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_step_detection(data, processed, result, path):
    _ensure_parent(path)
    timestamp = data["timestamp"]
    signal = processed["smooth_linear_acc"]
    step_indices = [record["index"] for record in result["step_records"]]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timestamp, signal, label="smoothed acceleration magnitude")
    if step_indices:
        ax.scatter(
            timestamp[step_indices],
            signal[step_indices],
            color="tab:red",
            s=24,
            label="detected steps",
            zorder=3,
        )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Acceleration residual [m/s^2]")
    ax.set_title("Step Detection")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_heading_and_bias(data, processed, result, config, path):
    _ensure_parent(path)
    timestamp = data["timestamp"]
    states = result["states"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(timestamp, np.rad2deg(data["true_heading"]), label="true")
    axes[0].plot(timestamp, np.rad2deg(processed["gyro_heading"]), label="gyro integrated")
    axes[0].plot(timestamp, np.rad2deg(processed["mag_heading"]), label="mag heading", alpha=0.55)
    axes[0].plot(timestamp, np.rad2deg(states[:, 2]), label="kalman")
    axes[0].set_ylabel("Heading [deg]")
    axes[0].set_title("Heading Estimates")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=2)

    true_bias = config.get("synthetic", {}).get("gyro_bias_z_rad_s")
    if true_bias is not None:
        axes[1].axhline(float(true_bias), color="tab:green", label="true bias")
    axes[1].plot(timestamp, states[:, 3], label="estimated bias")
    axes[1].axhline(processed["gyro_bias_z"], color="tab:orange", linestyle="--", label="initial estimate")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Gyro bias [rad/s]")
    axes[1].set_title("Gyroscope Bias")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_stride_features(result, path):
    _ensure_parent(path)
    records = result["step_records"]
    step_number = np.array([record["step_number"] for record in records], dtype=float)
    estimated_stride = np.array([record["estimated_stride"] for record in records], dtype=float)
    true_stride = np.array([record.get("true_stride", np.nan) for record in records], dtype=float)
    frequency = np.array([record["frequency"] for record in records], dtype=float)
    acc_variance = np.array([record["acc_variance"] for record in records], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(step_number, estimated_stride, marker="o", label="estimated stride")
    if np.isfinite(true_stride).any():
        axes[0].plot(step_number, true_stride, marker="x", label="true stride")
    axes[0].set_ylabel("Stride [m]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(step_number, frequency, marker="o", color="tab:purple")
    axes[1].set_ylabel("Frequency [Hz]")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(step_number, acc_variance, marker="o", color="tab:brown")
    axes[2].set_xlabel("Detected step number")
    axes[2].set_ylabel("Acc variance")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Stride Features and Estimates")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_stride_regression(training_data, model, path):
    _ensure_parent(path)
    prediction = predict_stride(
        model, training_data["step_frequency"], training_data["acc_variance"]
    )
    true_stride = training_data["true_stride"]
    lower = min(float(np.min(true_stride)), float(np.min(prediction))) - 0.03
    upper = max(float(np.max(true_stride)), float(np.max(prediction))) + 0.03

    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(
        true_stride,
        prediction,
        c=training_data["step_frequency"],
        cmap="viridis",
        s=28,
        alpha=0.8,
    )
    ax.plot([lower, upper], [lower, upper], color="black", linestyle="--", linewidth=1)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("True stride [m]")
    ax.set_ylabel("Predicted stride [m]")
    ax.set_title("Stride Regression")
    ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Step frequency [Hz]")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
