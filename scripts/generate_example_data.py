import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_io import load_config
from src.preprocessing import wrap_angle


def _step_schedule(rng):
    step_times = []
    step_headings = []
    step_periods = []

    t = 5.65
    for _ in range(10):
        period = float(np.clip(rng.normal(0.75, 0.035), 0.68, 0.84))
        step_times.append(t)
        step_headings.append(0.0)
        step_periods.append(period)
        t += period

    turn1_start = step_times[-1] + 0.45
    turn1_end = turn1_start + 1.7

    t = turn1_end + 0.55
    for _ in range(5):
        period = float(np.clip(rng.normal(0.73, 0.03), 0.66, 0.82))
        step_times.append(t)
        step_headings.append(np.pi / 2.0)
        step_periods.append(period)
        t += period

    turn2_start = step_times[-1] + 0.45
    turn2_end = turn2_start + 1.7

    t = turn2_end + 0.55
    for _ in range(10):
        period = float(np.clip(rng.normal(0.76, 0.035), 0.68, 0.86))
        step_times.append(t)
        step_headings.append(0.0)
        step_periods.append(period)
        t += period

    return {
        "step_times": np.array(step_times, dtype=float),
        "step_headings": np.array(step_headings, dtype=float),
        "step_periods": np.array(step_periods, dtype=float),
        "turn1_start": turn1_start,
        "turn1_end": turn1_end,
        "turn2_start": turn2_start,
        "turn2_end": turn2_end,
        "end_time": step_times[-1] + 1.2,
    }


def _make_heading(timestamp, schedule):
    heading = np.zeros_like(timestamp)
    turn1_mask = (timestamp >= schedule["turn1_start"]) & (
        timestamp <= schedule["turn1_end"]
    )
    heading[timestamp > schedule["turn1_end"]] = np.pi / 2.0
    heading[turn1_mask] = (
        (timestamp[turn1_mask] - schedule["turn1_start"])
        / (schedule["turn1_end"] - schedule["turn1_start"])
        * np.pi
        / 2.0
    )

    turn2_mask = (timestamp >= schedule["turn2_start"]) & (
        timestamp <= schedule["turn2_end"]
    )
    heading[timestamp > schedule["turn2_end"]] = 0.0
    heading[turn2_mask] = (
        (1.0 - (timestamp[turn2_mask] - schedule["turn2_start"])
        / (schedule["turn2_end"] - schedule["turn2_start"]))
        * np.pi
        / 2.0
    )
    return wrap_angle(heading)


def _make_positions(timestamp, schedule, rng):
    key_times = [0.0, 5.0]
    key_positions = [np.zeros(2), np.zeros(2)]
    position = np.zeros(2, dtype=float)
    true_strides = []

    for step_time, heading, period in zip(
        schedule["step_times"],
        schedule["step_headings"],
        schedule["step_periods"],
    ):
        frequency = 1.0 / period
        stride = float(
            np.clip(0.38 + 0.19 * frequency + rng.normal(0.0, 0.018), 0.55, 0.82)
        )
        true_strides.append(stride)
        position = position + stride * np.array([np.sin(heading), np.cos(heading)])
        key_times.append(float(step_time))
        key_positions.append(position.copy())

        if abs(step_time - schedule["step_times"][9]) < 1e-9:
            key_times.extend([schedule["turn1_start"], schedule["turn1_end"]])
            key_positions.extend([position.copy(), position.copy()])
        if abs(step_time - schedule["step_times"][14]) < 1e-9:
            key_times.extend([schedule["turn2_start"], schedule["turn2_end"]])
            key_positions.extend([position.copy(), position.copy()])

    key_times.append(schedule["end_time"])
    key_positions.append(position.copy())

    order = np.argsort(key_times)
    key_times = np.array(key_times, dtype=float)[order]
    key_positions = np.array(key_positions, dtype=float)[order]
    east = np.interp(timestamp, key_times, key_positions[:, 0])
    north = np.interp(timestamp, key_times, key_positions[:, 1])
    return np.column_stack([east, north]), np.array(true_strides, dtype=float)


def _make_acceleration(timestamp, true_heading, schedule, true_strides, config, rng):
    synthetic = config["synthetic"]
    bias = np.array(synthetic["acc_bias_mps2"], dtype=float)
    noise_std = float(synthetic["acc_noise_std_mps2"])

    impact = np.zeros_like(timestamp)
    horizontal = np.zeros_like(timestamp)
    for step_time, stride in zip(schedule["step_times"], true_strides):
        amplitude = 1.55 + 1.05 * (stride - 0.6) + rng.normal(0.0, 0.08)
        sigma = 0.052
        impact += amplitude * np.exp(-0.5 * ((timestamp - step_time) / sigma) ** 2)
        horizontal += 0.18 * np.exp(-0.5 * ((timestamp - (step_time - 0.10)) / 0.09) ** 2)

    acc = np.zeros((len(timestamp), 3), dtype=float)
    acc[:, 0] = horizontal * np.sin(true_heading)
    acc[:, 1] = horizontal * np.cos(true_heading)
    acc[:, 2] = 9.81 + impact
    acc += bias
    acc += rng.normal(0.0, noise_std, size=acc.shape)
    return acc


def _make_gyro(timestamp, true_heading, config, rng):
    synthetic = config["synthetic"]
    noise_std = float(synthetic["gyro_noise_std_rad_s"])
    bias_z = float(synthetic["gyro_bias_z_rad_s"])
    unwrapped_heading = np.unwrap(true_heading)
    yaw_rate = np.gradient(unwrapped_heading, timestamp)

    gyro = rng.normal(0.0, noise_std, size=(len(timestamp), 3))
    gyro[:, 2] = yaw_rate + bias_z + rng.normal(0.0, noise_std, size=len(timestamp))
    return gyro


def _make_magnetometer(true_heading, config, rng):
    synthetic = config["synthetic"]
    bias = np.array(synthetic["mag_bias"], dtype=float)
    noise_std = float(synthetic["mag_noise_std"])

    mag = np.zeros((len(true_heading), 3), dtype=float)
    mag[:, 0] = np.cos(true_heading)
    mag[:, 1] = -np.sin(true_heading)
    mag[:, 2] = 0.34
    mag += bias
    mag += rng.normal(0.0, noise_std, size=mag.shape)
    return mag


def _make_external_positions(timestamp, true_position, config, rng):
    external = np.full_like(true_position, np.nan)
    noise_std = float(config["synthetic"]["external_position_noise_m"])
    observation_times = list(np.arange(5.0, timestamp[-1], 2.0))
    observation_times.append(float(timestamp[-1]))
    for observation_time in observation_times:
        index = int(np.argmin(np.abs(timestamp - observation_time)))
        external[index] = true_position[index] + rng.normal(0.0, noise_std, size=2)
    return external


def _write_walk_csv(path, timestamp, acc, gyro, mag, external, true_position, true_heading, is_step):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, t in enumerate(timestamp):
            writer.writerow(
                {
                    "timestamp": f"{t:.4f}",
                    "acc_x": f"{acc[i, 0]:.8f}",
                    "acc_y": f"{acc[i, 1]:.8f}",
                    "acc_z": f"{acc[i, 2]:.8f}",
                    "gyro_x": f"{gyro[i, 0]:.8f}",
                    "gyro_y": f"{gyro[i, 1]:.8f}",
                    "gyro_z": f"{gyro[i, 2]:.8f}",
                    "mag_x": f"{mag[i, 0]:.8f}",
                    "mag_y": f"{mag[i, 1]:.8f}",
                    "mag_z": f"{mag[i, 2]:.8f}",
                    "external_e": "" if not np.isfinite(external[i, 0]) else f"{external[i, 0]:.8f}",
                    "external_n": "" if not np.isfinite(external[i, 1]) else f"{external[i, 1]:.8f}",
                    "true_e": f"{true_position[i, 0]:.8f}",
                    "true_n": f"{true_position[i, 1]:.8f}",
                    "true_heading": f"{true_heading[i]:.8f}",
                    "is_step": int(is_step[i]),
                }
            )


def _write_stride_training_csv(path, rng):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 260
    frequency = rng.uniform(1.05, 1.85, size=n)
    acc_variance = rng.uniform(0.08, 0.62, size=n) + 0.04 * (frequency - 1.45)
    acc_variance = np.clip(acc_variance, 0.05, 0.75)
    true_stride = (
        0.38
        + 0.19 * frequency
        + 0.12 * acc_variance
        + rng.normal(0.0, 0.022, size=n)
    )
    true_stride = np.clip(true_stride, 0.45, 0.95)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["step_frequency", "acc_variance", "true_stride"]
        )
        writer.writeheader()
        for f_hz, variance, stride in zip(frequency, acc_variance, true_stride):
            writer.writerow(
                {
                    "step_frequency": f"{f_hz:.8f}",
                    "acc_variance": f"{variance:.8f}",
                    "true_stride": f"{stride:.8f}",
                }
            )


def generate(config):
    rng = np.random.default_rng(int(config["synthetic"]["random_seed"]))
    fs = float(config["sampling_frequency_hz"])
    schedule = _step_schedule(rng)
    timestamp = np.arange(0.0, schedule["end_time"] + 0.5 / fs, 1.0 / fs)
    true_heading = _make_heading(timestamp, schedule)
    true_position, true_strides = _make_positions(timestamp, schedule, rng)
    acc = _make_acceleration(timestamp, true_heading, schedule, true_strides, config, rng)
    gyro = _make_gyro(timestamp, true_heading, config, rng)
    mag = _make_magnetometer(true_heading, config, rng)
    external = _make_external_positions(timestamp, true_position, config, rng)

    is_step = np.zeros(len(timestamp), dtype=bool)
    for step_time in schedule["step_times"]:
        is_step[int(np.argmin(np.abs(timestamp - step_time)))] = True

    return timestamp, acc, gyro, mag, external, true_position, true_heading, is_step


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic PDR example data.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "default.json",
        help="Path to config JSON.",
    )
    parser.add_argument(
        "--walk-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "example_walk.csv",
        help="Path for generated walking CSV.",
    )
    parser.add_argument(
        "--training-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "stride_training.csv",
        help="Path for generated stride-training CSV.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    timestamp, acc, gyro, mag, external, true_position, true_heading, is_step = generate(config)
    _write_walk_csv(
        args.walk_output,
        timestamp,
        acc,
        gyro,
        mag,
        external,
        true_position,
        true_heading,
        is_step,
    )
    rng = np.random.default_rng(int(config["synthetic"]["random_seed"]) + 101)
    _write_stride_training_csv(args.training_output, rng)

    print(f"Wrote {args.walk_output}")
    print(f"Wrote {args.training_output}")


if __name__ == "__main__":
    main()
