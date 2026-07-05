import numpy as np


def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def moving_average(values, window):
    window = int(max(1, window))
    if window == 1:
        return np.asarray(values, dtype=float).copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _apply_axis_transform(values, axis_config, prefix):
    order = axis_config.get(f"{prefix}_order", [0, 1, 2])
    sign = axis_config.get(f"{prefix}_sign", [1, 1, 1])
    transformed = values[:, order].astype(float).copy()
    return transformed * np.array(sign, dtype=float)


def preprocess_sensors(data, config):
    axis_config = config.get("sensor_axis", {})
    acc = _apply_axis_transform(data["acc"], axis_config, "acc")
    gyro = _apply_axis_transform(data["gyro"], axis_config, "gyro")
    mag = _apply_axis_transform(data["mag"], axis_config, "mag")

    if axis_config.get("gyro_in_degrees", False):
        gyro = np.deg2rad(gyro)

    timestamp = data["timestamp"]
    initial_duration = float(config["initial_stationary_s"])
    stationary_mask = timestamp <= timestamp[0] + initial_duration
    if stationary_mask.sum() < 2:
        raise ValueError("Initial stationary window is too short")

    acc_magnitude = np.linalg.norm(acc, axis=1)
    gravity_magnitude = float(np.mean(acc_magnitude[stationary_mask]))
    linear_acc_magnitude = acc_magnitude - gravity_magnitude
    smooth_window = int(config["step_detection"]["smooth_window_samples"])
    smooth_linear_acc = moving_average(linear_acc_magnitude, smooth_window)

    gyro_bias_z = float(np.mean(gyro[stationary_mask, 2]))
    gyro_z = gyro[:, 2]

    mag_reference = np.mean(mag[stationary_mask], axis=0)
    mag_reference_norm = float(np.linalg.norm(mag_reference))
    raw_reference_heading = float(np.arctan2(-mag_reference[1], mag_reference[0]))
    raw_mag_heading = np.arctan2(-mag[:, 1], mag[:, 0])
    mounting_offset = float(axis_config.get("mounting_heading_offset_rad", 0.0))
    mag_heading = wrap_angle(raw_mag_heading - raw_reference_heading + mounting_offset)

    gyro_heading = np.zeros_like(timestamp)
    for i in range(1, len(timestamp)):
        dt = timestamp[i] - timestamp[i - 1]
        gyro_heading[i] = wrap_angle(
            gyro_heading[i - 1] + (gyro_z[i] - gyro_bias_z) * dt
        )

    return {
        "acc": acc,
        "gyro": gyro,
        "mag": mag,
        "acc_magnitude": acc_magnitude,
        "gravity_magnitude": gravity_magnitude,
        "linear_acc_magnitude": linear_acc_magnitude,
        "smooth_linear_acc": smooth_linear_acc,
        "stationary_mask": stationary_mask,
        "gyro_bias_z": gyro_bias_z,
        "gyro_z": gyro_z,
        "gyro_heading": gyro_heading,
        "mag_reference": mag_reference,
        "mag_reference_norm": mag_reference_norm,
        "mag_heading": mag_heading,
    }
