import numpy as np


def detect_step_peaks(timestamp, acc_signal, config):
    settings = config["step_detection"]
    threshold = float(settings["threshold_mps2"])
    min_interval = float(settings["min_interval_s"])

    peaks = []
    for i in range(1, len(acc_signal) - 1):
        is_local_peak = acc_signal[i] > acc_signal[i - 1] and acc_signal[i] >= acc_signal[i + 1]
        if not is_local_peak or acc_signal[i] < threshold:
            continue

        if not peaks:
            peaks.append(i)
            continue

        elapsed = timestamp[i] - timestamp[peaks[-1]]
        if elapsed >= min_interval:
            peaks.append(i)
        elif acc_signal[i] > acc_signal[peaks[-1]]:
            peaks[-1] = i

    return np.array(peaks, dtype=int)


def build_step_features(timestamp, acc_signal, peak_indices, config, true_position=None):
    settings = config["step_detection"]
    before = float(settings["window_before_s"])
    after = float(settings["window_after_s"])
    min_frequency = 1.0 / float(settings["max_interval_s"])
    max_frequency = 1.0 / float(settings["min_interval_s"])

    features = []
    previous_true_position = true_position[0] if true_position is not None else None

    for k, peak_index in enumerate(peak_indices):
        peak_time = timestamp[peak_index]
        start_index = int(np.searchsorted(timestamp, peak_time - before, side="left"))
        end_index = int(np.searchsorted(timestamp, peak_time + after, side="right"))
        start_index = max(0, start_index)
        end_index = min(len(timestamp), max(start_index + 2, end_index))

        if len(peak_indices) == 1:
            period = 0.75
        elif k == 0:
            period = timestamp[peak_indices[k + 1]] - timestamp[peak_index]
        elif k == len(peak_indices) - 1:
            period = timestamp[peak_index] - timestamp[peak_indices[k - 1]]
        else:
            period = 0.5 * (
                timestamp[peak_indices[k + 1]] - timestamp[peak_indices[k - 1]]
            )

        if period <= 1e-6:
            step_frequency = min_frequency
        else:
            step_frequency = float(np.clip(1.0 / period, min_frequency, max_frequency))

        segment = acc_signal[start_index:end_index]
        acc_variance = float(np.var(segment))
        true_stride = np.nan
        if true_position is not None:
            current_true_position = true_position[peak_index]
            true_stride = float(np.linalg.norm(current_true_position - previous_true_position))
            previous_true_position = current_true_position

        features.append(
            {
                "step_number": k + 1,
                "index": int(peak_index),
                "time": float(peak_time),
                "frequency": step_frequency,
                "acc_variance": acc_variance,
                "window_start": int(start_index),
                "window_end": int(end_index),
                "true_stride": true_stride,
            }
        )

    return features


def detect_steps(timestamp, acc_signal, config, true_position=None):
    peak_indices = detect_step_peaks(timestamp, acc_signal, config)
    return build_step_features(timestamp, acc_signal, peak_indices, config, true_position)
