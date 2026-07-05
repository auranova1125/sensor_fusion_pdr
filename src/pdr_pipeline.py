import numpy as np

from .kalman_filter import KalmanFilterPDR
from .stride_model import predict_stride


def _is_valid_external(position):
    return np.all(np.isfinite(position))


def _is_valid_magnetic_field(mag_vector, reference_norm, threshold_ratio):
    norm = float(np.linalg.norm(mag_vector))
    if reference_norm <= 1e-9:
        return False
    ratio = abs(norm - reference_norm) / reference_norm
    return ratio <= threshold_ratio


def run_pdr(data, processed, step_features, stride_model, config):
    timestamp = data["timestamp"]
    initial_state = np.array([0.0, 0.0, 0.0, processed["gyro_bias_z"]], dtype=float)
    kf = KalmanFilterPDR(
        initial_state,
        config["kalman"]["initial_covariance"],
        config,
    )

    states = np.zeros((len(timestamp), 4), dtype=float)
    cov_diag = np.zeros((len(timestamp), 4), dtype=float)
    step_lookup = {feature["index"]: dict(feature) for feature in step_features}
    step_records = []
    counts = {
        "mag_accepted": 0,
        "mag_rejected": 0,
        "external_accepted": 0,
        "external_rejected": 0,
    }

    magnetic_threshold = float(config["kalman"]["magnetic_disturbance_ratio"])
    for i, _ in enumerate(timestamp):
        dt = 0.0 if i == 0 else timestamp[i] - timestamp[i - 1]
        kf.predict(processed["gyro_z"][i], dt)

        if _is_valid_magnetic_field(
            processed["mag"][i],
            processed["mag_reference_norm"],
            magnetic_threshold,
        ):
            accepted, _ = kf.update_heading(processed["mag_heading"][i])
            counts["mag_accepted" if accepted else "mag_rejected"] += 1
        else:
            counts["mag_rejected"] += 1

        external_position = data["external_position"][i]
        if _is_valid_external(external_position):
            accepted, _, _ = kf.update_position(external_position)
            counts["external_accepted" if accepted else "external_rejected"] += 1

        if i in step_lookup:
            record = step_lookup[i]
            stride = predict_stride(
                stride_model, record["frequency"], record["acc_variance"]
            )
            kf.propagate_step(stride)
            record["estimated_stride"] = float(stride)
            record["estimated_e"] = float(kf.x[0])
            record["estimated_n"] = float(kf.x[1])
            record["estimated_heading"] = float(kf.x[2])
            step_records.append(record)

        states[i] = kf.x
        cov_diag[i] = np.diag(kf.P)

    return {
        "states": states,
        "covariance_diag": cov_diag,
        "step_records": step_records,
        "counts": counts,
    }
