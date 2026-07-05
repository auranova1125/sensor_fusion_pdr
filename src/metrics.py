import numpy as np

from .preprocessing import wrap_angle


def compute_metrics(data, result):
    states = result["states"]
    true_position = data["true_position"]
    true_heading = data["true_heading"]
    step_records = result["step_records"]

    position_error = states[:, :2] - true_position
    trajectory_rmse_m = float(np.sqrt(np.mean(np.sum(position_error**2, axis=1))))
    endpoint_position_error_m = float(np.linalg.norm(position_error[-1]))
    heading_error = wrap_angle(states[:, 2] - true_heading)
    heading_rmse_rad = float(np.sqrt(np.mean(heading_error**2)))

    estimated_strides = np.array(
        [record["estimated_stride"] for record in step_records], dtype=float
    )
    true_strides = np.array(
        [record.get("true_stride", np.nan) for record in step_records], dtype=float
    )
    stride_mask = np.isfinite(true_strides)
    stride_mae_m = (
        float(np.mean(np.abs(estimated_strides[stride_mask] - true_strides[stride_mask])))
        if stride_mask.any()
        else None
    )

    true_step_count = int(np.sum(data["is_step"]))
    detected_step_count = int(len(step_records))

    return {
        "true_step_count": true_step_count,
        "detected_step_count": detected_step_count,
        "step_count_error": detected_step_count - true_step_count,
        "mean_estimated_stride_m": float(np.mean(estimated_strides))
        if len(estimated_strides)
        else 0.0,
        "estimated_total_distance_m": float(np.sum(estimated_strides))
        if len(estimated_strides)
        else 0.0,
        "endpoint_position_error_m": endpoint_position_error_m,
        "trajectory_rmse_m": trajectory_rmse_m,
        "heading_rmse_rad": heading_rmse_rad,
        "heading_rmse_deg": float(np.rad2deg(heading_rmse_rad)),
        "stride_mae_m": stride_mae_m,
        "estimated_gyro_bias_rad_s": float(states[-1, 3]),
        "mag_accepted": int(result["counts"]["mag_accepted"]),
        "mag_rejected": int(result["counts"]["mag_rejected"]),
        "external_position_accepted": int(result["counts"]["external_accepted"]),
        "external_position_rejected": int(result["counts"]["external_rejected"]),
    }
