import numpy as np


FEATURE_NAMES = ["step_frequency", "acc_variance"]


def make_feature_matrix(step_frequency, acc_variance):
    return np.column_stack([step_frequency, acc_variance]).astype(float)


def fit_stride_model(step_frequency, acc_variance, true_stride, stride_limits):
    x = make_feature_matrix(step_frequency, acc_variance)
    y = np.asarray(true_stride, dtype=float)
    feature_mean = x.mean(axis=0)
    feature_std = x.std(axis=0)
    feature_std[feature_std < 1e-9] = 1.0

    x_norm = (x - feature_mean) / feature_std
    design = np.column_stack([np.ones(len(x_norm)), x_norm])
    weights, *_ = np.linalg.lstsq(design, y, rcond=None)
    prediction = design @ weights
    metrics = evaluate_regression(y, prediction)

    return {
        "intercept": float(weights[0]),
        "weights": {
            "step_frequency": float(weights[1]),
            "acc_variance": float(weights[2]),
        },
        "feature_names": FEATURE_NAMES,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "stride_min_m": float(stride_limits["min_m"]),
        "stride_max_m": float(stride_limits["max_m"]),
        "training_metrics": metrics,
    }


def predict_stride(model, step_frequency, acc_variance):
    x = make_feature_matrix(np.asarray(step_frequency), np.asarray(acc_variance))
    if x.ndim == 1:
        x = x.reshape(1, -1)
    mean = np.array(model["feature_mean"], dtype=float)
    std = np.array(model["feature_std"], dtype=float)
    x_norm = (x - mean) / std
    weights = np.array(
        [
            model["intercept"],
            model["weights"]["step_frequency"],
            model["weights"]["acc_variance"],
        ],
        dtype=float,
    )
    design = np.column_stack([np.ones(len(x_norm)), x_norm])
    prediction = design @ weights
    prediction = np.clip(
        prediction, float(model["stride_min_m"]), float(model["stride_max_m"])
    )
    if prediction.size == 1:
        return float(prediction[0])
    return prediction


def evaluate_regression(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return {"mae": mae, "rmse": rmse, "r2": float(r2)}
