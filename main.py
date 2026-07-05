import argparse
from pathlib import Path

from src.data_io import (
    load_config,
    load_stride_model,
    load_walk_csv,
    save_metrics_json,
    save_trajectory_csv,
)
from src.metrics import compute_metrics
from src.pdr_pipeline import run_pdr
from src.preprocessing import preprocess_sensors
from src.step_detection import detect_steps
from src.visualization import (
    plot_heading_and_bias,
    plot_step_detection,
    plot_stride_features,
    plot_trajectory,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def run(config_path, data_path, model_path, output_dir):
    config = load_config(config_path)
    model = load_stride_model(model_path)
    data = load_walk_csv(data_path)
    processed = preprocess_sensors(data, config)
    step_features = detect_steps(
        data["timestamp"],
        processed["smooth_linear_acc"],
        config,
        true_position=data["true_position"],
    )
    result = run_pdr(data, processed, step_features, model, config)
    metrics = compute_metrics(data, result)

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    save_trajectory_csv(output_dir / "trajectory.csv", data, result)
    save_metrics_json(output_dir / "metrics.json", metrics)
    plot_trajectory(data, result, figure_dir / "trajectory_en.png")
    plot_step_detection(data, processed, result, figure_dir / "step_detection.png")
    plot_heading_and_bias(data, processed, result, config, figure_dir / "heading_and_bias.png")
    plot_stride_features(result, figure_dir / "stride_features.png")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run minimal sensor-fusion PDR.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "default.json",
        help="Path to config JSON.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "example_walk.csv",
        help="Path to walking CSV.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models" / "stride_model.json",
        help="Path to stride model JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Directory for trajectory, metrics, and figures.",
    )
    args = parser.parse_args()

    metrics = run(args.config, args.data, args.model, args.output_dir)
    print(f"Wrote {args.output_dir / 'trajectory.csv'}")
    print(f"Wrote {args.output_dir / 'metrics.json'}")
    print(f"Detected steps: {metrics['detected_step_count']} / true {metrics['true_step_count']}")
    print(f"Endpoint error: {metrics['endpoint_position_error_m']:.3f} m")
    print(f"Trajectory RMSE: {metrics['trajectory_rmse_m']:.3f} m")


if __name__ == "__main__":
    main()
