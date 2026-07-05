import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_io import load_config, load_stride_training_csv, save_stride_model
from src.stride_model import fit_stride_model, predict_stride
from src.visualization import plot_stride_regression


def main():
    parser = argparse.ArgumentParser(description="Train stride-length regression model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "default.json",
        help="Path to config JSON.",
    )
    parser.add_argument(
        "--training-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "stride_training.csv",
        help="Path to stride-training CSV.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=PROJECT_ROOT / "models" / "stride_model.json",
        help="Path for trained model JSON.",
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "figures" / "stride_regression.png",
        help="Path for regression figure.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    training = load_stride_training_csv(args.training_csv)
    model = fit_stride_model(
        training["step_frequency"],
        training["acc_variance"],
        training["true_stride"],
        config["stride"],
    )
    save_stride_model(args.model_output, model)
    plot_stride_regression(training, model, args.figure_output)

    metrics = model["training_metrics"]
    print(f"Wrote {args.model_output}")
    print(f"Wrote {args.figure_output}")
    print(
        "Training metrics: "
        f"MAE={metrics['mae']:.4f} m, RMSE={metrics['rmse']:.4f} m, R2={metrics['r2']:.4f}"
    )


if __name__ == "__main__":
    main()
