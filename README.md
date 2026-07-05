# Sensor Fusion PDR Minimal Implementation

This project implements a small 2D pedestrian dead reckoning (PDR) pipeline on
an East-North plane. It uses 100 Hz accelerometer, gyroscope, magnetometer, and
intermittent external East-North position observations.

The main purpose of the repository is the PDR algorithm itself: sensor
preprocessing, step detection, stride-length estimation, and Kalman-filter
fusion. The included CSV files are pre-generated demonstration data so the
algorithm can be run and inspected immediately. Data synthesis is provided as a
supporting appendix workflow, not as the central algorithm.

## Assumptions

The inertial and magnetic sensor package is assumed to be fixed on the user's
waist. The heading model also assumes that the offset between the sensor frame
and the walking direction is constant during the walk.

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Run

From this directory:

```bash
python3 scripts/train_stride_model.py
python3 main.py
```

The workflow is:

1. Train the stride-length linear regression model from the provided training
   CSV.
2. Run sensor preprocessing, step detection, stride estimation, and Kalman
   filtering.
3. Save trajectory, metrics, and figures.

The repository already includes `data/example_walk.csv` and
`data/stride_training.csv`, so data generation is not required for the normal
demo run.

## Input CSV Format

`data/example_walk.csv` contains:

```text
timestamp
acc_x, acc_y, acc_z
gyro_x, gyro_y, gyro_z
mag_x, mag_y, mag_z
external_e, external_n
true_e, true_n
true_heading
is_step
```

Rows without external position observations use empty `external_e` and
`external_n` fields. The parser converts those fields to `NaN`.

`data/stride_training.csv` contains:

```text
step_frequency
acc_variance
true_stride
```

## Outputs

`main.py` writes:

- `outputs/trajectory.csv`
- `outputs/metrics.json`
- `outputs/figures/trajectory_en.png`
- `outputs/figures/step_detection.png`
- `outputs/figures/heading_and_bias.png`
- `outputs/figures/stride_features.png`

`scripts/train_stride_model.py` writes:

- `models/stride_model.json`
- `outputs/figures/stride_regression.png`

## Demo Data Appendix

The synthetic data generator is included only to make the demonstration
reproducible. It creates the example walking CSV and the stride-regression
training CSV used by the default run.

To regenerate the bundled demo data:

```bash
python3 scripts/generate_example_data.py
```

The default synthetic scenario is:

1. Stand still for 5 seconds while facing north.
2. Walk 10 steps north.
3. Turn right 90 degrees.
4. Walk 5 steps east.
5. Turn left 90 degrees.
6. Walk 10 steps north.

Ground-truth position, heading, and step labels are saved in the example CSV for
evaluation and plots only. The PDR algorithm estimates its state from sensor and
external-position inputs.

## State Model

The Kalman filter state is:

```text
x = [p_E, p_N, heading, gyro_bias]^T
```

Heading is measured in radians clockwise from north, so:

```text
east_increment  = stride * sin(heading)
north_increment = stride * cos(heading)
```

The filter predicts heading from gyroscope yaw rate, propagates position on
detected steps, and updates with magnetometer heading and intermittent external
position observations.
