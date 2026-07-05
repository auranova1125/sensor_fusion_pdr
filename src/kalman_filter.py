import numpy as np

from .preprocessing import wrap_angle


class KalmanFilterPDR:
    def __init__(self, initial_state, initial_covariance_diag, config):
        self.x = np.array(initial_state, dtype=float)
        self.P = np.diag(np.array(initial_covariance_diag, dtype=float))
        self.config = config["kalman"]

    def predict(self, gyro_z, dt):
        dt = max(0.0, float(dt))
        self.x[2] = wrap_angle(self.x[2] + (float(gyro_z) - self.x[3]) * dt)

        f = np.eye(4)
        f[2, 3] = -dt

        process = self.config["process_noise"]
        q_diag = np.array(
            [
                process["position_per_s"] * dt,
                process["position_per_s"] * dt,
                process["heading_per_s"] * dt,
                process["gyro_bias_per_s"] * dt,
            ],
            dtype=float,
        )
        self.P = f @ self.P @ f.T + np.diag(q_diag)

    def propagate_step(self, stride):
        stride = float(stride)
        heading = float(self.x[2])
        self.x[0] += stride * np.sin(heading)
        self.x[1] += stride * np.cos(heading)

        f = np.eye(4)
        f[0, 2] = stride * np.cos(heading)
        f[1, 2] = -stride * np.sin(heading)
        q_step = float(self.config["process_noise"]["step_position"])
        q = np.diag([q_step, q_step, 0.0, 0.0])
        self.P = f @ self.P @ f.T + q

    def update_heading(self, observed_heading):
        h = np.array([[0.0, 0.0, 1.0, 0.0]])
        r_value = float(self.config["measurement_noise"]["mag_heading_rad"]) ** 2
        residual = float(wrap_angle(observed_heading - self.x[2]))
        if abs(residual) > float(self.config["mag_heading_gate_rad"]):
            return False, residual
        self._joseph_update(np.array([residual]), h, np.array([[r_value]]))
        self.x[2] = wrap_angle(self.x[2])
        return True, residual

    def update_position(self, observed_position):
        h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        r_value = float(self.config["measurement_noise"]["external_position_m"]) ** 2
        r = np.diag([r_value, r_value])
        residual = np.asarray(observed_position, dtype=float) - h @ self.x
        s = h @ self.P @ h.T + r
        mahalanobis = float(residual.T @ np.linalg.inv(s) @ residual)
        gate = float(self.config["external_position_gate_sigma"])
        if mahalanobis > gate**2:
            return False, residual, mahalanobis
        self._joseph_update(residual, h, r)
        self.x[2] = wrap_angle(self.x[2])
        return True, residual, mahalanobis

    def _joseph_update(self, residual, h, r):
        s = h @ self.P @ h.T + r
        k = self.P @ h.T @ np.linalg.inv(s)
        self.x = self.x + k @ residual
        i = np.eye(self.P.shape[0])
        self.P = (i - k @ h) @ self.P @ (i - k @ h).T + k @ r @ k.T
        self.P = 0.5 * (self.P + self.P.T)
