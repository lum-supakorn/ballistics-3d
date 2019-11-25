import numpy as np
from scipy.constants import g


class Ballistic3D:
    def __init__(self, ballistic_params: dict):
        """Initiate the 3D ballistic trajectory.

        Args:
        ballistic_params (dict of float):
            - CD: Payload drag coefficient {-}
            - A: Payload frontal area {m^2}
            - m: Payload mass {kg}
            - rho: Air density {kg/m^3}

        Returns:

        """
        self.CD = ballistic_params["CD"]
        self.A = ballistic_params["A"]
        self.m = ballistic_params["m"]
        self.rho = ballistic_params["rho"]
        self.b = 0.5 * self.rho * self.A * self.CD

    def calculate_trajectory(self, p: list, v: list, w: list):
        """Solve the IVP for payload drop.

        Args:
            p: Drop position (XYZ)
            v: Drop velocity (XYZ)
            w: Wind velocity (XYZ)

        Returns:
            result (dict):
                - p: Final position vector
                - v: Final velocity vector
                - tof: Time of flight
                - XY_dist: Distance from release point to impact point
        """
        # Axis enumerators
        # North-East-Down
        # Z-Plane is the ground
        X, Y, Z = (0, 1, 2)

        # Convert vectors from lists into NumPy arrays
        v = np.asarray(v, dtype=np.float64)
        p = np.asarray(p, dtype=np.float64)
        w = np.asarray(w, dtype=np.float64)

        def mag(x):
            """Returns the magnitude of a vector"""
            return np.sqrt(x.dot(x))

        p_drop_init = np.copy(p)  # Payload release position
        tof = 0  # Time of flight
        t_max = 1000  # Maximum time of flight

        # Euler's method integration
        h = 0.0001  # Step size
        for t in np.arange(0, t_max, h):
            # Positions
            p[X] += v[X] * h
            p[Y] += v[Y] * h
            p[Z] += v[Z] * h

            # Check if z-plane is crossed
            if (p[2] > 0):
                tof = t
                break

            v_old = v  # Velocity vector in previous iteration
            v[X] += -((self.b/self.m) * (v_old[X] - w[X]) * mag(v - w)) * h
            v[Y] += -((self.b/self.m) * (v_old[Y] - w[Y]) * mag(v - w)) * h
            v[Z] += \
                (g - ((self.b/self.m) * (v_old[Z] - w[Z]) * mag(v - w))) * h

        return {
            "p": p,
            "v": v,
            "tof": tof,
            "XY_dist": np.linalg.norm(p[:-1] - p_drop_init[:-1])
        }
