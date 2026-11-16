"""Reconstruct waist movement (orientation + position) from one SisFall trial.

Position comes from double-integrating acceleration, which drifts fast, so this is
an estimate rather than ground truth:

1. Still periods are detected (|a| steady and little rotation). The gyro bias is
   measured there and subtracted.
2. A Mahony filter fuses gyro + accel into orientation, so acceleration can be
   rotated into the world frame (z up). Heading is unobservable without a
   magnetometer, so world x/y are arbitrary but consistent within a trial.
3. Gravity is removed using the world-frame reading during still periods, which
   also absorbs the accelerometer's calibration error.
4. Velocity is integrated per moving segment and forced to zero wherever the
   person is still (zero-velocity updates). Segments longer than a few seconds
   (walking, jogging) are high-passed, so they show the body's bobbing and
   sway but not its forward progress.

Expect vertical movement to be roughly right (sitting down drops 0.3-0.5 m) while
falls end lower than they started but usually by less than the true drop.
"""

import math

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import binary_opening, uniform_filter1d
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation

from ysafetyml.sisfall import SAMPLE_HZ

G = 9.80665
DT = 1 / SAMPLE_HZ


def still_mask(acc: np.ndarray, gyr_deg: np.ndarray, win_s: float = 0.3) -> np.ndarray:
    """True where the sensor is at rest: |a| steady within 0.015 g and |w| < 20 deg/s."""
    n = int(win_s * SAMPLE_HZ)
    mag = np.linalg.norm(acc, axis=1)
    mean = uniform_filter1d(mag, n)
    sd = np.sqrt(np.maximum(uniform_filter1d(mag * mag, n) - mean**2, 0))
    rot = uniform_filter1d(np.linalg.norm(gyr_deg, axis=1), n)
    return binary_opening((sd < 0.015) & (rot < 20), np.ones(n))


def mahony(acc: np.ndarray, gyr: np.ndarray, kp: float = 1.0) -> np.ndarray:
    """Body->world orientation per sample as (x, y, z, w) quaternions. acc in g, gyr in rad/s."""
    x, y, z, w = Rotation.align_vectors([[0, 0, 1]], [acc[0]])[0].as_quat().tolist()
    out = np.empty((len(acc), 4))
    h = 0.5 * DT
    # Plain floats: this loop is ~10x faster than doing it with numpy scalars.
    for i, ((ax, ay, az), (gx, gy, gz)) in enumerate(zip(acc.tolist(), gyr.tolist())):
        nrm = math.sqrt(ax * ax + ay * ay + az * az)
        if abs(nrm - 1.0) < 0.15:  # accel only points at gravity when not accelerating much
            ax, ay, az = ax / nrm, ay / nrm, az / nrm
            # world up expressed in the body frame
            vx, vy, vz = 2 * (x * z - w * y), 2 * (w * x + y * z), w * w - x * x - y * y + z * z
            gx += kp * (ay * vz - az * vy)
            gy += kp * (az * vx - ax * vz)
            gz += kp * (ax * vy - ay * vx)
        w, x, y, z = (
            w + h * (-x * gx - y * gy - z * gz),
            x + h * (w * gx + y * gz - z * gy),
            y + h * (w * gy - x * gz + z * gx),
            z + h * (w * gz + x * gy - y * gx),
        )
        n = 1.0 / math.sqrt(w * w + x * x + y * y + z * z)
        w, x, y, z = w * n, x * n, y * n, z * n
        out[i] = (x, y, z, w)
    return out


def reconstruct(
    acc: np.ndarray, gyr_deg: np.ndarray, hp_hz: float = 0.25, hp_min_s: float = 3.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One trial at 200 Hz -> (position m (n,3), orientation quats xyzw (n,4), still mask (n,)).

    ``acc`` in g (the MMA8451Q is the better-calibrated of the two accelerometers),
    ``gyr_deg`` in deg/s. Position starts at the origin.
    """
    acc = acc.astype(np.float64)
    gyr_deg = gyr_deg.astype(np.float64)
    still = still_mask(acc, gyr_deg)
    bias = gyr_deg[still].mean(0) if still.sum() > SAMPLE_HZ // 2 else np.median(gyr_deg, 0)
    quat = mahony(acc, np.radians(gyr_deg - bias))
    a_world = Rotation.from_quat(quat).apply(acc)

    # what this sensor reads as gravity: its still-period readings, interpolated across movement
    if still.any():
        idx = np.arange(len(acc))
        gravity = np.column_stack([np.interp(idx, idx[still], a_world[still, k]) for k in range(3)])
    else:
        gravity = np.array([0.0, 0.0, 1.0])
    lin = (a_world - gravity) * G

    vel = np.zeros_like(lin)
    moving = ~still
    starts = np.flatnonzero(moving & ~np.r_[False, moving[:-1]])
    ends = np.flatnonzero(moving & ~np.r_[moving[1:], False])
    hb, ha = butter(2, hp_hz, "highpass", fs=SAMPLE_HZ)
    for a, b in zip(starts, ends):
        seg = cumulative_trapezoid(lin[a : b + 1], dx=DT, axis=0, initial=0)
        if b - a > hp_min_s * SAMPLE_HZ:
            seg = filtfilt(hb, ha, seg, axis=0)
        # zero-velocity anchors only where the segment touches a still period
        anchor_start, anchor_end = a > 0, b + 1 < len(vel)
        if anchor_start and anchor_end:
            frac = np.linspace(0, 1, len(seg))[:, None]
            seg -= seg[0] + frac * (seg[-1] - seg[0])
        elif anchor_start:
            seg -= seg[0]
        elif anchor_end:
            seg -= seg[-1]
        vel[a : b + 1] = seg

    pos = cumulative_trapezoid(vel, dx=DT, axis=0, initial=0)
    return pos, quat, still
