"""Data loading and point-cloud preprocessing utilities."""
import h5py
import numpy as np

def load_carhoods_h5(file_path):
    required = ("design_parameters", "points", "deformation", "stress", "mass")
    with h5py.File(file_path, "r") as f:
        missing = [k for k in required if k not in f]
        if missing:
            raise KeyError(f"Missing H5 datasets: {missing}")
        return {k: f[k][:] for k in required}

def valid_point_cloud_mask(points):
    points = np.asarray(points)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape (N, n_points, 3)")
    return np.any(np.abs(points) > 0, axis=(1, 2))

def filter_valid_designs(data):
    mask = valid_point_cloud_mask(data["points"])
    return {k: np.asarray(v)[mask] for k, v in data.items()}

def sample_and_center_points(points, n_points=1024, seed=42, scale=1000.0):
    """Deterministically sample, center, and scale while preserving physical size."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape (N, n_points, 3)")
    if n_points > points.shape[1]:
        raise ValueError("n_points cannot exceed available points")
    if n_points < points.shape[1]:
        rng = np.random.default_rng(seed)
        idx = rng.choice(points.shape[1], n_points, replace=False)
        points = points[:, idx, :].copy()
    else:
        points = points.copy()
    points -= points.mean(axis=1, keepdims=True)
    return (points / float(scale)).astype(np.float32)
