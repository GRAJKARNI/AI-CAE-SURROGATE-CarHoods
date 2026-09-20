"""Engineered geometry descriptors documented in the deformation study."""
import numpy as np

def geometry_descriptors(point_clouds):
    """Return 9 descriptors: XYZ spans/stds and radial mean/std/max."""
    feats = []
    for pts in np.asarray(point_clouds):
        xyz_min = pts.min(axis=0)
        xyz_max = pts.max(axis=0)
        spans = xyz_max - xyz_min
        centroid = pts.mean(axis=0)
        stds = pts.std(axis=0)
        radial = np.linalg.norm(pts - centroid, axis=1)
        feats.append([
            spans[0], spans[1], spans[2],
            stds[0], stds[1], stds[2],
            radial.mean(), radial.std(), radial.max()
        ])
    return np.asarray(feats, dtype=np.float32)

# Note: the final stress study documents 24 engineered descriptors.
# The complete original 24-feature extraction code is not preserved in the
# documentation notebook, so it is intentionally not reconstructed here.
