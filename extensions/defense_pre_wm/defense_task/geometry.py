"""Pure geometry helpers (no robosuite dependency)."""

import numpy as np


def quat_to_matrix(q):
    """MuJoCo quaternion order is w, x, y, z."""
    q = np.asarray(q, dtype=float).reshape(4)
    n = float(np.linalg.norm(q))
    if not np.isfinite(q).all() or n < 1e-12:
        raise ValueError("zero-norm quaternion")
    w, x, y, z = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _convex_hull(points):
    """Monotonic chain for the horizontal projection of the 3-D box."""
    pts = sorted(set(map(tuple, np.asarray(points, dtype=float))))
    if len(pts) <= 1:
        return np.asarray(pts, dtype=float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def footprint_distance(position, quat_wxyz, half_size, center_xy):
    """Exact min XY distance from circle center to projected oriented box.

    3D box's 8 corners are projected into XY; their convex hull is tested.
    Supports tipped cubes as well as yaw rotations. Returns meters >= 0.
    """
    p = np.asarray(position, dtype=float).reshape(3)
    h = np.asarray(half_size, dtype=float).reshape(3)
    if np.any(h <= 0):
        raise ValueError("half_size must be positive")
    c = np.asarray(center_xy, dtype=float).reshape(2)
    rot = quat_to_matrix(quat_wxyz)
    corners = np.array([[sx * h[0], sy * h[1], sz * h[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    poly = _convex_hull((p + corners @ rot.T)[:, :2])
    if len(poly) < 3:
        return float(np.linalg.norm(p[:2] - c))
    # Convex CCW polygon contains c iff all cross products >= 0.
    inside = True
    best = np.inf
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        v = b - a
        signed = v[0] * (c[1] - a[1]) - v[1] * (c[0] - a[0])
        if signed < -1e-10:
            inside = False
        t = float(np.clip(np.dot(c - a, v) / max(np.dot(v, v), 1e-15), 0, 1))
        best = min(best, np.linalg.norm(c - (a + t * v)))
    return 0.0 if inside else float(best)


def circle_intrusion(position, quat_wxyz, half_size, center_xy, radius, table_top, height_tolerance=0.035):
    """True iff a non-lifted block's projected footprint enters the circle.

    A cube carried wholly above table_top+height_tolerance is safe.
    Boundary contact counts as a breach.
    """
    p = np.asarray(position, dtype=float)
    h = np.asarray(half_size, dtype=float)
    rot = quat_to_matrix(quat_wxyz)
    vertical_extent = float(np.sum(np.abs(rot[2, :]) * h))
    bottom = p[2] - vertical_extent
    if bottom > table_top + height_tolerance:
        return False
    return footprint_distance(p, quat_wxyz, h, center_xy) <= float(radius) + 1e-9


def sample_spawns(rng, n, center_xy, radius_range, angle_range, half_xy, table_half_xy, min_gap=0.018):
    """Collision-free polar placements on tabletop; random but seedable.

    half_xy is the attacker half-size in XY. table_half_xy are table half-extents.
    """
    rng = np.random.default_rng(rng) if not hasattr(rng, "uniform") else rng
    if int(n) != n or n < 1:
        raise ValueError("n must be a positive integer")
    n = int(n)
    rmin, rmax = map(float, radius_range)
    a0, a1 = map(float, angle_range)
    if not (0 < rmin <= rmax and a0 < a1):
        raise ValueError("invalid radius or angle range")
    center_xy = np.asarray(center_xy, dtype=float).reshape(2)
    half_xy = np.asarray(half_xy, dtype=float).reshape(2)
    table_half_xy = np.asarray(table_half_xy, dtype=float).reshape(2)
    if (
        not np.isfinite(np.r_[center_xy, half_xy, table_half_xy, radius_range, angle_range, min_gap]).all()
        or np.any(half_xy <= 0)
        or np.any(table_half_xy <= 0)
        or min_gap < 0
    ):
        raise ValueError("Invalid placement geometry")
    result = []
    min_center_gap = 2 * np.linalg.norm(half_xy) + min_gap
    for i in range(n):
        found = False
        # Evenly spaced angular sectors with randomized offset, avoiding clustering.
        sector = (a1 - a0) / n
        for attempt in range(2000):
            ang = float(rng.uniform(a0 + i * sector + 0.07 * sector, a0 + (i + 1) * sector - 0.07 * sector))
            r = float(rng.uniform(rmin, rmax))
            pos = center_xy + r * np.array([np.cos(ang), np.sin(ang)])
            if np.any(np.abs(pos) + np.linalg.norm(half_xy) + 0.01 >= table_half_xy):
                continue
            if all(np.linalg.norm(pos - p) > min_center_gap for p, _ in result):
                result.append((pos, ang))
                found = True
                break
        if not found:
            raise RuntimeError("Cannot place all blocks; enlarge table or reduce n/range")
    return result
