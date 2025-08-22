import numpy as np
from typing import Callable, Sequence, Tuple, List, Union, Dict
from scipy.spatial import ConvexHull


def compute_tangential_planes_equation(
    points: Sequence[np.ndarray],
    func: Callable[..., float],
    func_grad: Sequence[Callable[..., float]],
) -> List[Tuple[float, np.ndarray]]:
    tangential_planes = []
    for point in points:
        if isinstance(point, float):
            grad_val = func_grad(point)
            z_val = func(point)
        else:
            grad_val = np.array([grad(*point) for grad in func_grad])
            z_val = func(*point)
        z0 = z_val - np.dot(grad_val, point)
        tangential_planes.append((z0, grad_val, point))
    return tangential_planes


def remove_intersecting_hyperplanes(
    tangential_planes: List[Tuple[float, np.ndarray]],
    func: Callable[..., float],
    points: Sequence[np.ndarray],
) -> List[Tuple[float, np.ndarray]]:
    for test_point in points:
        if isinstance(test_point, float):
            fval = func(test_point)
        else:
            fval = func(*test_point)
        updated_planes = []
        for z0, grad_vec, point_hyp in tangential_planes:
            under_val = z0 + np.dot(grad_vec, test_point)
            if under_val <= fval:
                updated_planes.append((z0, grad_vec, point_hyp))
        tangential_planes = updated_planes
    return tangential_planes


def tangential_planes_to_array(
    tangential_planes: Sequence[Tuple[float, np.ndarray]],
) -> np.ndarray:
    """
    Converts list of (z0, grad_vector) tuples into a 2D NumPy array
    where each row is [z0, *grad_vector].
    """
    rows = []
    for z0, grad_vec, point in tangential_planes:
        if isinstance(grad_vec, float):
            rows.append(np.array([z0, grad_vec, point]))
        else:
            rows.append(np.array([z0, *grad_vec, *point]))
    return np.vstack(rows)


def sample_over_grid_nd(
    intervals: Sequence[np.ndarray],
    constraint: Callable[..., np.ndarray] | None = None,
) -> np.ndarray:
    """
    Evaluates a function f over an N-dimensional meshgrid and returns stacked input-output samples.
    Optionally filters the samples using a constraint function.

    Parameters:
        intervals (Sequence[np.ndarray]): A list of 1D arrays representing the grid in each dimension.
        constraint (Callable, optional): Function c(x1, x2, ..., xn) → bool or array of bools.
                                         Only samples where c(...) ≤ 1 are kept (if provided).

    Returns:
        np.ndarray: Array of shape (n_points, n_dims), each row is [x1, x2, ..., xn].
    """
    mesh = np.meshgrid(*intervals, indexing="ij")

    if constraint is not None:
        mask = constraint(*mesh) <= 1
        flattened_inputs = [axis[mask] for axis in mesh]
    else:
        flattened_inputs = [axis.ravel() for axis in mesh]

    return np.column_stack(flattened_inputs)


# -----------------------------------------------------------
#  Helpers
# -----------------------------------------------------------
def _cross_2d(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Scalar z-component of the 2-D cross product (OA x OB)."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _monotone_chain_lower(points: np.ndarray) -> List[int]:
    """Indices of the lower convex hull (x-sorted, left→right)."""
    order = np.lexsort((points[:, 1], points[:, 0]))  # sort x, then y
    lower: List[int] = []
    for idx in order:
        while (
            len(lower) >= 2
            and _cross_2d(points[lower[-2]], points[lower[-1]], points[idx]) <= 0
        ):  #   ‘<=’ keeps collinear outer points
            lower.pop()
        lower.append(idx)
    return lower


def get_lower_hull_points(points: np.ndarray) -> np.ndarray:
    """
    Return all unique points that take part in facets/edges of the lower
    convex hull.

    Works for:
        (x, f(x))  -  shape (n, 2)
        (x, y, f(x, y))    shape (n, 3)
    """
    dim = points.shape[1]
    if dim not in (2, 3):
        raise ValueError("Only 2- or 3-D point clouds are supported.")

    if dim == 2:
        return points[_monotone_chain_lower(points)]

    # ── 3-D ────────────────────────────────────────────────────────────────
    hull = ConvexHull(points)

    # Each row of `equations` is (A, B, C, D) s.t. Ax + By + Cz + D = 0
    # and the normal (A, B, C) points *outwards*.
    lower_mask = hull.equations[:, 2] < 0  # C < 0  →  outward normal faces downward
    lower_vertices = np.unique(hull.simplices[lower_mask])
    return points[lower_vertices]


def identify_lower_hull_facets(
    hull: ConvexHull | None, points: np.ndarray
) -> List[Tuple[int, ...]]:
    """
    List of simplices (edges in 2-D, triangles in 3-D) that belong to
    the lower hull.
    """
    dim = points.shape[1]

    if dim == 2:
        lower = _monotone_chain_lower(points)
        return [(lower[i], lower[i + 1]) for i in range(len(lower) - 1)]

    # ── 3-D ────────────────────────────────────────────────────────────────
    if hull is None:
        hull = ConvexHull(points)
    lower_mask = hull.equations[:, 2] < 0
    return [
        tuple(simplex)
        for simplex, is_lower in zip(hull.simplices, lower_mask)
        if is_lower
    ]


def compute_plane_equation(  # ← NEW VERSION
    lower_simplices: List[Tuple[int, ...]], points: np.ndarray
) -> List[Tuple[float, ...]]:
    """
    2-D:   (alpha, a, x_ref)
    3-D:   (alpha, a, b, x_ref, y_ref)
    """
    dim = points.shape[1]
    planes: list[Tuple[float, ...]] = []

    for simplex in lower_simplices:
        pts = points[list(simplex)]

        if dim == 2:
            (x0, z0), (x1, z1) = pts
            if np.isclose(x0, x1):
                continue  # vertical – skip
            a = (z1 - z0) / (x1 - x0)
            alpha = z0 - a * x0
            x_min = min(x0, x1)  # smaller endpoint
            planes.append((alpha, a, x_min))

        else:
            (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = pts
            M = np.array([[1, x0, y0], [1, x1, y1], [1, x2, y2]], float)
            rhs = np.array([z0, z1, z2], float)
            if np.linalg.matrix_rank(M) < 3:
                continue  # degenerate
            alpha, a, b = np.linalg.solve(M, rhs)

            x_min = min(x0, x1, x2)
            y_min = min(y0, y1, y2)
            planes.append((alpha, a, b, x_min, y_min))

    # de-duplicate numerically identical planes
    unique: list[Tuple[float, ...]] = []
    for p in planes:
        if not any(np.allclose(p, q) for q in unique):
            unique.append(p)
    return unique


def lower_hull_planes(
    points: np.ndarray,
) -> List[Tuple[float, ...]]:
    """Wrapper that returns the extended tuples above."""
    dim = points.shape[1]
    hull = ConvexHull(points) if dim == 3 else None
    simplices = identify_lower_hull_facets(hull, points)
    return np.array(compute_plane_equation(simplices, points))


def tangential_planes_to_dict(
    tangential_planes: Sequence[Union[Tuple[float, np.ndarray], np.ndarray]],
    grad_names: Sequence[str],
) -> Dict[str, float]:
    """
    Brings tangential planes to dict format that can easily be saved - with separate lists for intercept, each gradient component and each point coordinates.

    Parameters:
        tangential_planes: List of (z0, grad_vector) tuples or flat arrays.
        grad_names: Names for each gradient component, e.g., ["volume_flow", "pressure_rise"]
        filename: Path to the output YAML file
    """
    intercepts = []
    grad_lists = {name: [] for name in grad_names}
    point_lists = {name: [] for name in grad_names}

    for plane in tangential_planes:
        if isinstance(plane, tuple) and len(plane) == 2:
            z0, grad = plane
        else:
            arr = np.asarray(plane)
            separator = int((arr.shape[0] - 1) / 2) + 1
            z0, grad, point = arr[0], arr[1:separator], arr[separator:]

        intercepts.append(float(z0))
        for name, val, coord in zip(grad_names, grad, point):
            grad_lists[name].append(float(val))
            point_lists[name].append(float(coord))

    # Prepare YAML dict
    yaml_dict = {"intercept": intercepts}
    for name in grad_names:
        yaml_dict[f"grad_{name}"] = grad_lists[name]
        yaml_dict[f"point_{name}"] = point_lists[name]

    return yaml_dict
