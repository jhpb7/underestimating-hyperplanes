import numpy as np
import sympy as sp


def filter_with_upper_limit(arr: list[float], upper_limit: float) -> np.ndarray:
    """
    Returns values from the input list that are less than or equal to the upper limit,
    plus the first value above the limit (if any), to ensure continuity.

    Parameters:
        arr (list[float]): Input sequence of values.
        upper_limit (float): The maximum allowed value.

    Returns:
        np.ndarray: Filtered array.
    """
    result = []
    for val in arr:
        result.append(val)
        if val > upper_limit:
            break
    return np.array([x for x in result if x <= upper_limit or x == val])


def get_pel_n_func(
    a: dict[int, float], b: dict[int, float]
) -> tuple[callable, callable, callable]:
    """
    Generates the pel function, its gradient, and the n function symbolically using SymPy.

    Parameters:
        a (dict[int, float]): Coefficients for dp = a₁*q² + a₂*q*n + a₃*n².
        b (dict[int, float]): Coefficients for pel = b₁*q³ + b₂*q²*n + b₃*q*n² + b₄*n³ + b₅.

    Returns:
        tuple:
            pel_func (callable): Function f(q, dp) → pel
            pel_func_grad (callable): Function f(q, dp) → [∂pel/∂q, ∂pel/∂dp]
            n_func (callable): Function f(q, dp) → n
    """
    pel, q, dp, n = sp.symbols("pel q dp n")

    H = sp.Eq(dp, a[1] * q**2 + a[2] * q * n + a[3] * n**2)
    pel_expr = b[1] * q**3 + b[2] * q**2 * n + b[3] * q * n**2 + b[4] * n**3 + b[5]

    n_eq = sp.solve(H, n)[1]
    pel_eq = pel_expr.subs(n, n_eq) - q * dp

    pel_func = sp.lambdify([q, dp], pel_eq, modules="numpy")
    # List of partial derivatives
    partials = [sp.diff(pel_eq, var) for var in (q, dp)]

    # Lambdify each partial into its own callable
    pel_func_grad = [sp.lambdify([q, dp], expr, modules="numpy") for expr in partials]
    n_func = sp.lambdify([q, dp], n_eq, modules="numpy")
    return pel_func, pel_func_grad, n_func


def calculate_max_values(
    a: dict[int, float], b: dict[int, float]
) -> tuple[float, float, float]:
    """
    Calculates Qmax, dpmax, and Pmax based on given coefficient dictionaries.

    Parameters:
        a (dict[int, float]): Coefficients for dp = a1*q² + a2*q*n + a3*n².
        b (dict[int, float]): Coefficients for
                    ploss = b1*q³ + b2*q² + b3*q + b4 + b5 - qdp.
                          = (b1-a1)q³ + (b2-a2)q² + (b3-a3)q + b4 + b5

    Returns:
        tuple:
            Qmax (float): Maximum flow rate.
            dpmax (float): Maximum pressure difference.
            Pmax (float): Maximum power.
    """
    Qmax = float((-a[2] - np.sqrt(a[2] ** 2 - 4 * a[3] * a[1])) / (2 * a[1]))
    dpmax = float(a[3] - a[2] ** 2 / (4 * a[1]))
    if np.sign(b[2] - a[2]) * np.sign(b[1] - a[1]) >= 0:
        q_s_P = (
            1
            / (3 * (b[1] - a[1]))
            * (
                -(b[2] - a[2])
                + np.sqrt((b[2] - a[2]) ** 2 - 3 * (b[1] - a[1]) * (b[3] - a[3]))
            )
        )
    else:
        q_s_P = (
            1
            / (3 * (b[1] - a[1]))
            * (
                -(b[2] - a[2])
                - np.sqrt((b[2] - a[2]) ** 2 - 3 * (b[1] - a[1]) * (b[3] - a[3]))
            )
        )

    Pmax = float(
        (b[1] - a[1]) * q_s_P**3
        + (b[2] - a[2]) * q_s_P**2
        + (b[3] - a[3]) * q_s_P
        + b[4]
        + b[5]
    )
    return {"q": Qmax, "dp": dpmax, "ploss": Pmax}
