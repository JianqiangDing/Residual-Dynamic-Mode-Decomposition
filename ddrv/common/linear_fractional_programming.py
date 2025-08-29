# this function is used to solve the linear fractional programming problem

import cvxpy as cp
import numpy as np


def linear_fractional_programming(A, b, c, d, lx, ux, e=0, f=0, minimize=True):
    """Linear fractional programming solver."""
    y = cp.Variable(c.shape[0])
    t = cp.Variable(pos=True)

    numerator = c.T @ y + e * t
    denominator = d.T @ y + f * t

    if np.count_nonzero(d >= 0) == 0 and f <= 0:
        numerator *= -1
        denominator *= -1

    # set constraints
    constraints = [denominator == 1, t >= 0]

    if A is not None and b is not None:
        constraints.append(A @ y <= b * t)
    if lx is not None:
        constraints.append(lx * t <= y)
    if ux is not None:
        constraints.append(ux * t >= y)

    objective = cp.Minimize(numerator) if minimize else cp.Maximize(numerator)

    problem = cp.Problem(objective, constraints)

    problem.solve(solver=cp.MOSEK, verbose=False)

    return problem.status, problem.value, y.value, t.value
