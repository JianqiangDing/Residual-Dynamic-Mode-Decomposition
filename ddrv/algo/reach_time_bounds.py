# this function is used to compute the reach-time bounds with given eigenvalues and evaluations of modes

import numpy as np
import portion as P

from ddrv.common import linear_fractional_programming


def _reach_time_bounds_with_magnitude(ef0_vals, efF_vals, lams, itol=1e-3):
    """Compute reach time bounds using magnitude."""
    mag0, magF = np.log(np.abs(ef0_vals)), np.log(np.abs(efF_vals))

    mag0_inf, mag0_sup = np.min(mag0, axis=0), np.max(mag0, axis=0)
    magF_inf, magF_sup = np.min(magF, axis=0), np.max(magF, axis=0)

    lams_real = np.real(lams)

    # determine the time bounds in the case of positive eigenfunctions
    ## If lambda > 0, unstable
    ### lower bound
    A = -lams_real
    b = np.zeros(1)

    c = magF_inf - mag0_sup
    d = lams_real

    _, bound_lower_pos_unstable, _, _ = linear_fractional_programming(
        A, b, c, d, 0, 1, 0, 0, minimize=False
    )

    ### upper bound
    c = magF_sup - mag0_inf

    _, bound_upper_pos_unstable, _, _ = linear_fractional_programming(
        A, b, c, d, 0, 1, 0, 0, minimize=True
    )

    bound_pos_unstable = P.closed(bound_lower_pos_unstable, bound_upper_pos_unstable)

    ## If lambda < 0, stable
    ### lower bound
    A = lams_real
    c = magF_sup - mag0_inf

    _, bound_lower_pos_stable, _, _ = linear_fractional_programming(
        A, b, c, d, 0, 1, 0, 0, minimize=False
    )

    ### upper bound
    c = magF_inf - mag0_sup

    _, bound_upper_pos_stable, _, _ = linear_fractional_programming(
        A, b, c, d, 0, 1, 0, 0, minimize=True
    )

    bound_pos_stable = P.closed(bound_lower_pos_stable, bound_upper_pos_stable)

    return bound_pos_unstable & bound_pos_stable


def _reach_time_bounds_with_imaginary(ef0_vals, efF_vals, lams, itol=1e-3):
    """Compute reach time bounds using imaginary part."""

    def __get_bound(s_, y_, t_, lams_, diff_min, diff_max):
        bound_lower_, bound_upper_, img_opt = -np.inf, np.inf, None
        if s_ == "optimal":
            alpha = y_ / t_
            img_opt = np.dot(alpha, lams_)

            if img_opt != 0:
                bound_lower_ = np.dot(alpha, diff_min) / img_opt
                bound_upper_ = np.dot(alpha, diff_max) / img_opt

                bound_lower_, bound_upper_ = np.minimum(
                    bound_lower_, bound_upper_
                ), np.maximum(bound_lower_, bound_upper_)

        return P.closed(bound_lower_, bound_upper_), img_opt

    ang0, angF = np.angle(ef0_vals), np.angle(efF_vals)

    # correct the angle
    def correct_angle(a):
        a_new = np.zeros_like(a)
        diff = np.abs(np.max(a, axis=0) - np.min(a, axis=0))

        for i in range(diff.shape[0]):
            if diff[i] > 0.5 * np.pi:
                a_new[:, i] = a[:, i]
                a_new[a_new[:, i] < 0, i] += np.pi * 2
            else:
                a_new[:, i] = a[:, i]
        return a_new

    ang0 = correct_angle(ang0)
    angF = correct_angle(angF)

    ang0_inf, ang0_sup = np.min(ang0, axis=0), np.max(ang0, axis=0)
    angF_inf, angF_sup = np.min(angF, axis=0), np.max(angF, axis=0)

    ang_diff_min = angF_inf - ang0_sup
    ang_diff_max = angF_sup - ang0_inf

    lams_img = np.imag(lams)

    # determine the time bounds in the case of λ > 0
    A = -lams_img
    b = np.zeros(1)
    c = angF_sup - ang0_inf - angF_inf + ang0_sup
    d = lams_img

    status, _, y, t = linear_fractional_programming(
        A, b, c, d, 0, 1, 0, 0, minimize=True
    )

    bound_pos, img_pos = __get_bound(status, y, t, lams_img, ang_diff_min, ang_diff_max)

    # determine the time bounds in the case of λ < 0
    A = lams_img
    c = ang0_sup - angF_inf - ang0_inf + angF_sup
    d = -lams_img

    status, _, y, t = linear_fractional_programming(
        A, b, c, d, 0, 1, 0, 0, minimize=True
    )

    bound_neg, img_neg = __get_bound(status, y, t, lams_img, ang_diff_max, ang_diff_min)

    # return bound_pos & bound_neg, img_pos, img_neg
    assert np.allclose(img_pos, abs(img_neg))
    return bound_pos, img_pos, img_neg


def compute_reach_time_bounds(ef0_vals, efF_vals, lams, itol=1e-3):
    """Compute reach-time bounds using KRTB method.

    Args:
        ef0_vals: the evaluations of the eigenfunctions at the initial set
        efF_vals: the evaluations of the eigenfunctions at the target set
        lams: the eigenvalues, which is the continuous eigenvalues
        itol: the tolerance for the reach-time bounds for merging close bounds

    Returns:
        bounds: the reach-time bounds
        status: the status of the reachability verification

    """

    bound_mag = _reach_time_bounds_with_magnitude(ef0_vals, efF_vals, lams)

    print(bound_mag, "bound_mag")

    # if all eigenvalues are real, then we skip the imaginary part
    if np.all(np.isreal(lams)):
        if (
            bound_mag.empty
            or bound_mag.upper < 0
            or np.all(
                np.isinf(
                    np.array(
                        [
                            bound_mag.lower,
                            bound_mag.upper,
                        ]
                    )
                )
            )
        ):
            return [], ("UNREACHABLE", 0)
        else:

            return [(bound_mag.lower, bound_mag.upper)], ("PROBABLY REACHABLE", 1)

    bound_img, img_pos, img_neg = _reach_time_bounds_with_imaginary(
        ef0_vals, efF_vals, lams
    )

    if (
        bound_mag.empty
        or bound_mag.upper < 0
        or np.all(
            np.isinf(
                np.array(
                    [bound_mag.lower, bound_mag.upper, bound_img.lower, bound_img.upper]
                )
            )
        )
    ):
        return [], ("UNREACHABLE", 0)

    if img_pos is None:
        assert img_neg is None
        return [(bound_mag.lower, bound_mag.upper)], ("PROBABLY REACHABLE", 1)

    period = (2 * np.pi) / abs(img_pos) if img_pos != 0 else np.inf

    if np.isposinf(float(bound_mag.upper)):
        if bound_img.lower < 0:
            min_shift = np.ceil(-float(bound_img.lower) / period) * period
            return [(bound_img.lower + min_shift, bound_img.upper + min_shift)], (
                "PROBABLY REACHABLE",
                period,
            )

    bounds = []

    if bound_mag.lower < 0:
        bound_mag = bound_mag.replace(lower=0)

    min_shift = (
        np.ceil((float(bound_mag.lower) - float(bound_img.upper)) / period) * period
    )
    assert min_shift >= 0

    bound_wind = P.closed(bound_img.lower + min_shift, bound_img.upper + min_shift)

    i = 0
    while True:
        this_bound = P.closed(
            bound_wind.lower + i * period, bound_wind.upper + i * period
        )

        this_intersection = this_bound & bound_mag

        if this_intersection.empty:
            break

        if len(bounds) >= 1:
            if abs(float(bounds[-1][1]) - float(this_intersection.lower)) <= itol:
                bounds[-1] = (bounds[-1][0], this_intersection.upper)
                i = i + 1
                continue

        bounds.append((this_intersection.lower, this_intersection.upper))
        i = i + 1

    if len(bounds) == 0:
        return [], ("UNREACHABLE", 0)

    return bounds, ("PROBABLY REACHABLE", len(bounds))
