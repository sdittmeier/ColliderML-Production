"""Independent helix and SciPy references; no ACTS propagator is used here.

Lengths are mm, momenta GeV/c, charges e and fields Tesla. The selected branch
is the local transverse-distance minimum nearest to production in |path|;
an exactly tied choice uses negative path. There is no material interaction.
"""

import numpy as np
from scipy.integrate import solve_ivp


KAPPA = 0.000299792458  # GeV / (Tesla mm)


def parameters(position, momentum):
    position, momentum = np.asarray(position, float), np.asarray(momentum, float)
    pt = np.hypot(*momentum[:2])
    if not np.isfinite(np.r_[position, momentum]).all() or pt == 0:
        raise ValueError("Finite phase space and nonzero transverse momentum required")
    return dict(d0=float((-position[0] * momentum[1] + position[1] * momentum[0]) / pt),
                z0=float(position[2]))


def uniform_state(position, momentum, charge, bz, path):
    position, momentum = np.asarray(position, float), np.asarray(momentum, float)
    p = np.linalg.norm(momentum)
    if p == 0:
        raise ValueError("Nonzero momentum required")
    omega = charge * KAPPA * bz / p
    if omega == 0:
        return position + path * momentum / p, momentum.copy()
    angle = -omega * path
    c, s = np.cos(angle), np.sin(angle)
    px, py, pz = momentum
    final_p = np.array([c * px - s * py, s * px + c * py, pz])
    q_b = charge * KAPPA * bz
    final_r = position + np.array([(py - final_p[1]) / q_b,
                                   (final_p[0] - px) / q_b, path * pz / p])
    return final_r, final_p


def analytic_perigee(position, momentum, charge, bz):
    position, momentum = np.asarray(position, float), np.asarray(momentum, float)
    p, pt = np.linalg.norm(momentum), np.hypot(*momentum[:2])
    parameters(position, momentum)
    q_b = charge * KAPPA * bz
    if q_b == 0:
        path = -p * np.dot(position[:2], momentum[:2]) / pt**2
    else:
        center = position[:2] + np.array([momentum[1], -momentum[0]]) / q_b
        radius = pt / abs(q_b)
        if np.linalg.norm(center) < 1e-12:
            raise ValueError("Concentric helix has no unique transverse-distance minimum")
        start = position[:2] - center
        end = -radius * center / np.linalg.norm(center)
        angle = np.arctan2(start[0] * end[1] - start[1] * end[0], np.dot(start, end))
        path = -angle * p / q_b
        if abs(abs(angle) - np.pi) < 1e-12:
            path = -np.pi * p / abs(q_b)
    r, m = uniform_state(position, momentum, charge, bz, path)
    return dict(**parameters(r, m), path_length_mm=float(path),
                position_mm=r.tolist(), momentum_GeV=m.tolist())


def slab_solenoid_perigee(position, momentum, charge, bz=2., zmax=3000., radial_bound=1160.):
    """Exact helix inside a uniform finite solenoid, straight line outside.

    Applicable only when the entire inner-field circle stays inside the given
    radial bound, so the radial return field cannot be encountered. This is a
    separate closed-form cross-check, not the general ODD numerical reference.
    """
    position, momentum = np.asarray(position, float), np.asarray(momentum, float)
    if abs(position[2]) >= zmax:
        raise ValueError("Slab check expects an interior production vertex")
    if charge == 0 or bz == 0 or momentum[2] == 0:
        return analytic_perigee(position, momentum, charge, bz)
    p, pt = np.linalg.norm(momentum), np.hypot(*momentum[:2])
    qb = charge * KAPPA * bz
    center = position[:2] + np.array([momentum[1], -momentum[0]]) / qb
    if np.linalg.norm(center) + pt / abs(qb) >= radial_bound:
        raise ValueError("Trajectory might encounter the radial return field")
    crossings = sorted((plane - position[2]) * p / momentum[2] for plane in (-zmax, zmax))
    exact = analytic_perigee(position, momentum, charge, bz)
    period = 2 * np.pi * p / abs(qb)
    lower = int(np.ceil((crossings[0] - exact["path_length_mm"]) / period))
    upper = int(np.floor((crossings[1] - exact["path_length_mm"]) / period))
    candidates = []
    if lower <= upper:
        for turn in {lower, upper, min(max(0, lower), upper)}:
            path = exact["path_length_mm"] + turn * period
            r, m = uniform_state(position, momentum, charge, bz, path)
            candidates.append(dict(**parameters(r, m), path_length_mm=float(path), region="inner"))
    for side, crossing in zip((-1, 1), crossings):
        r, m = uniform_state(position, momentum, charge, bz, crossing)
        outside = analytic_perigee(r, m, 0., 0.)
        if side * outside["path_length_mm"] >= 0:
            candidates.append(dict(**parameters(outside["position_mm"], outside["momentum_GeV"]),
                                   path_length_mm=crossing + outside["path_length_mm"], region="outside_z"))
    if not candidates:
        raise ValueError("No minimum found in the analytic slab trajectory")
    return min(candidates, key=lambda candidate: (abs(candidate["path_length_mm"]), candidate["path_length_mm"]))


def reference_perigee(position, momentum, charge, field, rtol=2e-9,
                       atol=1e-8, max_path=100000., step_scale=1.):
    """Integrate Lorentz motion with DOP853 and locate minima independently.

    Event roots locate x*ux+y*uy=0. Positive second derivative distinguishes
    minima from maxima. The other direction is searched up to the first
    candidate's |path|, sufficient to rule out a nearer branch. A finite search
    bound is explicit; failure to find a minimum is not converted to zero.
    """
    position, momentum = np.asarray(position, float), np.asarray(momentum, float)
    parameters(position, momentum)
    if not np.isfinite(charge) or rtol <= 0 or atol <= 0 or max_path <= 0 or step_scale <= 0:
        raise ValueError("Invalid charge or integration settings")
    initial = np.r_[position, momentum]
    magnitude = float(np.linalg.norm(momentum))
    calls = 0

    def rhs(_, state):
        nonlocal calls
        calls += 1
        p = state[3:]
        norm = np.linalg.norm(p)
        b = np.asarray(field(state[:3]), float)
        if b.shape != (3,) or not np.isfinite(b).all():
            raise ValueError("Invalid field lookup")
        return np.r_[p / norm, KAPPA * charge * np.cross(p, b) / norm]

    def gradient(state):
        return float(np.dot(state[:2], state[3:5]) / np.linalg.norm(state[3:]))

    def curvature(state):
        velocity = state[3:] / np.linalg.norm(state[3:])
        acceleration = rhs(0., state)[3:] / np.linalg.norm(state[3:])
        return float(np.dot(velocity[:2], velocity[:2]) + np.dot(state[:2], acceleration[:2]))

    g0 = gradient(initial)
    if abs(g0) < 1e-12 and curvature(initial) > 0:
        return dict(status="ok", **parameters(position, momentum), path_length_mm=0.,
                    curvature=curvature(initial), field_calls=calls, momentum_drift=0.,
                    search_bound_mm=0.)
    b0 = np.linalg.norm(field(position))
    half_period = np.pi * magnitude / (KAPPA * abs(charge) * b0) if charge and b0 else max_path
    straight_path = -magnitude * np.dot(position[:2], momentum[:2]) / np.dot(momentum[:2], momentum[:2])
    bound = min(max_path, max(half_period * (1 + 1e-8) + 1e-6, 2 * abs(straight_path) + 10.))
    max_step = step_scale * min(50., half_period / 32.)
    preferred = -1 if g0 > 0 else 1
    candidates = []
    failures = []
    drift = 0.
    for direction in (preferred, -preferred):
        limit = min(bound, abs(candidates[0][0]) + 1e-6) if candidates else bound

        def minimum(_, state):
            return gradient(state)

        minimum.terminal = True
        minimum.direction = direction
        solution = solve_ivp(rhs, (0., direction * limit), initial,
                             method="DOP853", events=minimum, rtol=rtol, atol=atol,
                             max_step=max_step)
        drift = max(drift, float(np.max(np.abs(np.linalg.norm(solution.y[3:], axis=0) / magnitude - 1))))
        if not solution.success:
            failures.append(solution.message)
        if len(solution.t_events[0]):
            path = float(solution.t_events[0][0])
            state = solution.y_events[0][0]
            second = curvature(state)
            if second > 0:
                candidates.append((path, state, second))
    if not candidates:
        return dict(status="no_minimum_within_bound", search_bound_mm=bound,
                    failures=failures, field_calls=calls, momentum_drift=drift)
    shortest = min(abs(value[0]) for value in candidates)
    tied = [value for value in candidates if abs(value[0]) <= shortest + 1e-7 + 1e-8 * shortest]
    path, state, second = min(tied, key=lambda value: value[0])
    return dict(status="ok", **parameters(state[:3], state[3:]), path_length_mm=path,
                curvature=second, stationarity_mm=gradient(state),
                field_calls=calls, momentum_drift=drift, search_bound_mm=bound)
