"""
Online learners for signal cycle parameters.

Both learners implement the same interface::

    model = learner.update(transition)   # online update -> PeriodModel
    model = learner.get_model()
    learner.reset()

They consume PhaseTransition events and produce PeriodModel estimates in the
unified phase convention (``phi_offset`` = start of a red phase; see
``core.models``).

Estimation notes
----------------
Observed transitions are almost always RED->GREEN (a waiting robot leaves
shortly after green appears).  From RG transition times alone, only
``phi_offset + T_red`` is identifiable.  To separate the two, the learners
additionally use the waiting time of each episode: for a robot that arrived
at ``episode_start`` during red and saw green at ``tau``, the remaining red
``tau - episode_start`` is a sample of a uniform(0, T_red) variable, so
``2 * mean(remaining)`` is an unbiased estimate of T_red.

Priors are generic and honest: they must NOT be seeded from ground-truth
signal configurations.

Shared-green constraint (field observation): the GREEN duration of a signal
is constant across time-of-day periods; only T_red (and hence T_cycle)
varies.  The pipeline therefore pools each intersection's per-period green
estimates ``T_cycle - T_red`` into one shared T_green and applies it back:

    T_red(period) = T_cycle(period) - T_green

This lets data-scarce periods (e.g. night) inherit a precise red duration
from the accurately estimated cycle plus the pooled green.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np

from crossphase_miner.core.models import PeriodModel, PhaseTransition, SignalColor
from crossphase_miner.utils.math_utils import (
    gaussian_product,
    ukf_sigma_points,
    ukf_weights,
)


def _is_rg(t: PhaseTransition) -> bool:
    return t.from_color == SignalColor.RED and t.to_color == SignalColor.GREEN


def _remaining_red(t: PhaseTransition) -> Optional[float]:
    """Remaining red time sampled from this transition's episode, if known."""
    if not _is_rg(t) or t.episode_start is None:
        return None
    r = t.timestamp - t.episode_start
    return r if r > 0 else None


class BayesianPeriodLearner:
    """
    Online Bayesian estimation of signal cycle parameters.

    Maintains Gaussian posteriors over T_cycle and T_red plus a phase
    offset derived from the most recent RED->GREEN transition.

    Attributes:
        mu_T_cycle, sigma_T_cycle: Posterior over the cycle duration (s).
        mu_T_red, sigma_T_red: Posterior over the red phase duration (s).
        phi_offset: Estimated cycle origin (start of red), absolute time.
    """

    def __init__(self, prior_T_cycle: Tuple[float, float] = (120.0, 40.0)) -> None:
        """
        Args:
            prior_T_cycle: (mean, std) of the generic T_cycle prior.
        """
        self.prior_T_cycle: Tuple[float, float] = (float(prior_T_cycle[0]), float(prior_T_cycle[1]))
        self.mu_T_cycle: float = float(prior_T_cycle[0])
        self.sigma_T_cycle: float = float(prior_T_cycle[1])
        self.mu_T_red: float = float(prior_T_cycle[0]) / 2.0
        self.sigma_T_red: float = 30.0
        self.phi_offset: float = 0.0
        self._rg_times: Deque[float] = deque(maxlen=50)
        self._red_samples: Deque[float] = deque(maxlen=50)
        # Shared-green constraint (T_green is constant across TOD periods):
        # when the pipeline applies a fleet-wide T_green estimate, the output
        # model's T_red is T_cycle - T_green instead of the local wait-sample
        # estimate.  The local estimate (mu_T_red) is still maintained and
        # used as this period's vote for the shared T_green.
        self._shared_T_red: Optional[float] = None
        self.sample_count: int = 0

    def update(self, transition: PhaseTransition) -> PeriodModel:
        """Update the posterior with a new transition; return the model."""
        self.sample_count += 1

        if _is_rg(transition):
            self._rg_times.append(transition.timestamp)
            r = _remaining_red(transition)
            if r is not None:
                self._red_samples.append(r)

        if len(self._rg_times) >= 2:
            self._estimate_T_cycle()
        if self._red_samples:
            self._estimate_T_red()
        if self._rg_times:
            self._update_phi_offset()

        return self.get_model()

    def _estimate_T_cycle(self) -> None:
        """
        Estimate T_cycle from sparse RED->GREEN transition times.

        Robots do not observe every cycle, so pairwise differences are
        integer multiples of the true period.  Three cues are combined:

        - inlier count of pairwise differences (integer-multiple fit),
        - a soft preference for the current posterior mean (tie-breaker),
        - a hard lower bound from observed waiting times (a wait of W
          seconds rules out any period below W), which resolves the
          sub-harmonic ambiguity (T/2, T/3, ... fit the RG differences
          exactly as well as T does).

        The search range is fixed (NOT tied to the current posterior), and
        the posterior is recomputed from scratch (prior + full-data
        estimate) on every update, so an early wrong estimate can neither
        lock the search range nor freeze the posterior.
        """
        rg = np.array(sorted(self._rg_times))
        diffs = np.array(
            [
                rg[j] - rg[i]
                for i in range(len(rg))
                for j in range(i + 1, len(rg))
                if rg[j] - rg[i] > 30.0
            ]
        )
        if len(diffs) == 0:
            return

        prior = self.mu_T_cycle
        candidates = np.linspace(30.0, 500.0, 941)
        n_diffs = len(diffs)

        # Hard lower bound from waiting times: a robot that waited W seconds
        # for green rules out any period (and red phase) shorter than W.
        # This resolves the sub-harmonic ambiguity (T/2, T/3, ... fit the
        # RG time differences exactly as well as T does).
        if self._red_samples:
            min_T = max(self._red_samples) + 5.0
            candidates = candidates[candidates >= min_T]
        if len(candidates) == 0:
            return

        best_T = prior
        best_score = -np.inf

        for T in candidates:
            k = np.maximum(np.round(diffs / T).astype(int), 1)
            errors = np.abs(diffs - k * T)
            tolerance = 0.05 * T
            inlier_mask = errors < tolerance
            # Soft inlier score: a candidate is rewarded by how EXACTLY the
            # differences match integer multiples, not just by the count of
            # passable fits.  This keeps a near-prior "almost divisor" from
            # beating the true period when both pass the tolerance.
            fit_score = float(np.sum(1.0 - errors[inlier_mask] / tolerance))
            # Prefer candidates near the prior (soft tie-breaker only).
            # Deviating far from the prior needs proportionally strong
            # evidence: with few, possibly noisy differences, a spurious
            # exact fit at a far-off period must not win.
            prior_penalty = 1.0 * n_diffs * abs(T - prior) / prior
            # Phase consistency: resultant length of rg phases mod T in [0,1].
            angles = 2.0 * np.pi * (rg % T) / T
            R = float(np.abs(np.mean(np.exp(1j * angles))))
            phase_penalty = n_diffs * (1.0 - R)
            score = fit_score - prior_penalty - phase_penalty
            if score > best_score:
                best_score = score
                best_T = T

        # Refine with the inlier pairs of the best candidate.
        k = np.maximum(np.round(diffs / best_T).astype(int), 1)
        errors = np.abs(diffs - k * best_T)
        inlier_mask = errors < 0.05 * best_T
        n_inliers = int(np.sum(inlier_mask))
        if n_inliers == 0:
            return
        refined_T = float(np.mean(diffs[inlier_mask] / k[inlier_mask]))
        if not np.isfinite(refined_T) or refined_T <= 0:
            return

        # Recompute the posterior from scratch (prior + full-data estimate)
        # so early ambiguous updates are overridden as data accumulates.
        # The likelihood tightens with the number of inlier pairs (their
        # mean has error ~1/sqrt(n)); with many pairs the data fully
        # dominates the prior - sub-second T_cycle accuracy is needed to
        # keep phase drift small over hundreds of cycles.
        obs_sigma = max(2.0, 40.0 / n_inliers)
        self.mu_T_cycle, self.sigma_T_cycle = gaussian_product(
            self.prior_T_cycle[0], self.prior_T_cycle[1], refined_T, obs_sigma
        )
        self.mu_T_cycle = float(np.clip(self.mu_T_cycle, 30.0, 500.0))
        self.sigma_T_cycle = max(self.sigma_T_cycle, 2.0)

    def _estimate_T_red(self) -> None:
        """
        Estimate T_red from per-episode remaining-red samples.

        For an arrival during red at ``a`` with green observed at ``tau``,
        ``tau - a ~ Uniform(0, T_red)`` (uniform arrival assumption).  The
        maximum-likelihood estimate for a uniform upper bound is the largest
        sample; ``max * (n+1)/n`` is the standard unbiased correction.  The
        posterior is recomputed from scratch against a weak prior
        (red fraction ~65% of the cycle).
        """
        samples = np.array(self._red_samples)
        n = len(samples)
        est = float(np.max(samples)) * (n + 1.0) / n
        est = float(np.clip(est, 5.0, self.mu_T_cycle - 5.0))
        # Spread of the estimator grows with T_red and shrinks with n.
        obs_sigma = max(10.0, est / np.sqrt(n))

        prior_mu = 0.65 * self.mu_T_cycle
        prior_sigma = 0.3 * self.mu_T_cycle
        self.mu_T_red, self.sigma_T_red = gaussian_product(prior_mu, prior_sigma, est, obs_sigma)
        self.mu_T_red = float(np.clip(self.mu_T_red, 5.0, self.mu_T_cycle - 5.0))
        self.sigma_T_red = max(self.sigma_T_red, 1.0)

    def _update_phi_offset(self) -> None:
        """
        Update the cycle origin from the most recent RED->GREEN transition.

        An RG transition occurs at phase position T_red, so the cycle origin
        is ``t_rg - T_red``.
        """
        last_rg = float(self._rg_times[-1])
        self.phi_offset = last_rg - self.mu_T_red

    def apply_shared_green(self, T_green: float) -> None:
        """
        Apply the shared-green constraint.

        Field observation: the GREEN duration of a signal is constant across
        time-of-day periods; only the RED duration (and thus the cycle)
        changes.  With a fleet-wide T_green estimate, this period's red
        duration follows from the (much more accurately estimated) cycle:

            T_red = T_cycle - T_green

        Args:
            T_green: Shared green duration estimate (seconds).
        """
        self._shared_T_red = float(np.clip(self.mu_T_cycle - T_green, 5.0, self.mu_T_cycle - 5.0))

    def green_estimate(self) -> Tuple[float, float]:
        """
        This period's vote for the shared T_green: (estimate, variance).

        Computed from the learner's OWN cycle and red estimates, before any
        shared value is applied, so pooling never becomes circular.
        """
        green = self.mu_T_cycle - self.mu_T_red
        variance = self.sigma_T_cycle**2 + self.sigma_T_red**2
        return float(green), float(variance)

    def get_model(self) -> PeriodModel:
        """Return the current PeriodModel estimate."""
        confidence = 1.0 - np.exp(-self.sample_count / 5.0)
        confidence *= np.exp(-self.sigma_T_cycle / 30.0)
        confidence = float(np.clip(confidence, 0.0, 1.0))

        T_red = self._shared_T_red if self._shared_T_red is not None else self.mu_T_red
        # The latest RG transition marks phase position T_red.
        phi_offset = float(self._rg_times[-1]) - T_red if self._rg_times else self.phi_offset

        return PeriodModel(
            T_cycle=float(self.mu_T_cycle),
            T_red=float(T_red),
            T_green=max(0.0, self.mu_T_cycle - T_red),
            phi_offset=float(phi_offset),
            confidence=confidence,
            sample_count=self.sample_count,
            last_updated=time.time(),
        )

    def reset(self) -> None:
        """Reset all state to the generic prior."""
        self.mu_T_cycle = 120.0
        self.sigma_T_cycle = 40.0
        self.mu_T_red = 60.0
        self.sigma_T_red = 30.0
        self.phi_offset = 0.0
        self._rg_times.clear()
        self._red_samples.clear()
        self._shared_T_red = None
        self.sample_count = 0


class UKFPeriodLearner:
    """
    Unscented Kalman Filter for online cycle estimation.

    State vector: ``x = [T_cycle, T_red, phi_offset]^T`` (unified phase
    convention: phi_offset marks the start of a red phase).

    Measurements:
      - RG transition time:  h(x, k) = phi_offset + T_red + k * T_cycle
        (non-linear, handled with sigma points; k is the integer cycle index)
      - T_red pseudo-measurement from episode waiting time (linear, exact
        scalar Kalman update on the T_red coordinate)

    Process model: random walk with adaptive process noise.
    """

    def __init__(
        self,
        initial_state: Optional[np.ndarray] = None,
        process_noise: float = 1.0,
    ) -> None:
        """
        Args:
            initial_state: Optional [T_cycle, T_red, phi_offset].  Defaults to
                the generic prior [120, 60, 0].
            process_noise: Base process noise for the random walk.  Must be
                large enough to let T_cycle travel from the generic prior to
                the true value within a handful of transitions.
        """
        self.x: np.ndarray = (
            np.array(initial_state, dtype=float).flatten()
            if initial_state is not None
            else np.array([120.0, 60.0, 0.0])
        )
        if len(self.x) != 3:
            raise ValueError("initial_state must be a 3-element array")

        self.n: int = 3
        # phi_offset is essentially unknown before the first transition.
        self.P: np.ndarray = np.diag([1600.0, 900.0, 1e8])
        self.base_Q: np.ndarray = np.eye(self.n) * process_noise
        self.Q: np.ndarray = self.base_Q.copy()
        self.R_time: float = 5.0  # RG transition time noise (s)
        self.R_tred: float = 400.0  # T_red pseudo-measurement variance (s^2)
        self.kappa: float = 0.0
        self.alpha: float = 1.0
        self.beta: float = 2.0
        self.sample_count: int = 0
        self._recent_errors: Deque[float] = deque(maxlen=20)
        self._phi_initialized: bool = False
        # Longest red wait seen so far: a wait of W seconds rules out any
        # period below W, which resolves the sub-harmonic aliasing of the
        # cycle-index computation (T/2, T/3, ... fit RG times as well as T).
        self._max_wait: float = 0.0

    # -- Update ------------------------------------------------------------

    def update(self, transition: PhaseTransition) -> PeriodModel:
        """Update the UKF state with a new transition; return the model."""
        self.sample_count += 1

        if _is_rg(transition):
            if not self._phi_initialized:
                # Anchor the phase: this RG happened at phase position T_red.
                self.x[2] = transition.timestamp - self.x[1]
                self.P[2, 2] = 100.0
                self._phi_initialized = True
            self._time_update(transition.timestamp)

            r = _remaining_red(transition)
            if r is not None:
                self._max_wait = max(self._max_wait, r)
                self._tred_update(r)

        self._adapt_process_noise()
        return self.get_model()

    def _time_update(self, t_obs: float) -> None:
        """UKF update with an RG transition time measurement."""
        # Prediction: random walk.
        self.P = self.P + self.Q
        self._stabilize_P()

        T_cycle = self.x[0] if self.x[0] > 0 else 120.0
        k = max(0, int(np.round((t_obs - (self.x[2] + self.x[1])) / T_cycle)))

        sigma_pts = ukf_sigma_points(self.x, self.P, kappa=self.kappa)
        Z_sigma = np.array([sp[2] + sp[1] + k * sp[0] for sp in sigma_pts])
        Wm, Wc, _ = ukf_weights(self.n, kappa=self.kappa, alpha=self.alpha, beta=self.beta)

        z_pred = float(np.dot(Wm, Z_sigma))
        S = float(np.dot(Wc, (Z_sigma - z_pred) ** 2)) + self.R_time

        P_xz = np.zeros(self.n)
        for i in range(len(sigma_pts)):
            P_xz += Wc[i] * (sigma_pts[i] - self.x) * (Z_sigma[i] - z_pred)

        if abs(S) < 1e-10:
            S = 1e-10
        K = P_xz / S

        innovation = t_obs - z_pred
        self._recent_errors.append(abs(innovation))

        self.x = self.x + K * innovation
        self.P = self.P - np.outer(K, P_xz)
        self._stabilize_P()
        self._clamp_state()

    def _tred_update(self, remaining_red: float) -> None:
        """
        Scalar Kalman update on T_red from an episode waiting sample.

        The measurement is ``z = 2 * remaining_red`` (unbiased under the
        uniform-arrival assumption) with ``h(x) = x[1]``.
        """
        z = float(np.clip(2.0 * remaining_red, 5.0, self.x[0] - 5.0))
        H = np.array([0.0, 1.0, 0.0])

        S = float(H @ self.P @ H) + self.R_tred
        K = (self.P @ H) / S
        self.x = self.x + K * (z - float(H @ self.x))
        self.P = self.P - np.outer(K, H @ self.P)
        self._stabilize_P()
        self._clamp_state()

    # -- Numerical helpers ---------------------------------------------------

    def _stabilize_P(self) -> None:
        """Symmetrize P and keep it positive semi-definite."""
        self.P = 0.5 * (self.P + self.P.T)
        eigvals = np.linalg.eigvalsh(self.P)
        if np.any(eigvals < 0):
            self.P += np.eye(self.n) * (abs(float(np.min(eigvals))) + 1e-6)

    def _clamp_state(self) -> None:
        """Keep the state within physically meaningful bounds."""
        min_T = max(30.0, self._max_wait + 5.0)
        self.x[0] = np.clip(self.x[0], min_T, 500.0)
        self.x[1] = np.clip(self.x[1], 5.0, self.x[0] - 5.0)

    def _adapt_process_noise(self) -> None:
        """Raise process noise when recent prediction errors are large."""
        if len(self._recent_errors) < 3:
            return
        recent_rmse = float(np.sqrt(np.mean(np.array(self._recent_errors) ** 2)))
        if recent_rmse > 15.0:
            scale_factor = min(recent_rmse / 10.0, 10.0)
            self.Q = self.base_Q * scale_factor
        else:
            self.Q = self.base_Q.copy()

    # -- Model ---------------------------------------------------------------

    def get_model(self) -> PeriodModel:
        """Return the current PeriodModel estimate."""
        trace_P = float(np.trace(self.P))
        confidence = np.exp(-trace_P / 2000.0)
        confidence *= 1.0 - np.exp(-self.sample_count / 10.0)
        confidence = float(np.clip(confidence, 0.0, 1.0))

        T_cycle = float(self.x[0])
        T_red = float(self.x[1])
        return PeriodModel(
            T_cycle=T_cycle,
            T_red=T_red,
            T_green=max(0.0, T_cycle - T_red),
            phi_offset=float(self.x[2]),
            confidence=confidence,
            sample_count=self.sample_count,
            last_updated=time.time(),
        )

    def reset(self) -> None:
        """Reset the filter to the generic prior."""
        self.x = np.array([120.0, 60.0, 0.0])
        self.P = np.diag([1600.0, 900.0, 1e8])
        self.Q = self.base_Q.copy()
        self.sample_count = 0
        self._recent_errors.clear()
        self._phi_initialized = False
        self._max_wait = 0.0
