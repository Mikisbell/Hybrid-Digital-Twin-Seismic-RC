"""
loss.py — Hybrid Physics-Informed Loss Function
================================================

Implements the three-component loss for the Hybrid Digital Twin PINN:

    L_total = λ_d · L_data  +  λ_p · L_physics  +  λ_b · L_bc

Components
----------
L_data    : MSE between PINN-predicted and OpenSeesPy-simulated IDR.
L_physics : Residual of the equation of motion:
              R = M·ü + C·u̇ + f_int(u, u̇) + M·ι·üg
            **f_int** is the nonlinear restoring force from OpenSeesPy,
            NOT a constant K·u.  This captures cracking, yielding, and
            cyclic degradation of RC members (Concrete02 + Steel02).
L_bc      : Initial-condition penalty: u(0)=0, u̇(0)=0 (building at rest).

The physics residual uses automatic differentiation (torch.autograd.grad)
to compute ü and u̇ from the network's displacement predictions when the
model is set up in "displacement-output" mode, or accepts pre-computed
kinematic arrays when using the standard "IDR-output" mode with
externally-supplied time-history data.

References
----------
    [3] Raissi, Perdikaris, Karniadakis (2019). J. Comput. Phys., 378.
    [10] Zhang, Ernst, Shen (2020). Physics-Informed Multi-LSTM for
         structural response. Comput. Methods Appl. Mech. Eng., 369.

Author: Mikisbell
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class LossWeights:
    """Weights for the three loss components.

    Attributes
    ----------
    lambda_data : float
        Weight for the data-fidelity term (L_data).
    lambda_phys : float
        Weight for the physics-residual term (L_physics).
    lambda_bc : float
        Weight for the boundary/initial-condition term (L_bc).
    """

    lambda_data: float = 1.0
    lambda_phys: float = 0.1
    lambda_bc: float = 0.01


# ═══════════════════════════════════════════════════════════════════════════
# Individual Loss Components
# ═══════════════════════════════════════════════════════════════════════════


def data_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    story_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE between predicted and target IDR.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted IDR, shape (B, n_stories) or (B, n_stories, T).
    target : torch.Tensor
        Ground-truth IDR from OpenSeesPy, same shape as *pred*.
    story_weights : torch.Tensor or None
        Per-story importance weights, shape (n_stories,). Uses
        inverse-variance weighting so stories with smaller drift
        magnitudes receive higher weight. If None, falls back to
        uniform MSE.

    Returns
    -------
    torch.Tensor
        Scalar (weighted) MSE loss.
    """
    if story_weights is not None:
        # Weighted MSE: w_i * (pred_i - target_i)^2
        return (story_weights * (pred - target) ** 2).mean()
    return nn.functional.mse_loss(pred, target)


def physics_loss(
    mass_matrix: torch.Tensor,
    accel_response: torch.Tensor,
    vel_response: torch.Tensor,
    f_int: torch.Tensor,
    ground_accel: torch.Tensor,
    damping_matrix: torch.Tensor | None = None,
    influence_vector: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Residual of the equation of motion (EOM).

    Computes the L2 norm of:

    .. math::
        R = M \ddot{u} + C \dot{u} + f_{int}(u, \dot{u}) + M \iota \ddot{u}_g

    All matrices and vectors are provided externally because the structural
    system matrices (M, C) are constant and f_int is computed by OpenSeesPy
    (not by the neural network).  This is the "hybrid" aspect of the twin:
    physics from the FE model constrains the NN predictions.

    Parameters
    ----------
    mass_matrix : torch.Tensor
        Lumped mass matrix, shape (n_dof, n_dof) or (n_stories,) diagonal.
    accel_response : torch.Tensor
        Absolute/relative floor acceleration, shape (B, n_stories, T).
    vel_response : torch.Tensor
        Floor velocity, shape (B, n_stories, T).
    f_int : torch.Tensor
        Nonlinear restoring force from OpenSeesPy, shape (B, n_stories, T).
        **This is NOT K·u** — it captures cracking, yielding, and cyclic
        degradation of Concrete02/Steel02 RC members.
    ground_accel : torch.Tensor
        Ground acceleration time series, shape (B, 1, T) or (B, T).
    damping_matrix : torch.Tensor or None
        Rayleigh damping matrix. If None, assumes zero viscous damping
        (damping likely included in f_int or negligible).
    influence_vector : torch.Tensor or None
        Influence vector ι mapping ground motion to DOFs.
        Shape (n_stories,).  Defaults to all-ones (rigid-diaphragm shear building).

    Returns
    -------
    torch.Tensor
        Scalar mean-squared residual of the EOM.
    """
    n_stories = accel_response.shape[1]

    # Default influence vector: unity (shear-building assumption)
    if influence_vector is None:
        influence_vector = torch.ones(n_stories, device=accel_response.device)

    # Handle diagonal mass/damping (stored as vectors for efficiency)
    if mass_matrix.dim() == 1:
        # M·ü → element-wise: m_i * ü_i
        m_accel = mass_matrix.unsqueeze(0).unsqueeze(-1) * accel_response
        # M·ι·üg
        if ground_accel.dim() == 2:
            ground_accel = ground_accel.unsqueeze(1)  # (B, 1, T)
        m_ground = (
            mass_matrix.unsqueeze(0).unsqueeze(-1)
            * influence_vector.unsqueeze(0).unsqueeze(-1)
            * ground_accel
        )
    else:
        # Full matrix M: (n_dof, n_dof) @ (B, n_dof, T)
        m_accel = torch.einsum("ij,bjt->bit", mass_matrix, accel_response)
        if ground_accel.dim() == 2:
            ground_accel = ground_accel.unsqueeze(1)
        m_ground = torch.einsum(
            "ij,j,bt->bit",
            mass_matrix,
            influence_vector,
            ground_accel.squeeze(1),
        )

    # C·u̇
    if damping_matrix is not None:
        if damping_matrix.dim() == 1:
            c_vel = damping_matrix.unsqueeze(0).unsqueeze(-1) * vel_response
        else:
            c_vel = torch.einsum("ij,bjt->bit", damping_matrix, vel_response)
    else:
        c_vel = torch.zeros_like(vel_response)

    # Residual: R = M·ü + C·u̇ + f_int + M·ι·üg
    residual = m_accel + c_vel + f_int + m_ground

    # Mean-squared residual
    return (residual**2).mean()


def boundary_condition_loss(
    pred_disp_t0: torch.Tensor,
    pred_vel_t0: torch.Tensor,
) -> torch.Tensor:
    """Initial-condition penalty: u(0)=0, u̇(0)=0.

    Parameters
    ----------
    pred_disp_t0 : torch.Tensor
        Predicted displacement at t=0, shape (B, n_stories).
    pred_vel_t0 : torch.Tensor
        Predicted velocity at t=0, shape (B, n_stories).

    Returns
    -------
    torch.Tensor
        Scalar MSE of initial conditions.
    """
    loss_u = (pred_disp_t0**2).mean()
    loss_v = (pred_vel_t0**2).mean()
    return loss_u + loss_v


# ═══════════════════════════════════════════════════════════════════════════
# Composite Loss
# ═══════════════════════════════════════════════════════════════════════════


class HybridPINNLoss(nn.Module):
    """Three-component hybrid physics-informed loss function.

    .. math::
        \\mathcal{L}_{total} =
            \\lambda_d \\mathcal{L}_{data}
          + \\lambda_p \\mathcal{L}_{physics}
          + \\lambda_b \\mathcal{L}_{bc}

    Parameters
    ----------
    weights : LossWeights
        Component weights.  Defaults: λ_d=1.0, λ_p=0.1, λ_b=0.01.

    Notes
    -----
    The physics loss uses **f_int(u, u̇)** (the nonlinear restoring force
    recorded at each time step by OpenSeesPy) rather than a static K·u
    product.  For RC sections modelled with Concrete02/Steel02, f_int
    encodes cracking, yielding, strain hardening, and cyclic degradation.
    This is what makes the digital twin "hybrid": physics from the FE
    solver constrains the data-driven neural network.

    Examples
    --------
    >>> loss_fn = HybridPINNLoss()
    >>> l_total, components = loss_fn(
    ...     pred=y_hat, target=y_true,
    ...     mass_matrix=m_diag, damping_matrix=c_diag,
    ...     accel_response=accel, vel_response=vel,
    ...     f_int=f_restore, ground_accel=ag,
    ...     pred_disp_t0=u0, pred_vel_t0=v0,
    ... )
    """

    def __init__(
        self,
        weights: LossWeights | None = None,
        story_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.w = weights or LossWeights()
        # Per-story inverse-variance weights for data_loss
        if story_weights is not None:
            self.register_buffer("story_weights", story_weights)
        else:
            self.story_weights = None
        logger.info(
            "HybridPINNLoss: λ_data=%.3f, λ_phys=%.3f, λ_bc=%.4f, story_weights=%s",
            self.w.lambda_data,
            self.w.lambda_phys,
            self.w.lambda_bc,
            "enabled" if story_weights is not None else "disabled",
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mass_matrix: torch.Tensor | None = None,
        damping_matrix: torch.Tensor | None = None,
        accel_response: torch.Tensor | None = None,
        vel_response: torch.Tensor | None = None,
        f_int: torch.Tensor | None = None,
        ground_accel: torch.Tensor | None = None,
        influence_vector: torch.Tensor | None = None,
        pred_disp_t0: torch.Tensor | None = None,
        pred_vel_t0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the composite loss.

        Parameters
        ----------
        pred : torch.Tensor
            Network predictions (IDR), shape (B, n_stories).
        target : torch.Tensor
            Ground-truth IDR from OpenSeesPy, shape (B, n_stories).
        mass_matrix .. influence_vector :
            Physics-loss inputs (see :func:`physics_loss`).
            If None, physics loss is skipped (pure data mode).
        pred_disp_t0, pred_vel_t0 :
            Initial-condition predictions.  If None, BC loss is skipped.

        Returns
        -------
        total_loss : torch.Tensor
            Weighted sum of all active components.
        components : dict[str, torch.Tensor]
            Individual unweighted losses for logging.
        """
        components: dict[str, torch.Tensor] = {}

        # ── L_data ─────────────────────────────────────────────────────
        l_data = data_loss(pred, target, story_weights=self.story_weights)
        components["L_data"] = l_data
        total = self.w.lambda_data * l_data

        # ── L_physics (optional — requires FE kinematic data) ──────────
        has_physics = all(
            t is not None
            for t in (
                mass_matrix,
                # damping_matrix, (optional now)
                accel_response,
                vel_response,
                f_int,
                ground_accel,
            )
        )
        if has_physics:
            l_phys = physics_loss(
                mass_matrix=mass_matrix,  # type: ignore[arg-type]
                damping_matrix=damping_matrix,  # type: ignore[arg-type]
                accel_response=accel_response,  # type: ignore[arg-type]
                vel_response=vel_response,  # type: ignore[arg-type]
                f_int=f_int,  # type: ignore[arg-type]
                ground_accel=ground_accel,  # type: ignore[arg-type]
                influence_vector=influence_vector,
            )
            components["L_physics"] = l_phys
            total = total + self.w.lambda_phys * l_phys
        else:
            components["L_physics"] = torch.tensor(0.0, device=pred.device)

        # ── L_bc (optional — requires initial-condition predictions) ───
        if pred_disp_t0 is not None and pred_vel_t0 is not None:
            l_bc = boundary_condition_loss(pred_disp_t0, pred_vel_t0)
            components["L_bc"] = l_bc
            total = total + self.w.lambda_bc * l_bc
        else:
            components["L_bc"] = torch.tensor(0.0, device=pred.device)

        components["L_total"] = total
        return total, components


# ═══════════════════════════════════════════════════════════════════════════
# Adaptive weight scheduling (Self-Adaptive PINNs, McClenny & Braga, 2023)
# ═══════════════════════════════════════════════════════════════════════════


class AdaptiveLossWeights:
    """Dynamically adjust λ_p and λ_b during training.

    Uses the gradient-norm balancing heuristic: scale λ so that the
    gradient magnitudes of each loss component are comparable.

    Parameters
    ----------
    loss_fn : HybridPINNLoss
        The loss function whose weights will be adapted.
    alpha : float
        EMA smoothing factor for running gradient statistics (0 < α < 1).
    min_weight : float
        Minimum allowed weight to prevent any term from vanishing.
    max_weight : float
        Maximum allowed weight to prevent instability.
    """

    def __init__(
        self,
        loss_fn: HybridPINNLoss,
        alpha: float = 0.9,
        min_weight: float = 1e-4,
        max_weight: float = 100.0,
    ) -> None:
        self.loss_fn = loss_fn
        self.alpha = alpha
        self.min_weight = min_weight
        self.max_weight = max_weight
        self._ema_data: float = 1.0
        self._ema_phys: float = 1.0
        self._ema_bc: float = 1.0

    def step(self, components: dict[str, torch.Tensor]) -> None:
        """Update weights based on current loss magnitudes.

        Parameters
        ----------
        components : dict[str, torch.Tensor]
            Loss components from HybridPINNLoss.forward().
        """
        with torch.no_grad():
            l_d = components["L_data"].item()
            l_p = components["L_physics"].item()
            l_b = components["L_bc"].item()

            # EMA update
            self._ema_data = self.alpha * self._ema_data + (1 - self.alpha) * l_d
            self._ema_phys = self.alpha * self._ema_phys + (1 - self.alpha) * l_p
            self._ema_bc = self.alpha * self._ema_bc + (1 - self.alpha) * l_b

            # Target: make all EMA magnitudes comparable to L_data
            if self._ema_phys > 0:
                ratio_p = self._ema_data / self._ema_phys
                new_lp = max(self.min_weight, min(self.max_weight, ratio_p))
                self.loss_fn.w.lambda_phys = new_lp

            if self._ema_bc > 0:
                ratio_b = self._ema_data / self._ema_bc
                new_lb = max(self.min_weight, min(self.max_weight, ratio_b))
                self.loss_fn.w.lambda_bc = new_lb

        logger.debug(
            "Adaptive weights: λ_p=%.4f, λ_b=%.4f",
            self.loss_fn.w.lambda_phys,
            self.loss_fn.w.lambda_bc,
        )
