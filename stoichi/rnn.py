"""Stoichiometric mass-conserving recurrent dynamics.

This module encodes conservation laws directly into the recurrent tensor weights.
For a stoichiometric matrix A with rows corresponding to conserved quantities,
we enforce

    A @ W_rec = A
    A @ W_in = 0

so that the state update preserves the conserved mass vector by construction:

    A @ h[t] = A @ h[t-1]

for all time steps.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class StoichiometricConstraint(nn.Module):
    """Tensor operator that enforces conservation via null-space projectors."""

    def __init__(self, A: torch.Tensor, dtype: torch.dtype = torch.float64):
        super().__init__()
        A = A.to(dtype)
        if A.dim() != 2:
            raise ValueError(f"A must be 2D; got shape {tuple(A.shape)}")

        n_elements, n_species = A.shape
        if n_elements >= n_species:
            raise ValueError(
                "A should be under-determined: number of conserved quantities should be "
                f"less than state dimensionality. Got {n_elements} x {n_species}."
            )

        self.register_buffer("A", A)
        self.register_buffer("A_pinv", torch.linalg.pinv(A))

        # Row-space projector: P_row = A^+ A
        P_row = self.A_pinv @ A
        # Null-space projector: P_null = I - P_row
        P_null = torch.eye(n_species, dtype=dtype) - P_row

        self.register_buffer("P_row", P_row)
        self.register_buffer("P_null", P_null)

        self.n_elements = n_elements
        self.n_species = n_species
        self.dtype = dtype

    def get_conserved_quantities(self, h: torch.Tensor) -> torch.Tensor:
        """Return A @ h, the conserved mass vector."""
        return self.A @ h

    def build_conserving_weight(self, W_raw: torch.Tensor) -> torch.Tensor:
        """Return W_rec such that A @ W_rec = A."""
        return self.P_row + self.P_null @ W_raw @ self.P_null

    def build_mass_neutral_weight(self, W_raw: torch.Tensor) -> torch.Tensor:
        """Return W_in such that A @ W_in = 0."""
        return self.P_null @ W_raw

    def verify_conservation(
        self, h_sequence: torch.Tensor, atol: float = 1e-12, rtol: float = 1e-12
    ) -> Dict[str, Any]:
        """Check conservation over a trajectory of states."""
        if h_sequence.dim() != 2:
            raise ValueError(f"h_sequence must be 2D [T, n_species]; got {h_sequence.shape}")

        mass_sequence = self.A @ h_sequence.T
        mass_initial = mass_sequence[:, :1]
        drift = (mass_sequence - mass_initial).abs()
        max_drift = float(drift.max().item())
        mean_drift = float(drift.mean().item())
        violations = int((drift > (atol + rtol * mass_initial.abs())).sum().item())

        return {
            "max_drift": max_drift,
            "mean_drift": mean_drift,
            "violations": violations,
            "passes": violations == 0,
            "n_elements": self.n_elements,
            "n_species": self.n_species,
        }


class StoichiometricRNNCell(nn.Module):
    """RNN cell with native stoichiometric conservation in recurrent weights."""

    def __init__(
        self,
        A: torch.Tensor,
        state_dim: Optional[int] = None,
        input_dim: Optional[int] = None,
        hidden_scale: float = 0.01,
    ):
        super().__init__()
        A = A.to(torch.float64)
        if A.dim() != 2:
            raise ValueError(f"A must be 2D; got shape {tuple(A.shape)}")

        n_elements, n_species = A.shape
        if state_dim is None:
            state_dim = n_species
        if state_dim != n_species:
            raise ValueError(
                f"state_dim ({state_dim}) must match A.shape[1] ({n_species})"
            )
        if input_dim is None:
            input_dim = state_dim

        self.state_dim = state_dim
        self.input_dim = input_dim
        self.constraint = StoichiometricConstraint(A, dtype=torch.float64)

        self.W_rec_raw = nn.Parameter(
            torch.randn(state_dim, state_dim, dtype=torch.float64) * hidden_scale
        )
        self.W_in_raw = nn.Parameter(
            torch.randn(state_dim, input_dim, dtype=torch.float64) * hidden_scale
        )

    @property
    def W_rec(self) -> torch.Tensor:
        """Conserving recurrent matrix: A @ W_rec = A."""
        return self.constraint.build_conserving_weight(self.W_rec_raw)

    @property
    def W_in(self) -> torch.Tensor:
        """Mass-neutral input matrix: A @ W_in = 0."""
        return self.constraint.build_mass_neutral_weight(self.W_in_raw)

    def forward(self, h: torch.Tensor, x: torch.Tensor, clamp_nonnegative: bool = True):
        h_next = self.W_rec @ h + self.W_in @ x
        if clamp_nonnegative:
            h_next = torch.clamp(h_next, min=0.0)
        return h_next

    def get_mass_pool(self, h: torch.Tensor) -> torch.Tensor:
        return self.constraint.get_conserved_quantities(h)


class StoichiometricRNN(nn.Module):
    """Sequence model composed of stoichiometric cells."""

    def __init__(
        self,
        A: torch.Tensor,
        state_dim: Optional[int] = None,
        input_dim: Optional[int] = None,
        num_layers: int = 1,
        hidden_scale: float = 0.01,
    ):
        super().__init__()
        A = A.to(torch.float64)
        n_elements, n_species = A.shape

        if state_dim is None:
            state_dim = n_species
        if input_dim is None:
            input_dim = state_dim

        self.state_dim = state_dim
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.cells = nn.ModuleList(
            [
                StoichiometricRNNCell(
                    A,
                    state_dim=state_dim,
                    input_dim=input_dim if i == 0 else state_dim,
                    hidden_scale=hidden_scale,
                )
                for i in range(num_layers)
            ]
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        return_all_states: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x_seq.dim() != 2:
            raise ValueError(f"x_seq must be 2D [T, input_dim]; got {x_seq.shape}")

        T, input_dim = x_seq.shape
        if input_dim != self.input_dim:
            raise ValueError(
                f"input_dim mismatch: x_seq has {input_dim}, expected {self.input_dim}"
            )

        device = x_seq.device
        dtype = x_seq.dtype

        if h0 is None:
            h0 = torch.zeros(self.state_dim, device=device, dtype=dtype)

        h = h0.clone()
        outputs: list[torch.Tensor] = []

        for t in range(T):
            x_t = x_seq[t]
            for cell in self.cells:
                h = cell(h, x_t)
            outputs.append(h.clone())

        h_sequence = torch.stack(outputs) if return_all_states else None
        return h_sequence, h

    def verify_trajectory_conservation(
        self, x_seq: torch.Tensor, h0: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        h_seq, _ = self.forward(x_seq, h0=h0, return_all_states=True)
        return self.cells[0].constraint.verify_conservation(h_seq)

    def get_conserved_quantities(
        self, x_seq: torch.Tensor, h0: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h_seq, _ = self.forward(x_seq, h0=h0, return_all_states=True)
        mass_seq = self.cells[0].constraint.A @ h_seq.T
        return mass_seq.T


def create_test_stoichiometry() -> torch.Tensor:
    """Minimal water-like system with conserved hydrogen and oxygen."""
    return torch.tensor(
        [
            [2.0, 2.0, 0.0],  # H balance: H2O + H2 + 0*O2
            [1.0, 0.0, 2.0],  # O balance: H2O + 0*H2 + 2*O2
        ],
        dtype=torch.float64,
    )
