"""Tests for the STOICHI native mass-conserving RNN."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from stoichi import StoichiometricConstraint, StoichiometricRNN, StoichiometricRNNCell
from stoichi.rnn import create_test_stoichiometry


class TestStoichiometricConstraint:
    def test_projection_completeness(self):
        A = create_test_stoichiometry()
        c = StoichiometricConstraint(A)
        identity = c.P_row + c.P_null
        np.testing.assert_allclose(
            identity.detach().numpy(),
            np.eye(3),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_mass_conservation_on_weight(self):
        A = create_test_stoichiometry()
        c = StoichiometricConstraint(A)
        W_raw = torch.randn(3, 3, dtype=torch.float64)
        W_rec = c.build_conserving_weight(W_raw)
        np.testing.assert_allclose(
            (A @ W_rec).detach().numpy(),
            A.detach().numpy(),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_input_is_mass_neutral(self):
        A = create_test_stoichiometry()
        c = StoichiometricConstraint(A)
        W_raw = torch.randn(3, 3, dtype=torch.float64)
        W_in = c.build_mass_neutral_weight(W_raw)
        np.testing.assert_allclose(
            (A @ W_in).detach().numpy(),
            np.zeros((2, 3)),
            rtol=1e-12,
            atol=1e-12,
        )


class TestStoichiometricRNNCell:
    @pytest.fixture
    def cell(self):
        return StoichiometricRNNCell(create_test_stoichiometry(), state_dim=3, input_dim=3)

    def test_exact_mass_conservation_single_step(self, cell):
        h = torch.tensor([10.0, 5.0, 2.0], dtype=torch.float64)
        x = torch.randn(3, dtype=torch.float64)
        mass_before = cell.get_mass_pool(h)
        h_next = cell(h, x)
        mass_after = cell.get_mass_pool(h_next)

        np.testing.assert_allclose(
            mass_before.detach().numpy(),
            mass_after.detach().numpy(),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_exact_mass_conservation_long_trajectory(self, cell):
        torch.manual_seed(42)
        h = torch.tensor([10.0, 5.0, 2.0], dtype=torch.float64)
        mass_target = cell.get_mass_pool(h)

        for step in range(1000):
            x = torch.randn(3, dtype=torch.float64)
            h = cell(h, x)
            mass_current = cell.get_mass_pool(h)
            drift = (mass_current - mass_target).abs().max().item()
            assert drift < 1e-12, f"Drift too large at step {step}: {drift}"


class TestStoichiometricRNN:
    def test_sequence_conservation(self):
        A = create_test_stoichiometry()
        rnn = StoichiometricRNN(A, state_dim=3, input_dim=3, num_layers=2)

        torch.manual_seed(42)
        x_seq = torch.randn(100, 3, dtype=torch.float64)
        h0 = torch.tensor([10.0, 5.0, 2.0], dtype=torch.float64)
        h_seq, _ = rnn(x_seq, h0=h0, return_all_states=True)

        mass_target = A @ h0
        for t in range(100):
            mass_t = A @ h_seq[t]
            drift = (mass_t - mass_target).abs().max().item()
            assert drift < 1e-12, f"Conservation drift at t={t}: {drift}"


if __name__ == "__main__":
    pytest.main(["-q", __file__])
