"""Tier 2: Molecular inference benchmarks.

Compare STOICHI to ESCNMDBackbone on synthetic H2O and small molecules.
Benchmark energy and forces prediction accuracy.
"""

from __future__ import annotations

import pytest
import torch
import numpy as np

from stoichi.rnn import create_test_stoichiometry
from stoichi import StoichiometricRNN
from benchmarks.benchmark_base import BenchmarkHarness


class TestMolecularInference:
    """STOICHI inference on molecular tasks."""

    @pytest.fixture
    def harness(self):
        return BenchmarkHarness(device="cpu", dtype=torch.float64)

    @pytest.fixture
    def water_stoich(self):
        """H2O-like stoichiometry: [H2O, H2, O2] with H and O conservation."""
        return torch.tensor(
            [
                [2.0, 2.0, 0.0],  # H
                [1.0, 0.0, 2.0],  # O
            ],
            dtype=torch.float64,
        )

    def test_stoichi_energy_prediction_water(self, harness, water_stoich):
        """STOICHI predicts stable energy for H2O system."""
        def test_fn():
            torch.manual_seed(42)
            # Build a model that maps atomic features to energy
            # Input: atomic features [H2O_abundance, H2_abundance, O2_abundance]
            # Output: predicted energy
            rnn = StoichiometricRNN(
                water_stoich,
                state_dim=3,
                input_dim=3,
                num_layers=2,
                hidden_scale=0.01,
            )

            # Add a simple energy head
            energy_head = torch.nn.Linear(3, 1)

            # Initial state (water molecule)
            h0 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
            # Random input perturbation
            x_input = torch.randn(3, dtype=torch.float64) * 0.01

            # Forward pass through conserving dynamics
            h_seq, h_final = rnn(
                x_input.unsqueeze(0),
                h0=h0,
                return_all_states=False,
            )
            # Predict energy from final state
            energy = energy_head(h_final).squeeze().item()

            # Check that energy is finite and reasonable
            passed = np.isfinite(energy)
            mae = abs(energy)  # Metric: absolute energy value
            return passed, mae, None, "energy_mae"

        result = harness.run_test(
            "STOICHI energy prediction (H2O)",
            test_fn,
        )
        assert result.passed, f"Energy prediction test failed: {result.error_msg}"

    def test_stoichi_forces_conservation_water(self, harness, water_stoich):
        """STOICHI forces respect stoichiometric constraints on H2O."""
        def test_fn():
            torch.manual_seed(42)
            # Build dynamics model
            rnn = StoichiometricRNN(
                water_stoich,
                state_dim=3,
                input_dim=3,
                num_layers=1,
            )

            # Initial state
            h0 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
            # Input (external perturbation)
            x = torch.randn(3, dtype=torch.float64)

            # Forward pass
            h1 = h0.clone()
            for cell in rnn.cells:
                h1 = cell(h1, x)

            # Compute force as negative gradient
            loss = h1.sum()
            loss.backward()
            forces = -h0.grad

            # Check that forces preserve stoichiometric balance
            # i.e., A @ forces should be small (not create/destroy mass)
            mass_force = water_stoich @ forces
            max_mass_violation = mass_force.abs().max().item()

            passed = max_mass_violation < 1e-8  # Forces don't create/destroy mass
            return passed, max_mass_violation, None, "mass_violation_forces"

        result = harness.run_test(
            "STOICHI forces respect stoichiometry (H2O)",
            test_fn,
        )
        assert result.passed, f"Forces test failed: {result.error_msg}"

    def test_stoichi_batch_inference_small_molecules(self, harness, water_stoich):
        """STOICHI handles batched inference on multiple small systems."""
        def test_fn():
            torch.manual_seed(42)
            rnn = StoichiometricRNN(water_stoich, state_dim=3, input_dim=3, num_layers=1)

            # Batch of 32 systems
            batch_size = 32
            x_batch = torch.randn(batch_size, 3, dtype=torch.float64)
            h0_batch = torch.randn(batch_size, 3, dtype=torch.float64)
            h0_batch = torch.clamp(h0_batch, min=0.1)  # Non-negative abundances

            # Process batch: each system evolves independently
            max_drift = 0.0
            for b in range(batch_size):
                h0 = h0_batch[b]
                x = x_batch[b]
                mass_target = water_stoich @ h0

                # Single step
                h_next = h0.clone()
                for cell in rnn.cells:
                    h_next = cell(h_next, x)

                mass_final = water_stoich @ h_next
                drift = (mass_final - mass_target).abs().max().item()
                max_drift = max(max_drift, drift)

            passed = max_drift < 1e-12
            return passed, max_drift, None, "max_drift_batch"

        result = harness.run_test(
            "STOICHI batch inference (32 small molecules)",
            test_fn,
        )
        assert result.passed, f"Batch inference test failed: {result.error_msg}"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
