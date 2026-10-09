"""
Benchmark tests for STOICHI.

These tests validate the benchmark suite itself and the conservation logic.
"""

import numpy as np
import pytest
import torch

from stoichi import StoichiometricRNNCell
from stoichi.rnn import create_test_stoichiometry
from benchmarks.benchmark import benchmark_conservation_exact, benchmark_runtime


class DummyStoichiModel:
    """Minimal model for testing benchmark harness."""

    def __init__(self):
        A = create_test_stoichiometry()
        self.constraint = type("Constraint", (), {"A": A})()
        self.input_dim = 3
        self.dtype = torch.float64

    def __call__(self, x):
        return x.sum()


def test_benchmark_conservation_exact():
    """Test conservation benchmark on dummy model."""
    model = DummyStoichiModel()
    max_drift, mean_drift, violations = benchmark_conservation_exact(model, steps=100)

    assert max_drift < 1e-12
    assert violations == 0


def test_benchmark_runtime_returns_positive_values():
    """Test that runtime benchmark returns reasonable values."""
    model = DummyStoichiModel()
    runtime_ms, throughput = benchmark_runtime(model, warmup_runs=5, eval_runs=10)

    assert runtime_ms > 0
    assert throughput > 0


def test_stoichiometric_cell_conservation_exact():
    """Test that StoichiometricRNNCell preserves conservation exactly."""
    A = create_test_stoichiometry()
    cell = StoichiometricRNNCell(A, state_dim=3, input_dim=3)

    h = torch.tensor([10.0, 5.0, 2.0], dtype=torch.float64)
    mass0 = A @ h

    for _ in range(200):
        x = torch.randn(3, dtype=torch.float64)
        h = cell(h, x)
        mass = A @ h

    np.testing.assert_allclose(
        mass0.detach().numpy(),
        (A @ h).detach().numpy(),
        rtol=1e-12,
        atol=1e-12,
    )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
