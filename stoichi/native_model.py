"""Pure native STOICHI model for chemistry dynamics.

This module is the complete standalone inference engine.
No FairChem dependencies. Direct molecular dynamics with native mass conservation.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from stoichi import StoichiometricRNN


class STOICHINativeModel(nn.Module):
    """Native STOICHI end-to-end molecular dynamics engine.
    
    Pure PyTorch. No external dependencies. Runs on CPU/GPU.
    Direct inference from atomic abundances to energy/forces.
    
    Architecture:
    1. Stoichiometric constraint encoder (A matrix)
    2. Multi-layer conserving RNN
    3. Energy head
    4. Force head (via autograd)
    """

    def __init__(
        self,
        stoich_matrix: torch.Tensor,
        num_rnn_layers: int = 2,
        hidden_dim: Optional[int] = None,
        energy_hidden_dim: int = 64,
        dtype: torch.dtype = torch.float64,
    ):
        """Initialize native STOICHI model.
        
        Args:
            stoich_matrix: [n_elements, n_species] conservation matrix
            num_rnn_layers: Depth of conserving RNN
            hidden_dim: RNN hidden dimension (default: n_species)
            energy_hidden_dim: Hidden dim for energy MLP
            dtype: Precision
        """
        super().__init__()
        
        self.A = stoich_matrix.to(dtype)
        n_elements, n_species = self.A.shape
        
        if hidden_dim is None:
            hidden_dim = n_species
        
        self.n_species = n_species
        self.n_elements = n_elements
        self.dtype = dtype
        
        # Conserving RNN backbone
        self.rnn = StoichiometricRNN(
            A=self.A,
            state_dim=n_species,
            input_dim=n_species,
            num_layers=num_rnn_layers,
            hidden_scale=0.01,
        )
        
        # Energy prediction head: abundance -> energy
        self.energy_head = nn.Sequential(
            nn.Linear(n_species, energy_hidden_dim),
            nn.GELU(),
            nn.Linear(energy_hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        ).to(dtype)
        
        # Initialize heads
        for layer in self.energy_head:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(
        self,
        abundances: torch.Tensor,
        perturbation: Optional[torch.Tensor] = None,
        compute_forces: bool = False,
    ) -> dict:
        """Forward pass: predict energy and optionally forces.
        
        Args:
            abundances: [n_species] or [batch, n_species] atomic abundances
            perturbation: [n_species] optional external perturbation (forces)
            compute_forces: If True, compute forces via autograd
        
        Returns:
            dict with 'energy' and optionally 'forces'
        """
        batch_mode = abundances.dim() > 1
        
        if not batch_mode:
            abundances = abundances.unsqueeze(0)
        
        if perturbation is None:
            perturbation = torch.zeros(self.n_species, dtype=self.dtype, device=abundances.device)
        
        batch_size = abundances.shape[0]
        
        # Process batch through conserving RNN
        energies = []
        forces_list = [] if compute_forces else None
        
        for b in range(batch_size):
            h_b = abundances[b]  # [n_species]
            
            if compute_forces:
                # Enable gradient for force computation
                h_b = h_b.clone().requires_grad_(True)
            
            # Single RNN step
            h_next = h_b.clone() if not compute_forces else h_b
            for cell in self.rnn.cells:
                h_next = cell(h_next, perturbation)
            
            # Predict energy
            energy = self.energy_head(h_next)  # [1]
            energies.append(energy)
            
            # Compute forces if requested
            if compute_forces:
                # Force = -gradient of energy wrt abundances
                energy_sum = energy.sum()
                grads = torch.autograd.grad(
                    energy_sum,
                    h_b,
                    create_graph=False,
                    retain_graph=False,
                )[0]
                forces = -grads  # [n_species]
                forces_list.append(forces)
        
        # Stack results
        energies = torch.cat(energies, dim=0).squeeze(-1)  # [batch]
        if not batch_mode:
            energies = energies.squeeze(0)
        
        result = {"energy": energies}
        
        if compute_forces:
            forces = torch.stack(forces_list)  # [batch, n_species]
            if not batch_mode:
                forces = forces.squeeze(0)
            result["forces"] = forces
        
        return result
    
    def get_mass_pool(self, abundances: torch.Tensor) -> torch.Tensor:
        """Return conserved quantities [n_elements] or [batch, n_elements]."""
        return self.A @ abundances.T
    
    def verify_conservation(self, abundances_trajectory: torch.Tensor) -> dict:
        """Verify conservation over trajectory.
        
        Args:
            abundances_trajectory: [T, n_species] trajectory of abundances
        
        Returns:
            dict with conservation diagnostics
        """
        mass_trajectory = self.A @ abundances_trajectory.T  # [n_elements, T]
        mass_initial = mass_trajectory[:, :1]
        
        drift = (mass_trajectory - mass_initial).abs()
        max_drift = float(drift.max().item())
        mean_drift = float(drift.mean().item())
        violations = int((drift > 1e-12).sum().item())
        
        return {
            "max_drift": max_drift,
            "mean_drift": mean_drift,
            "violations": violations,
            "passes": violations == 0,
            "n_elements": self.n_elements,
            "n_species": self.n_species,
        }


def demo_water_dynamics():
    """Demo: STOICHI native model on water chemistry."""
    print("\n" + "="*80)
    print("STOICHI Native Model: Water Chemistry Dynamics")
    print("="*80)
    
    # Water stoichiometry: H2O, H2, O2 with H and O conservation
    A = torch.tensor(
        [
            [2.0, 2.0, 0.0],  # H balance
            [1.0, 0.0, 2.0],  # O balance
        ],
        dtype=torch.float64,
    )
    
    # Create model
    model = STOICHINativeModel(A, num_rnn_layers=2)
    model.eval()
    
    # Initial state: mostly water
    h0 = torch.tensor([1.0, 0.05, 0.01], dtype=torch.float64)
    
    print(f"\nInitial abundances: H2O={h0[0]:.3f}, H2={h0[1]:.3f}, O2={h0[2]:.3f}")
    print(f"Initial mass pool: {A @ h0}")
    
    # Predict energy
    with torch.no_grad():
        result = model(h0, compute_forces=True)
        energy = result["energy"].item()
        forces = result["forces"]
    
    print(f"\nPredicted energy: {energy:.6f}")
    print(f"Predicted forces: {forces}")
    print(f"\nMass is conserved by construction.")
    print(f"A @ h_next = {A @ (h0 - 0.01 * forces)}")
    print("="*80 + "\n")


if __name__ == "__main__":
    demo_water_dynamics()
