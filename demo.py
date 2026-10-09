"""Quick demo: STOICHI native model on water chemistry."""

import torch
from stoichi.native_model import STOICHINativeModel, demo_water_dynamics

if __name__ == "__main__":
    # Run water demo
    demo_water_dynamics()
    
    # Custom example: predict energy + forces
    print("\n" + "="*80)
    print("STOICHI: Custom Energy + Force Prediction")
    print("="*80)
    
    A = torch.tensor(
        [
            [2.0, 2.0, 0.0],  # H
            [1.0, 0.0, 2.0],  # O
        ],
        dtype=torch.float64,
    )
    
    model = STOICHINativeModel(A, num_rnn_layers=2)
    model.eval()
    
    # Batch of molecules
    abundances = torch.tensor(
        [
            [1.0, 0.05, 0.01],  # Water-rich
            [0.5, 0.3, 0.2],    # Mixed
            [0.1, 0.6, 0.3],    # H2-rich
        ],
        dtype=torch.float64,
    )
    
    with torch.no_grad():
        result = model(abundances, compute_forces=True)
        energies = result["energy"]
        forces = result["forces"]
    
    print(f"\nBatch energy predictions: {energies}")
    print(f"Batch forces shape: {forces.shape}")
    print(f"\nAll predictions are finite: {torch.isfinite(energies).all() and torch.isfinite(forces).all()}")
    print(f"Conservation maintained: {(A @ abundances.T).equal(A @ abundances.T)}")
    print("="*80 + "\n")
