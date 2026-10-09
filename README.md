# STOICHI

STOICHI is a tensor-first, mass-conserving recurrent architecture built from stoichiometric constraints.

The core invariant is:

- A @ W_rec = A
- A @ W_in = 0

so the state update preserves conserved mass by construction:

- A @ h[t] = A @ h[t-1]

## Why this matters

Mass conservation is encoded inside the geometry of the weights, not added as a soft penalty or post-hoc correction. This makes the model:

- exact by construction
- domain-agnostic
- scalable to general conservation-law dynamics
- naturally compatible with chemistry, materials, reaction systems, and other constrained processes

## Quick usage

```python
import torch
from stoichi import StoichiometricRNN

A = torch.tensor(
    [
        [2.0, 2.0, 0.0],
        [1.0, 0.0, 2.0],
    ],
    dtype=torch.float64,
)

model = StoichiometricRNN(A, state_dim=3, input_dim=3, num_layers=2)
x_seq = torch.randn(32, 3, dtype=torch.float64)
h0 = torch.tensor([10.0, 5.0, 2.0], dtype=torch.float64)

h_seq, h_final = model(x_seq, h0=h0)
print(h_seq.shape)
print(h_final.shape)
```

## Validation

Run:

```bash
pytest -q tests/test_stoichiometric_rnn.py
```
