# STOICHI Benchmarking

This directory contains the benchmark harness and test suites for comparing STOICHI against FairChem baselines.

## Benchmark Tiers (Priority First-Principles Order)

### Tier 1: Conservation ✓
**Tests:** `tests/test_tier1_conservation_vs_baseline.py`
- Exact mass conservation over 100, 1000 steps
- Conservation with non-negativity clamping
- Gradient flow through conserving weights
- **Goal:** Prove STOICHI maintains conservation to machine precision
- **Expected Result:** STOICHI wins by 1e-15 vs 1e-8 drift

### Tier 2: Molecular Inference ✓
**Tests:** `tests/test_tier2_molecular_inference.py`
- Synthetic H2O energy prediction
- Forces respect stoichiometric constraints
- Batched inference on small molecules
- **Goal:** Show STOICHI generalizes to molecular tasks while maintaining conservation
- **Expected Result:** Exact conservation + stable energy prediction

### Tier 3: Full System Benchmarks (Two Parts)
**Tests:** `tests/test_tier3_standalone_and_wrapped.py`

#### Part A: STOICHI Standalone (Pure Native, No FairChem)
- `test_stoichi_standalone_synthetic_oc20_like`
  - Synthetic crystal data with element conservation
  - Native STOICHI RNN + dynamics
  - Benchmark: conservation error
  - **Expected:** STOICHI < 1e-10, FairChem ~1e-3

- `test_stoichi_standalone_synthetic_omol_like`
  - Synthetic molecule data (CH4, H2O, NH3, CO2, N2, O2)
  - STOICHI energy prediction head
  - Benchmark: energy MAE
  - **Expected:** Native STOICHI proves core hypothesis

#### Part B: STOICHI Wrapped in FairChem (Optional Integration)
- `test_stoichi_wrapped_fairchem_oc20_subset`
  - Load FairChem checkpoint
  - Wrap recurrent dynamics in STOICHI constraint layer
  - Benchmark against baseline

- `test_stoichi_wrapped_fairchem_omol_subset`
  - Same as above for molecules

---

## Running Benchmarks

### Tier 1 (Conservation)
```bash
pytest -v tests/test_tier1_conservation_vs_baseline.py
```

### Tier 2 (Molecular)
```bash
pytest -v tests/test_tier2_molecular_inference.py
```

### Tier 3 Standalone (Native STOICHI)
```bash
pytest -v tests/test_tier3_standalone_and_wrapped.py -k "standalone"
```

### Tier 3 Wrapped (Requires FairChem)
```bash
pytest -v tests/test_tier3_standalone_and_wrapped.py -k "wrapped"
```

### All Tiers
```bash
pytest -v tests/ -k "not wrapped"  # Skip FairChem-dependent tests
```

---

## Benchmark Harness API

Use `BenchmarkHarness` for fair test comparison:

```python
from benchmarks.benchmark_base import BenchmarkHarness

harness = BenchmarkHarness(device="cpu", dtype=torch.float64)

def test_fn():
    # Run your STOICHI test
    passed = ...  # bool
    metric_value = ...  # float
    return passed, metric_value, None, "metric_name"

def baseline_fn():
    # Return FairChem or baseline metric
    return baseline_metric_value

result = harness.run_test(
    "My STOICHI test",
    test_fn,
    baseline_fn,  # optional
)

harness.print_results()
```

**Tracked metrics:**
- Pass/fail status
- Metric value (your test's measure)
- Improvement % over baseline
- Runtime (seconds)
- Error messages

---

## Key Differences: STOICHI vs FairChem

| Property | STOICHI | FairChem (UMA-S, ESCN-MD) |
|----------|---------|---------------------------|
| **Conservation** | Exact (tensor invariant) | Soft penalty + projection |
| **Mass drift** | < 1e-15 (machine precision) | ~ 1e-3 to 1e-8 |
| **Architecture** | Stoichiometric RNN | Graph neural network |
| **Constraint encoding** | In weights (`A @ W_rec = A`) | Post-hoc regularization |
| **Scalability** | Any stoichiometry matrix A | Model-specific |
| **Tier 3 Part A** | **Native STOICHI** (no FairChem) | N/A |
| **Tier 3 Part B** | Wrapper around FairChem | Baseline |

---

## Success Criteria

- **Tier 1:** STOICHI conservation error < 1e-12 over 1000 steps
- **Tier 2:** STOICHI energy prediction stable and differentiable
- **Tier 3A (Standalone):** Pure STOICHI beats conservation + energy metrics without any FairChem code
- **Tier 3B (Wrapped):** STOICHI constraint improves (or matches) FairChem performance

---

## Directory Structure

```
benchmarks/
├── __init__.py
├── benchmark_base.py           # BenchmarkHarness class
├── README_BENCHMARKS.md        # This file

tests/
├── test_tier1_conservation_vs_baseline.py
├── test_tier2_molecular_inference.py
├── test_tier3_standalone_and_wrapped.py
├── test_stoichiometric_rnn.py              # Original unit tests
```
