#!/usr/bin/env python3
"""
Quick test script for BSDS Complete implementation.
Validates that all modules load correctly and basic functionality works.
"""

import sys
import numpy as np
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that all modules import correctly."""
    print("Testing imports...")

    try:
        from bsds_complete import BSDSModel, BSDSConfig
        print("  ✓ Core imports")

        from bsds_complete.inference import (
            vbhmm_forward, vbhmm_backward, vbhmm_estep,
            viterbi_decode, infer_qx, compute_log_out_probs
        )
        print("  ✓ Inference imports")

        from bsds_complete.learning import (
            infer_ql, infer_qnu, infer_psii, infer_mcl,
            infer_qtheta, infer_ar, mstep_vbvar
        )
        print("  ✓ Learning imports")

        from bsds_complete.analysis import (
            compute_occupancy_group, compute_mean_lifetime_group,
            get_dominant_states_group, create_summary_report
        )
        print("  ✓ Analysis imports")

        from bsds_complete.utils import (
            logsumexp, kl_dirichlet, safe_cholesky,
            preprocess_data, validate_input
        )
        print("  ✓ Utils imports")

        return True

    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False


def test_synthetic_data():
    """Test BSDS on synthetic data."""
    print("\nTesting on synthetic data...")

    from bsds_complete import BSDSModel, BSDSConfig

    # Generate synthetic data
    np.random.seed(42)
    n_subjects = 3
    n_rois = 20
    n_timepoints = 100
    n_true_states = 3

    # Create state-specific means
    state_means = [np.random.randn(n_rois) * 2 for _ in range(n_true_states)]

    # Generate data
    data_list = []
    for _ in range(n_subjects):
        # Random state sequence
        states = np.random.choice(n_true_states, n_timepoints)

        # Generate observations
        Y = np.zeros((n_rois, n_timepoints))
        for t in range(n_timepoints):
            Y[:, t] = state_means[states[t]] + np.random.randn(n_rois) * 0.5

        data_list.append(Y)

    print(f"  Generated {n_subjects} subjects, {n_rois} ROIs, {n_timepoints} timepoints")

    # Configure model (small for quick test)
    config = BSDSConfig(
        n_states=3,
        max_ldim=5,
        n_iter=20,
        n_init_learning=2,
        verbose=False
    )

    # Fit model
    print("  Fitting BSDS model...")
    model = BSDSModel(config)
    model.fit(data_list)

    # Check results
    states = model.get_states()
    stats = model.get_summary_statistics()

    print(f"  ✓ Model fitted successfully")
    print(f"  ✓ Final log-likelihood: {model.log_lik_history[-1]:.2f}")
    print(f"  ✓ Dominant states: {stats['dominant_states_group']}")
    print(f"  ✓ Occupancy: {[f'{o:.2f}' for o in stats['occupancy_group']]}")

    return True


def test_save_load():
    """Test model save/load functionality."""
    print("\nTesting save/load...")

    from bsds_complete import BSDSModel, BSDSConfig
    import tempfile
    import os

    # Quick synthetic data
    np.random.seed(42)
    data_list = [np.random.randn(10, 50) for _ in range(2)]

    # Fit small model
    config = BSDSConfig(n_states=2, max_ldim=3, n_iter=10, n_init_learning=1, verbose=False)
    model = BSDSModel(config)
    model.fit(data_list)

    # Save
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_model.pkl")
        model.save(save_path)
        print(f"  ✓ Model saved")

        # Load
        loaded_model = BSDSModel.load(save_path)
        print(f"  ✓ Model loaded")

        # Compare
        original_ll = model.log_lik_history[-1]
        loaded_ll = loaded_model.log_lik_history[-1]

        if np.abs(original_ll - loaded_ll) < 1e-6:
            print(f"  ✓ Model parameters match")
            return True
        else:
            print(f"  ✗ Model mismatch: {original_ll} vs {loaded_ll}")
            return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("BSDS Complete - Test Suite")
    print("=" * 50)

    tests = [
        ("Imports", test_imports),
        ("Synthetic Data", test_synthetic_data),
        ("Save/Load", test_save_load),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  ✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
