#!/usr/bin/env python3

import argparse

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Compare a candidate Laplace HDF5 field against a reference."
    )
    parser.add_argument(
        "reference",
        help="Reference HDF5 file, normally produced by NumPy.",
    )
    parser.add_argument(
        "candidate",
        help="Candidate HDF5 file to validate.",
    )
    parser.add_argument("--rtol", type=float, default=1.0e-11)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    args = parser.parse_args()

    if not np.isfinite(args.rtol) or args.rtol < 0.0:
        raise ValueError("rtol must be finite and non-negative")

    if not np.isfinite(args.atol) or args.atol < 0.0:
        raise ValueError("atol must be finite and non-negative")

    with (
        h5py.File(args.reference, "r") as fref,
        h5py.File(args.candidate, "r") as fcandidate,
    ):
        if "u" not in fref:
            print("FAILED: dataset 'u' is missing from the reference")
            raise SystemExit(1)

        if "u" not in fcandidate:
            print("FAILED: dataset 'u' is missing from the candidate")
            raise SystemExit(1)

        dset_ref = fref["u"]
        dset_candidate = fcandidate["u"]

        if dset_ref.shape != dset_candidate.shape:
            print(
                "FAILED: shape mismatch "
                f"{dset_ref.shape} != {dset_candidate.shape}"
            )
            raise SystemExit(1)

        if dset_ref.ndim != 2:
            print(
                "FAILED: dataset 'u' must be two-dimensional; "
                f"found ndim={dset_ref.ndim}"
            )
            raise SystemExit(1)

        if dset_ref.shape[0] < 3 or dset_ref.shape[1] < 3:
            print(
                "FAILED: dataset 'u' must have at least three rows "
                "and three columns"
            )
            raise SystemExit(1)

        if dset_ref.dtype != dset_candidate.dtype:
            print(
                "FAILED: dtype mismatch "
                f"{dset_ref.dtype} != {dset_candidate.dtype}"
            )
            raise SystemExit(1)

        reference = dset_ref[...]
        candidate = dset_candidate[...]

    finite_reference = np.isfinite(reference).all()
    finite_candidate = np.isfinite(candidate).all()
    finite = finite_reference and finite_candidate

    print(f"reference         : {args.reference}")
    print(f"candidate         : {args.candidate}")
    print(f"shape             : {reference.shape}")
    print(f"dtype             : {reference.dtype}")
    print(f"rtol              : {args.rtol:.6e}")
    print(f"atol              : {args.atol:.6e}")

    if not finite:
        print(f"reference finite  : {finite_reference}")
        print(f"candidate finite  : {finite_candidate}")
        print("FAILED: one or both datasets contain non-finite values")
        raise SystemExit(1)

    diff = candidate - reference

    max_abs = np.max(np.abs(diff))
    rms_abs = np.sqrt(np.mean(diff * diff))

    reference_l2 = np.linalg.norm(reference.ravel())
    rel_l2 = np.linalg.norm(diff.ravel()) / max(reference_l2, 1.0e-300)

    interior_diff = diff[1:-1, 1:-1]
    interior_reference = reference[1:-1, 1:-1]

    interior_max_abs = np.max(np.abs(interior_diff))
    interior_reference_l2 = np.linalg.norm(interior_reference.ravel())
    interior_rel_l2 = (
        np.linalg.norm(interior_diff.ravel())
        / max(interior_reference_l2, 1.0e-300)
    )

    # The second argument is the reference for NumPy's relative tolerance.
    close = np.allclose(
        candidate,
        reference,
        rtol=args.rtol,
        atol=args.atol,
    )

    print(f"max abs diff      : {max_abs:.16e}")
    print(f"RMS abs diff      : {rms_abs:.16e}")
    print(f"relative L2 diff  : {rel_l2:.16e}")
    print(f"interior max abs  : {interior_max_abs:.16e}")
    print(f"interior rel L2   : {interior_rel_l2:.16e}")

    if not close:
        print("FAILED: candidate exceeds the requested tolerances")
        raise SystemExit(1)

    print("PASSED")


if __name__ == "__main__":
    main()
