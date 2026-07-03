#!/usr/bin/env python3

import argparse
import numpy as np
import h5py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--rtol", type=float, default=1e-11)
    parser.add_argument("--atol", type=float, default=1e-12)
    args = parser.parse_args()

    with h5py.File(args.a, "r") as fa:
        ua = fa["u"][...]

    with h5py.File(args.b, "r") as fb:
        ub = fb["u"][...]

    if ua.shape != ub.shape:
        raise ValueError(f"Shape mismatch: {ua.shape} vs {ub.shape}")

    diff = ua - ub

    max_abs = np.max(np.abs(diff))
    rms_abs = np.sqrt(np.mean(diff * diff))
    rel_l2 = np.linalg.norm(diff.ravel()) / max(np.linalg.norm(ua.ravel()), 1e-300)

    interior = diff[1:-1, 1:-1]
    interior_ref = ua[1:-1, 1:-1]

    interior_max_abs = np.max(np.abs(interior))
    interior_rel_l2 = np.linalg.norm(interior.ravel()) / max(np.linalg.norm(interior_ref.ravel()), 1e-300,)

    print(f"shape             : {ua.shape}")
    print(f"max abs diff      : {max_abs:.16e}")
    print(f"RMS abs diff      : {rms_abs:.16e}")
    print(f"relative L2 diff  : {rel_l2:.16e}")
    print(f"interior max abs  : {interior_max_abs:.16e}")
    print(f"interior rel L2   : {interior_rel_l2:.16e}")
    if not np.allclose(ua, ub, rtol=args.rtol, atol=args.atol):
        print("FAILED")
    print("PASSED")


if __name__ == "__main__":
    main()
