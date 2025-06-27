import argparse
import os
import ase
from ase.io import read, write
import numpy as np


def run(data_path, save_path, num_points, seed):
    ase_atoms: list[ase.Atoms] = []

    read_ase_atoms = read(data_path, index=":")
    if isinstance(read_ase_atoms, ase.Atoms):
        # workaround edge case of a single configuration
        ase_atoms = [read_ase_atoms]
    else:
        ase_atoms = read_ase_atoms

    rng = np.random.default_rng(seed)
    rng.shuffle(ase_atoms)  # type: ignore
    ase_atoms = ase_atoms[:num_points]
    print(f"Subsampled dataset at {data_path} with random seed {seed}.")
    print(f"Requested {num_points}. Output dataset will have {len(ase_atoms)} points.")

    # Save
    if os.path.exists(save_path):
        print(f"Output path {save_path} will be overwritten!")
    write(save_path, ase_atoms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="Path to the .xyz file to be subsampled")
    parser.add_argument("--save-path", type=str, required=True, help="Path where to save the output .xyz file")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--points", type=int, required=True, help="Number of points to randomly sample")
    args = parser.parse_args()

    run(args.data_path, args.save_path, args.points, args.seed)
