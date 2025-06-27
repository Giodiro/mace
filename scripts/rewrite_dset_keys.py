import argparse
import os
import ase
from ase.io import read, write


def run(data_path, save_path):
    read_ase_atoms = read(data_path, index=":")
    if isinstance(read_ase_atoms, ase.Atoms):
        # workaround edge case of a single configuration
        ase_atoms = [read_ase_atoms]
    else:
        ase_atoms = read_ase_atoms

    for atoms in ase_atoms:
        try:
            atoms.info["REF_energy"] = atoms.get_potential_energy()
        except Exception as e:
            print(f"Failed to extract energy: {e}")
            atoms.info["REF_energy"] = None
        try:
            atoms.arrays["REF_forces"] = atoms.get_forces()
        except Exception as e:  # pylint: disable=W0703
            print(f"Failed to extract forces: {e}")
            atoms.arrays["REF_forces"] = None

    if os.path.exists(save_path):
        print(f"Output path {save_path} will be overwritten!")
    write(save_path, ase_atoms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="Path to the .xyz file to be subsampled")
    parser.add_argument("--save-path", type=str, required=True, help="Path where to save the output .xyz file")
    args = parser.parse_args()

    run(args.data_path, args.save_path)

"""
python scripts/rewrite_dset_keys.py --data-path=/leonardo_scratch/fast/IIT24_AtomSim/franken/franken_cache/water/ML_AB_dataset_1.xyz --save-path=/leonardo_scratch/fast/IIT24_AtomSim/franken/franken_cache/water/water_train.xyz
python scripts/rewrite_dset_keys.py --data-path=/leonardo_scratch/fast/IIT24_AtomSim/franken/franken_cache/water/ML_AB_dataset_2-val.xyz --save-path=/leonardo_scratch/fast/IIT24_AtomSim/franken/franken_cache/water/water_val.xyz
"""