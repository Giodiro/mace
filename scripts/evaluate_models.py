import json
from pathlib import Path
import time
import ase
import ase.io
import numpy as np
import torch

from mace.calculators.mace import MACECalculator


def eval_model(model_path: Path, data: list[ase.Atoms], device):
    calc = MACECalculator(
        model_paths=str(model_path.expanduser().resolve()),
        device=device,
        default_dtype="float64"
    )
    t_s = time.time()
    energy_mae, energy_rmse = [], []
    forces_mae, forces_rmse = [], []
    for atoms in data:
        num_atoms = len(atoms)
        true_energy = atoms.get_potential_energy()
        true_forces = atoms.get_forces()
        calc.calculate(atoms)
        res = calc.results
        pred_energy = res["energy"]
        pred_forces = res["forces"]
        # error
        energy_mae.append(
            (1000 * np.abs(true_energy - pred_energy) / num_atoms)
        )
        energy_rmse.append(
            np.square((true_energy - pred_energy) / num_atoms)
        )
        forces_mae.append(
            np.mean(1000 * np.abs(true_forces - pred_forces))
        )
        forces_rmse.append(
            np.square(true_forces - pred_forces).mean()
        )
        # modify atoms for new forces
        atoms.info["REF_energy"] = pred_energy
        atoms.arrays["REF_forces"] = pred_forces
    t_elapsed = time.time() - t_s

    energy_mae = np.mean(energy_mae)
    energy_rmse = np.sqrt(np.mean(energy_rmse)) * 1000
    forces_mae = np.mean(forces_mae)
    forces_rmse = np.sqrt(np.mean(forces_rmse)) * 1000

    # Save output atoms and stats
    out_atom_fp = model_path.parent / "results" / "water_val.xyz"
    ase.io.write(out_atom_fp, data)
    out_stats_fp = model_path.parent / "results" / "val_stats.json"
    with open(out_stats_fp, "w") as fh:
        json.dump({
            "energy_mae": energy_mae,
            "energy_rmse": energy_rmse,
            "forces_mae": forces_mae,
            "forces_rmse": forces_rmse,
            "elapsed_time": t_elapsed,
        }, fh)
    print(f"Saved predicted trajectory to {out_atom_fp} and error stats to {out_stats_fp}")
    print(f"Evaluation complete in {t_elapsed:.2f}s")


def eval_all_models(base_model_path: Path, val_data_path: Path):
    val_data = ase.io.read(val_data_path, index=":")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for seed in [1, 2, 3, 4, 5]:
        for samples in [8, 32, 1024]:
            for ft_mode in ["all", "head"]:
                model_name = f"water_{samples}smpl_seed{seed}_fw10_lr1e-2_ft{ft_mode}"
                model_path = base_model_path / model_name / f"{model_name}.model"
                if not model_path.is_file():
                    raise FileNotFoundError(model_path)
                print(f"Running evaluation of model {model_name}")
                eval_model(model_path, val_data, device)
                print()


if __name__ == "__main__":
    eval_all_models(
        Path("/leonardo_work/IIT24_AtomSim/franken/rebuttal/mace_finetune"),
        Path("/leonardo_scratch/fast/IIT24_AtomSim/franken/franken_cache/water/water_val.xyz"),
    )
