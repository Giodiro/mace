import os
from pathlib import Path
import subprocess
import uuid
import yaml


def replace_name(orig_name, new_seed, new_samples):
    spl = orig_name.split("_")
    new_spl = []
    for spl_prt in spl:
        if "smpl" in spl_prt:
            new_spl.append(f"{new_samples}smpl")
        elif "seed" in spl_prt:
            new_spl.append(f"seed{new_seed}")
        else:
            new_spl.append(spl_prt)
    return "_".join(new_spl)


def replace_workdir(orig_wdir, new_name):
    orig_wdir = Path(orig_wdir)
    new_wdir = orig_wdir.with_name(new_name)
    return str(new_wdir)


def replace_trainfile(orig_trainfile, new_seed, new_samples):
    orig_tf = Path(orig_trainfile)
    new_name = f"train_{new_samples}_seed{new_seed}.xyz"
    new_tf = orig_tf.with_name(new_name)
    if not new_tf.is_file():
        raise FileNotFoundError(new_tf)
    return str(new_tf)


def get_num_epoch_from_samples(samples: int, steps: int):
    return min(steps // samples, 1000)


def run_one_exp(cfg_cache: Path, base_yaml: Path, seed: int, samples: int, num_steps: int):
    with open(base_yaml, "r") as fh:
        cfg = yaml.safe_load(fh)
    # replace name
    new_name = replace_name(cfg["name"], seed, samples)
    cfg["name"] = new_name
    # replace workdir
    new_wdir = replace_workdir(cfg["work_dir"], new_name)
    cfg["work_dir"] = new_wdir
    # replace train_file
    new_tf = replace_trainfile(cfg["train_file"], seed, samples)
    cfg["train_file"] = new_tf
    # modify seed
    cfg["seed"] = seed
    # modify batch size
    cfg["batch_size"] = min(cfg["batch_size"], samples)
    # modify num epochs
    cfg["max_num_epochs"] = get_num_epoch_from_samples(samples, num_steps)

    # write config to a file
    cfg_fn = cfg_cache / f"{uuid.uuid4().hex}.yaml"
    with open(cfg_fn, "w") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False)
    # Run with slurm
    print(f"Running experiment with yaml at {cfg_fn} - {seed=} {samples=}")
    script_path = Path("~/mace/slurm/run_with_cfg.sh").expanduser().resolve()
    subprocess.run([str(script_path), str(cfg_fn.resolve())])


def run_all_exp(cfg_cache: Path, base_yaml: Path, seeds: list[int], samples: list[int], num_steps: int):
    if not base_yaml.is_file():
        raise FileNotFoundError(base_yaml)
    cfg_cache.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        for sample in samples:
            run_one_exp(cfg_cache, base_yaml, seed, sample, num_steps)


if __name__ == "__main__":
    run_all_exp(
        Path("~/.cache/macefinetune").expanduser(),
        Path("~/mace/mace/configs/water_ft_head.yaml").expanduser(),
        seeds=[1, 2, 3, 4, 5],
        samples=[8, 32, 1024],
        num_steps=1024 * 20,  # 20 epochs for 1024 samples, rescale for the other num samples
    )