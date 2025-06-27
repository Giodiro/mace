#### Path / file download issues:
MACE really likes to download data in `~/.cache/mace`, which we cannot easily share.
I have linked that folder to another folder in scratch, you can do the same by running (assuming the cache doesn't already exist in your home directory)
```bash
ln -s /leonardo_scratch/fast/IIT24_AtomSim/franken/franken_cache/mace_cache ~/.cache/mace
```
so you shouldn't need to download anything new.

#### Configurations
 - **water_ft_all.yaml** finetune all of MACE
 - **water_ft_head.yaml** finetune the readout head
 - **water_ft_head_replay.yaml** finetune the readout head using multihead finetuning with replays from the MP data

The configs are setup to save results in `/leonardo_work/IIT24_AtomSim/franken/rebuttal/mace_finetune`

The data is all stored in `/leonardo_scratch/fast/IIT24_AtomSim/franken/franken_cache/water`:
 - `train_1024_seed1.xyz` contains 1k samples
 - `train_32_seed1.xyz` contains 32 samples
 - `train_8_seed1.xyz` contains 8 samples
these have been generated with the `create_subsampled_dataset.py` script.

#### Running ft

```bash
PYTHONPATH='.' python -m mace.cli.run_train --config mace/configs/water_ft_all.yaml
```

#### New scripts

For dataset stuff:
 - `rewrite_dset_keys.py` : rewrites energy to REF_energy, forces to REF_forces in the xyz
 - `create_subsampled_dataset.py` : creates subsampled datasets
