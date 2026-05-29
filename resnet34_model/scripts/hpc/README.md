# Sunway HPC — GPU Training

## Path A (try first): Docker / Singularity

1. Build: `docker build -t tomato-trainer ../..` (from `resnet34_model/`)
2. Convert to Singularity on HPC if required by cluster policy
3. Submit: `sbatch scripts/hpc/train_slurm.sh`

## Path B (fallback): Python venv

```bash
python -m venv ~/venvs/tomato-ml
source ~/venvs/tomato-ml/bin/activate
module load cuda   # confirm module name with Sunway HPC docs
pip install -r requirements.txt
python src/train.py
```

## Artifacts

Write to `outputs/best_model.pth` and `outputs/class_labels.json` on shared filesystem.

## TODO (fill when IT confirms)

- [ ] Slurm partition name
- [ ] GPU GRES syntax
- [ ] CUDA module name
- [ ] Singularity image path
