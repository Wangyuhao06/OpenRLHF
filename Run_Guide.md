# Hopper Run Guide

This project should run on Hopper compute nodes, not on the login node.

## Core workflow

Use `PBS + Singularity + uv`.

- PBS: request GPU resources and submit jobs.
- Singularity: provide the base runtime, such as CUDA and PyTorch.
- uv: create and use the project-specific Python environment on top of the container.

This means Singularity is not the full project environment by itself. It gives the system stack and major AI packages, while the repo's own Python dependencies should still be installed with `uv` inside the job or interactive session.

## Current project

Check your project with:

```bash
hpc project
```

For this account on March 24, 2026, the available Hopper project is:

```bash
CFP04-CF-077
```

In PBS scripts, include:

```bash
#PBS -P CFP04-CF-077
```

## Interactive GPU debug

Use this only for short debugging runs:

```bash
qsub -I -P CFP04-CF-077 -l select=1:ngpus=1 -l walltime=01:00:00
```

After entering the GPU node:

```bash
module load singularity
singularity exec -e /app1/common/singularity-img/hopper/pytorch/pytorch_2.3.0_cuda_12.4_ngc_24.04.sif bash
```

Then inside the container, go to the repo and prepare the project environment:

```bash
cd /home/svu/e1137518/OpenRLHF
uv sync
uv run python your_script.py
```

## Batch job pattern

Typical Hopper jobs should follow this pattern:

```bash
#!/bin/bash
#PBS -P CFP04-CF-077
#PBS -j oe
#PBS -k oed
#PBS -N openrlhf_job
#PBS -l walltime=24:00:00
#PBS -l select=1:ngpus=1

cd $PBS_O_WORKDIR
module load singularity

image="/app1/common/singularity-img/hopper/pytorch/pytorch_2.3.0_cuda_12.4_ngc_24.04.sif"

singularity exec -e $image bash -lc '
  uv sync
  uv run python your_script.py
'
```

Submit with:

```bash
qsub job.pbs
```

## Useful commands

```bash
hpc project
qstat -awn1
qstat -fx <job_id>
qgpu_smi <job_id>
gstat
```

## Notes

- Do not run long training directly on `hopper-l-01`.
- Hopper expects the correct `#PBS -P Project-Name` in the job script.
- Do not set `CUDA_VISIBLE_DEVICES` manually in the program.
- Hopper containers are under `/app1/common/singularity-img/hopper/`.
