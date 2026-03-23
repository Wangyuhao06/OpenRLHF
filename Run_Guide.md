# HPC Execution Notes

This project is intended to **run on the NUS HPC system**, not on a normal standalone server.

This project's location on the NUS HPC system is `/home/svu/e1137518/OpenRLHF`.

## Execution model
- Development can happen locally or via coding agents.
- Final training / heavy computation should be submitted to the **HPC cluster** through **PBS Pro**.
- SSH login lands on a **login node**, which is only for light work:
  - editing files
  - preparing environments
  - submitting jobs
  - checking logs / job status
- Long-running jobs must **not** be run directly on the login node.
- Actual compute runs on allocated **compute nodes** after job submission.

## Environment strategy
Use **module + uv**.

### Recommended pattern
1. Load required system modules first, such as:
   - Python
   - CUDA
   - compiler toolchains if needed
2. Then use `uv` for Python project dependency management.

Typical idea:

```bash
module load python/3.12.3
module load cuda12.4/toolkit/12.4.1
uv sync
uv run python train.py
````

## Notes on environments

* The project environment lives in the shared filesystem and should be accessible from both login and compute nodes.
* Prefer `uv sync` to reproduce the Python environment from project metadata.
* Avoid assuming that local machine paths or locally installed system packages exist on the HPC nodes.

## Job submission

* Jobs are submitted with **PBS Pro** (`qsub`).
* Monitor jobs with `qstat`.
* Cancel jobs with `qdel`.
* Use log files for stdout/stderr and monitor progress with `tail -f`.

## GPU usage

* GPU resources are requested in the PBS job script.
* Multi-GPU jobs are supported in principle by requesting the required number of GPUs in the resource specification.
* The exact allowed GPU count depends on cluster policy, queue limits, and node configuration.

## Operational assumptions for the agent

* Do not assume interactive long-running execution on the login node.
* Prepare code to run as a **batch job**.
* Keep commands reproducible and non-interactive where possible.
* Prefer writing scripts that can be launched by PBS and whose progress is visible from log output.