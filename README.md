# Baseline Benchmarking

Scripts for easy Megatron and DeepSpeed benchmarking in a Docker environment.

## Getting Started

First, run `setup.sh` to initialize the Docker container.

Note: For TorchTitan, run `setup_torchtitan.sh` to initialize the Docker container.

Inside the container:

- Run `benchmark_megatron.sh` to benchmark Megatron.
- Run `benchmark_deepspeed.sh` to benchmark DeepSpeed.
- Run `benchmark_torchtitan.sh` to benchmark DeepSpeed.

See the script files to see which arguments to pass to each training script.

## Megatron Benchmarking

Megatron training and model configurations live under `megatron/configs`.

Example configs and a template config are provided under there, so you can either:

- reuse one of the example configs, or
- create your own config by starting from the template or copying an existing config.

## DeepSpeed Benchmarking

Currently, DeepSpeed benchmarking is done by the existing benchmark scripts under `deepspeed/`.

Use `benchmark_deepspeed.sh` to select which benchmark script to run. If no script is specified, it defaults to `benchmark_single.py`.

## TorchTitan Benchmarking
TorchTitan defines models inside a model_registry. This benchmarks imports the neccesary torchtitan modules to insert additional models and configs in `torchtitan/custom_configs.py`. Additionally, the training configuration can be defined in .toml files located in `torchtitan/configs/`

Example workflow:
```bash
bash setup_torchtitan.sh # use --no-build argument for subsequent runs 
NGPUS=4 bash ./benchmark_torchtitan.sh qwen3_30b_a3b_mini # run a custom MoE model with 4 GPUs.
```

Average stats from a run log:
```bash
LOG_RANK=1 NGPU=2 ./benchmark_torchtitan.sh llama3_2_1b_dp_ds_like -- --training.steps=50 2>&1 | tee torchtitan_run.log
./torchtitan_avg_stats.py torchtitan_run.log --skip-steps 5
```

## Cleanup

To clean up the environment after benchmarking, including generated checkpoints and TensorBoard logs, run `cleanup.sh`.

## Current TODOs

- Decide which Megatron config settings to keep and which to remove to make life easier.
- Add interleaved configs via Megatron's virtual-stages argument to allow for 1f1b-interleaved.
- Include scripts for DeepSpeed ZeRO-2 and ZeRO-3
- (if time) Improve the DeepSpeed benchmark workflow, possibly by creating a unified training script
