# Baseline Benchmarking

Scripts for easy Megatron and DeepSpeed benchmarking in a Docker environment.

## Getting Started

First, run `setup.sh` to initialize the Docker container.

Inside the container:

- Run `benchmark_megatron.sh` to benchmark Megatron.
- Run `benchmark_deepspeed.sh` to benchmark DeepSpeed.

See the script files to see which arguments to pass to each training script.

## Megatron Benchmarking

Megatron training and model configurations live under `megatron/configs`.

Example configs and a template config are provided under there, so you can either:

- reuse one of the example configs, or
- create your own config by starting from the template or copying an existing config.

## DeepSpeed Benchmarking

Currently, DeepSpeed benchmarking is done by the existing benchmark scripts under `deepspeed/`.

Use `benchmark_deepspeed.sh` to select which benchmark script to run. If no script is specified, it defaults to `benchmark_single.py`.

## Cleanup

To clean up the environment after benchmarking, including generated checkpoints and TensorBoard logs, run `cleanup.sh`.

## Current TODOs

- Decide which Megatron config settings to keep and which to remove to make life easier.
- Add interleaved configs via Megatron's virtual-stages argument to allow for 1f1b-interleaved.
- Megatron ZeroBubble?
- Improve the DeepSpeed benchmark workflow, possibly by creating a unified training script.
