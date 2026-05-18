# Pretraining Data Augmentation

This note documents the first-stage augmentation used for broad StarVLA pretraining data.

## Scope

The current implementation applies online visual augmentation in the dataset transforms for:

- `examples/SimplerEnv/train_files/data_registry/data_config.py`
  - OXE Droid
  - OXE Bridge
  - OXE RT-1 / Fractal RT-1
- `examples/LIBERO/train_files/data_registry/data_config.py`
  - LIBERO
- `examples/Robotwin/train_files/data_registry/data_config.py`
  - RobotWin / Agilex
  - RobotWin 50-step action horizon
  - ARX X5

It does not rewrite parquet/LeRobot data on disk. Augmentation runs after the dataset sample is loaded and before it is passed to the model.

## Implemented Policy

The policy is intentionally conservative and action-consistent:

```text
VideoToTensor
-> VideoResize(224, 224)
-> VideoColorJitter
-> VideoToNumpy
```

Default OXE/LIBERO strength:

```text
brightness = 0.15
contrast   = 0.15
saturation = 0.10
hue        = 0.02
```

RobotWin uses a slightly weaker policy because it has multiple wrist/high cameras and more task-specific visual geometry:

```text
brightness = 0.12
contrast   = 0.12
saturation = 0.08
hue        = 0.01
```

`VideoColorJitter` is train-only in the StarVLA video transform implementation. Evaluation mode keeps deterministic resize/format conversion and skips random color jitter.

## Why This Is Safe

Robot actions encode spatial direction. The first-stage policy avoids transforms that can change spatial semantics without changing action labels:

- no horizontal flip
- no random rotation
- no perspective transform
- no object relocation
- no random erasing
- no generated replacement objects

The goal is to improve robustness to lighting, color, camera encoding, and mild visual domain shift while preserving observation-action alignment.

## Config Reference

`configs/data_aug/weak_aug.yaml` mirrors this online policy. It is a strategy record for experiments; the current training code reads augmentation from the `data_config.py` transform lists rather than dynamically parsing this YAML.

## Future Extensions

Add only after baseline pretraining/evaluation is available:

- language paraphrase tables for task instructions
- action statistics and smoothness checks
- failure-aware resampling from CALVIN/RLinf rollout logs
- region-aware augmentation if object masks or bounding boxes are available

