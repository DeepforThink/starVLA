# Implementation Log

## 2026-05-18: First-stage pretraining augmentation

Branch:

```text
data-rlinf-pipeline
```

Purpose:

```text
Add a low-risk, action-consistent visual augmentation baseline for pretraining data.
```

Changed files:

- `examples/SimplerEnv/train_files/data_registry/data_config.py`
- `examples/LIBERO/train_files/data_registry/data_config.py`
- `examples/Robotwin/train_files/data_registry/data_config.py`
- `configs/data_aug/weak_aug.yaml`
- `docs/competition/pretrain_augmentation.md`
- `docs/competition/data_contract.md`

Implementation:

- Added `weak_pretrain_video_transforms(...)` helpers to the OXE/SimplerEnv, LIBERO, and RobotWin data registries.
- Applied `VideoToTensor -> VideoResize(224, 224) -> VideoColorJitter -> VideoToNumpy` before existing state/action transforms.
- Standardized OXE/LIBERO color jitter to brightness `0.15`, contrast `0.15`, saturation `0.10`, hue `0.02`.
- Used weaker RobotWin jitter: brightness `0.12`, contrast `0.12`, saturation `0.08`, hue `0.01`.
- Replaced the stronger OXE Droid crop/jitter path with the conservative shared policy.

Design choice:

- No flip, rotation, perspective transform, object relocation, or erasing in the first stage.
- These transforms can change spatial semantics while leaving action labels unchanged, which risks corrupting VLA supervision.

Validation:

```text
python -m py_compile examples/SimplerEnv/train_files/data_registry/data_config.py examples/LIBERO/train_files/data_registry/data_config.py examples/Robotwin/train_files/data_registry/data_config.py
```

Local result:

```text
py_compile passed.
Full import check was blocked on the Windows local environment because training dependencies are not installed there:
accelerate, torch, torchvision, albumentations.
```

Data-dependent checks still needed on the SII server once datasets are available:

- one-batch dataloader smoke test for OXE/Bridge, LIBERO, and RobotWin
- verify video shape after transform is still accepted by the model path
- compare pretraining loss stability with and without weak augmentation
