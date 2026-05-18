# Data Contract for CALVIN + StarVLA + RLinf

This document records the data interfaces used by our competition pipeline. It is intentionally practical: every training, augmentation, finetuning, and RL post-training step should stay compatible with this contract.

## 1. Current Scope

Primary benchmark:

- CALVIN ABC -> D long-horizon manipulation.

Primary codebase:

- `DeepforThink/starVLA`, branch `starVLA_dev`.

Current working branch:

- `data-rlinf-pipeline`.

Training and evaluation formats differ:

- Training: CALVIN converted to LeRobot format.
- Evaluation: original CALVIN format containing a `validation/` directory.

## 2. StarVLA Calvin Training Interface

Reference files:

- `examples/calvin/README.md`
- `examples/calvin/train_files/starvla_train_calvin.yaml`
- `examples/calvin/train_files/modality.json`
- `starVLA/dataloader/gr00t_lerobot/mixtures.py`

Default training config currently uses:

```yaml
framework:
  name: QwenPI
  action_model:
    action_dim: 7
    state_dim: 7
    action_horizon: 8
    repeated_diffusion_steps: 8
    num_inference_timesteps: 4

datasets:
  vla_data:
    dataset_py: lerobot_datasets
    include_state: false
    data_root_dir: playground/Datasets/calvin
    data_mix: calvin_task_D_D
    action_type: delta_qpos
    video_backend: torchvision_av
```

Open questions:

- `modality.json` exposes state indices 0..8, while the Calvin YAML has `state_dim: 7`.
- `include_state: false` means state may not be used by this baseline even though state schema exists.
- Need verify real LeRobot sample keys once data is available.

## 3. LeRobot Training Dataset Requirements

Expected dataset root:

```text
<lerobot_calvin_root>/
  meta/
    info.json
    stats.json
    tasks.jsonl
    modality.json
  data/
  videos/
```

The training dataset should include or map to:

| Field | Expected meaning | Current source/key |
|---|---|---|
| image | third-person RGB observation | `image` |
| wrist_image | wrist camera RGB observation | `wrist_image` |
| state | robot state/proprioception | `state` |
| actions | 7D robot action | `actions` |
| task description/index | language task label | `task_index` |

Action schema from `modality.json`:

```text
action[0] = x
action[1] = y
action[2] = z
action[3] = roll
action[4] = pitch
action[5] = yaw
action[6] = gripper
```

State schema from `modality.json`:

```text
state[0] = x
state[1] = y
state[2] = z
state[3] = roll
state[4] = pitch
state[5] = yaw
state[6] = pad
state[7] = gripper
```

Need verify:

- Whether model actually consumes state.
- Whether action is delta pose, delta qpos, or another normalized representation.
- Whether `stats.json` stores min/max or mean/std for action unnormalization.
- Whether task descriptions are text strings or indices that need lookup.

## 4. Data Mix Contract

StarVLA Calvin README says data mix must be configured in:

```text
starVLA/dataloader/gr00t_lerobot/mixtures.py
```

Expected key:

```python
"calvin_task_D_D": [
    ("task_D_D", 1.0, "libero_franka"),
]
```

Need verify on platform:

- Actual dataset folder name.
- Whether `task_D_D` contains ABC training data, D validation data, or a converted split naming convention.
- Whether we need additional keys for `calvin_abc`, `calvin_d`, or mixed pretraining datasets.

## 5. Evaluation Dataset Contract

StarVLA Calvin evaluation uses original CALVIN format.

Expected path:

```text
<calvin_raw_root>/
  validation/
  calvin_models/
  ...
```

Evaluation files:

- `examples/calvin/eval_files/eval_calvin.py`
- `examples/calvin/eval_files/eval_calvin.sh`
- `examples/calvin/eval_files/run_policy_server.sh`
- `examples/calvin/eval_files/eval_sequences.json`

Need verify:

- `dataset_path`
- `calvin_config_path`
- `eval_sequences_path`
- checkpoint path in both policy server and eval script.
- whether videos are saved.
- whether failed step and per-episode success can be exported.

## 6. Policy Server Output Contract

StarVLA Calvin README indicates evaluation is two-process:

1. StarVLA policy server.
2. CALVIN environment evaluator.

From existing eval interface, action should be a 7D vector assembled from:

```text
world_vector   -> 3D translation
rotation_delta -> 3D rotation
open_gripper   -> 1D gripper
```

For action chunk policies:

```text
actions: [B, T, D_action]
T = action_horizon / num_action_chunks
D_action = 7
```

Evaluation likely executes a subset of the predicted chunk before replanning. Need confirm `replan_steps` in eval scripts.

## 7. Data Augmentation Contract

Implemented first-stage pretraining augmentation:

- OXE/SimplerEnv, LIBERO, and RobotWin data registries now apply weak online visual augmentation in `data_config.py`.
- The implemented path is `VideoToTensor -> VideoResize(224, 224) -> VideoColorJitter -> VideoToNumpy`.
- OXE/LIBERO jitter strength: brightness `0.15`, contrast `0.15`, saturation `0.10`, hue `0.02`.
- RobotWin jitter strength: brightness `0.12`, contrast `0.12`, saturation `0.08`, hue `0.01`.
- `VideoColorJitter` is train-only; eval skips random color jitter.

Allowed default augmentations must be action-consistent:

- brightness/contrast/saturation jitter.
- slight blur.
- slight noise.
- JPEG compression.
- slight crop/resize.
- small random erasing.

Avoid by default:

- horizontal flip.
- large rotation.
- strong perspective transform.
- object relocation without corresponding action transform.

Reason:

```text
Robot actions encode spatial direction. Augmentations that change spatial semantics without transforming action labels break observation-action alignment.
```

## 8. RLinf Contract To Confirm

Public RLinf docs provide:

- CALVIN + OpenVLA-OFT / pi0 / pi0.5 PPO/GRPO examples.
- StarVLA + LIBERO GRPO example.

Not yet confirmed:

- StarVLA + CALVIN + RLinf end-to-end config.
- whether rollout exports per-step reward trace.
- whether rollout exports failed step.
- whether rollout exports action chunks.
- whether rollout exports failure videos with machine-readable metadata.

If missing, we should add JSONL logging around rollout/eval.

Minimum rollout record:

```json
{
  "episode_id": "...",
  "rollout_id": "...",
  "chain_step": 0,
  "task_name": "...",
  "reward": 0.0,
  "success": false,
  "failed_step": 123,
  "failure_type": "unknown",
  "action_chunk_norm": null,
  "video_path": "...",
  "use_for_reweight": true,
  "use_for_replay": false
}
```

## 9. Immediate Checks Once Data Is Available

Run dataset inspection before any training:

```text
num episodes
num frames
episode length distribution
task frequency
image keys and resolution
state shape and stats
action shape and stats
NaN / Inf count
missing files
bad videos
normalization stats
```

Then decide:

- task-balanced sampling weights.
- action-consistent augmentation strength.
- whether failure-aware finetuning can be prepared after baseline eval.
