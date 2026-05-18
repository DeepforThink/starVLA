#!/usr/bin/env python3
"""Analyze LIBERO LeRobot data for trajectory and language augmentation.

This script is intentionally read-only. It first reuses LeRobot metadata, then
adds the temporal statistics that are usually missing from meta/stats.json:
action deltas, jerk, gripper switches, task frequency, and length distribution.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ACTION_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


@dataclass
class ActionSpec:
    name: str
    source: str
    start: int | None = None
    end: int | None = None


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def read_tasks(dataset_root: Path) -> dict[int, str]:
    tasks_jsonl = dataset_root / "meta" / "tasks.jsonl"
    rows = read_jsonl(tasks_jsonl)
    if rows:
        return {int(row["task_index"]): str(row.get("task", "")) for row in rows}

    tasks_parquet = dataset_root / "meta" / "tasks.parquet"
    if tasks_parquet.exists():
        pd = import_pandas()
        df = pd.read_parquet(tasks_parquet)
        return {int(row.task_index): str(row.task) for row in df.itertuples(index=False)}
    return {}


def import_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "pandas/pyarrow is required for parquet trajectory statistics. "
            "Install them in the SII training environment, or run with --metadata-only."
        ) from exc
    return pd


def discover_parquets(dataset_root: Path, limit_episodes: int | None) -> list[Path]:
    parquet_paths = sorted((dataset_root / "data").glob("*/*.parquet"))
    if limit_episodes is not None:
        parquet_paths = parquet_paths[:limit_episodes]
    return parquet_paths


def infer_action_specs(dataset_root: Path) -> list[ActionSpec]:
    """Infer action vector layout from gr00t modality.json when available."""
    modality = read_json(dataset_root / "meta" / "modality.json")
    action_meta = modality.get("action", {})
    specs: list[ActionSpec] = []

    if isinstance(action_meta, dict):
        for name, meta in action_meta.items():
            if not isinstance(meta, dict):
                continue
            source = meta.get("original_key") or meta.get("key") or f"action.{name}"
            start = meta.get("start")
            end = meta.get("end")
            specs.append(ActionSpec(name=str(name), source=str(source), start=start, end=end))

    if specs:
        return specs

    return [ActionSpec(name=name, source=f"action.{name}") for name in ACTION_NAMES]


def extract_action_matrix(df: Any, specs: list[ActionSpec]) -> tuple[np.ndarray | None, list[str]]:
    columns = set(map(str, df.columns))

    if all(spec.source in columns for spec in specs):
        arrays = []
        names = []
        for spec in specs:
            values = np.asarray(df[spec.source].tolist(), dtype=np.float32)
            if values.ndim == 1:
                values = values[:, None]
            if spec.start is not None and spec.end is not None and values.shape[1] > 1:
                values = values[:, int(spec.start): int(spec.end)]
            arrays.append(values)
            names.extend(expand_names(spec.name, values.shape[1]))
        return np.concatenate(arrays, axis=1), names

    if "action" in columns:
        values = np.asarray(df["action"].tolist(), dtype=np.float32)
        return values, expand_names("action", values.shape[1])

    action_cols = [col for col in df.columns if str(col).startswith("action")]
    if action_cols:
        arrays = []
        names = []
        for col in sorted(action_cols):
            values = np.asarray(df[col].tolist(), dtype=np.float32)
            if values.ndim == 1:
                values = values[:, None]
            arrays.append(values)
            names.extend(expand_names(str(col), values.shape[1]))
        return np.concatenate(arrays, axis=1), names

    return None, []


def expand_names(prefix: str, dim: int) -> list[str]:
    if dim == 1:
        return [prefix]
    return [f"{prefix}[{i}]" for i in range(dim)]


def vector_stats(values: np.ndarray) -> dict[str, list[float]]:
    if values.size == 0:
        return {}
    return {
        "mean": np.nanmean(values, axis=0).round(6).tolist(),
        "std": np.nanstd(values, axis=0).round(6).tolist(),
        "min": np.nanmin(values, axis=0).round(6).tolist(),
        "max": np.nanmax(values, axis=0).round(6).tolist(),
        "q01": np.nanquantile(values, 0.01, axis=0).round(6).tolist(),
        "q99": np.nanquantile(values, 0.99, axis=0).round(6).tolist(),
    }


def norm_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


def task_template(text: str) -> str:
    cleaned = " ".join(text.lower().strip().split())
    for token in [",", ".", "!", "?"]:
        cleaned = cleaned.replace(token, "")
    return cleaned


def analyze_trajectories(dataset_root: Path, limit_episodes: int | None) -> dict[str, Any]:
    pd = import_pandas()
    parquet_paths = discover_parquets(dataset_root, limit_episodes)
    specs = infer_action_specs(dataset_root)
    tasks = read_tasks(dataset_root)

    all_actions = []
    all_deltas = []
    all_jerks = []
    all_continuous_deltas = []
    all_continuous_jerks = []
    delta_norms = []
    jerk_norms = []
    continuous_delta_norms = []
    continuous_jerk_norms = []
    action_norms = []
    gripper_values = []
    gripper_switches = 0
    gripper_steps = 0
    episode_lengths = []
    task_frame_counts: Counter[str] = Counter()
    task_episode_counts: Counter[str] = Counter()
    action_names: list[str] = []
    missing_action_files = []
    spike_candidates = []
    continuous_spike_candidates = []

    for parquet_path in parquet_paths:
        df = pd.read_parquet(parquet_path)
        actions, names = extract_action_matrix(df, specs)
        if actions is None:
            missing_action_files.append(str(parquet_path))
            continue
        if not action_names:
            action_names = names

        episode_lengths.append(int(actions.shape[0]))
        all_actions.append(actions)
        action_norms.extend(np.linalg.norm(actions, axis=1).tolist())

        if actions.shape[0] > 1:
            deltas = np.diff(actions, axis=0)
            all_deltas.append(deltas)
            norms = np.linalg.norm(deltas, axis=1)
            delta_norms.extend(norms.tolist())
            if norms.size:
                spike_candidates.append((float(np.max(norms)), str(parquet_path)))

            continuous_deltas = deltas[:, :-1] if deltas.shape[1] > 1 else deltas
            all_continuous_deltas.append(continuous_deltas)
            continuous_norms = np.linalg.norm(continuous_deltas, axis=1)
            continuous_delta_norms.extend(continuous_norms.tolist())
            if continuous_norms.size:
                continuous_spike_candidates.append((float(np.max(continuous_norms)), str(parquet_path)))

        if actions.shape[0] > 2:
            jerks = np.diff(actions, n=2, axis=0)
            all_jerks.append(jerks)
            jerk_norms.extend(np.linalg.norm(jerks, axis=1).tolist())

            continuous_jerks = jerks[:, :-1] if jerks.shape[1] > 1 else jerks
            all_continuous_jerks.append(continuous_jerks)
            continuous_jerk_norms.extend(np.linalg.norm(continuous_jerks, axis=1).tolist())

        if actions.shape[1] >= 1:
            gripper = actions[:, -1]
            gripper_values.extend(gripper.tolist())
            if len(gripper) > 1:
                gripper_switches += int(np.count_nonzero(np.diff(gripper) != 0))
                gripper_steps += int(len(gripper) - 1)

        task_name = get_task_name(df, tasks)
        task_frame_counts[task_name] += int(actions.shape[0])
        task_episode_counts[task_name] += 1

    action_matrix = np.concatenate(all_actions, axis=0) if all_actions else np.empty((0, 0))
    delta_matrix = np.concatenate(all_deltas, axis=0) if all_deltas else np.empty((0, 0))
    jerk_matrix = np.concatenate(all_jerks, axis=0) if all_jerks else np.empty((0, 0))
    continuous_delta_matrix = (
        np.concatenate(all_continuous_deltas, axis=0) if all_continuous_deltas else np.empty((0, 0))
    )
    continuous_jerk_matrix = (
        np.concatenate(all_continuous_jerks, axis=0) if all_continuous_jerks else np.empty((0, 0))
    )

    spike_candidates = sorted(spike_candidates, reverse=True)[:10]
    continuous_spike_candidates = sorted(continuous_spike_candidates, reverse=True)[:10]
    gripper_arr = np.asarray(gripper_values, dtype=np.float32) if gripper_values else np.asarray([])

    return {
        "num_parquet_files": len(parquet_paths),
        "num_analyzed_episodes": len(episode_lengths),
        "num_frames": int(sum(episode_lengths)),
        "action_names": action_names,
        "episode_length": summarize_scalar(episode_lengths),
        "action_stats": vector_stats(action_matrix),
        "delta_stats": vector_stats(delta_matrix),
        "jerk_stats": vector_stats(jerk_matrix),
        "continuous_delta_stats": vector_stats(continuous_delta_matrix),
        "continuous_jerk_stats": vector_stats(continuous_jerk_matrix),
        "action_norm": norm_summary(action_norms),
        "delta_norm": norm_summary(delta_norms),
        "jerk_norm": norm_summary(jerk_norms),
        "continuous_delta_norm": norm_summary(continuous_delta_norms),
        "continuous_jerk_norm": norm_summary(continuous_jerk_norms),
        "gripper": summarize_gripper(gripper_arr, gripper_switches, gripper_steps),
        "task_frame_counts": dict(task_frame_counts.most_common()),
        "task_episode_counts": dict(task_episode_counts.most_common()),
        "top_delta_spike_candidates": [
            {"max_delta_norm": score, "parquet": path} for score, path in spike_candidates
        ],
        "top_continuous_delta_spike_candidates": [
            {"max_continuous_delta_norm": score, "parquet": path}
            for score, path in continuous_spike_candidates
        ],
        "missing_action_files": missing_action_files[:20],
    }


def get_task_name(df: Any, tasks: dict[int, str]) -> str:
    if "task_index" not in df.columns or not tasks:
        return "unknown"
    values = df["task_index"].dropna().unique()
    if len(values) == 0:
        return "unknown"
    task_index = int(values[0])
    return tasks.get(task_index, f"task_index={task_index}")


def summarize_scalar(values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
    }


def summarize_gripper(values: np.ndarray, switches: int, steps: int) -> dict[str, float]:
    if values.size == 0:
        return {}
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "switch_rate": float(switches / steps) if steps else 0.0,
        "near_zero_ratio": float(np.mean(np.isclose(values, 0.0, atol=1e-4))),
        "near_one_ratio": float(np.mean(np.isclose(values, 1.0, atol=1e-4))),
    }


def analyze_language(dataset_root: Path) -> dict[str, Any]:
    tasks = read_tasks(dataset_root)
    templates = Counter(task_template(text) for text in tasks.values())
    verbs = Counter()
    objects = Counter()
    for text in tasks.values():
        words = task_template(text).split()
        if words:
            verbs[words[0]] += 1
        for word in words:
            if word not in {"the", "a", "an", "to", "on", "in", "into", "of", "and"}:
                objects[word] += 1
    return {
        "num_tasks": len(tasks),
        "tasks": tasks,
        "template_counts": dict(templates.most_common()),
        "first_verb_counts": dict(verbs.most_common()),
        "content_word_counts": dict(objects.most_common(50)),
    }


def summarize_metadata(dataset_root: Path) -> dict[str, Any]:
    info = read_json(dataset_root / "meta" / "info.json")
    stats = read_json(dataset_root / "meta" / "stats.json")
    episodes = read_jsonl(dataset_root / "meta" / "episodes.jsonl")
    return {
        "info": {
            key: info.get(key)
            for key in [
                "codebase_version",
                "robot_type",
                "total_episodes",
                "total_frames",
                "total_tasks",
                "fps",
                "data_path",
                "video_path",
            ]
            if key in info
        },
        "stats_keys": sorted(stats.keys()),
        "num_episode_meta_rows": len(episodes),
    }


def write_markdown(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# LIBERO Data Report",
        "",
        "## Dataset Metadata",
        "",
        fenced_json(report["metadata"]),
        "",
        "## Language Summary",
        "",
        f"- Number of tasks: {report['language'].get('num_tasks', 0)}",
        f"- First verbs: `{report['language'].get('first_verb_counts', {})}`",
        "",
        "### Task Instructions",
        "",
    ]
    for idx, text in report["language"].get("tasks", {}).items():
        lines.append(f"- `{idx}`: {text}")

    if "trajectory" in report:
        traj = report["trajectory"]
        lines.extend(
            [
                "",
                "## Trajectory Summary",
                "",
                f"- Analyzed episodes: {traj.get('num_analyzed_episodes')}",
                f"- Frames: {traj.get('num_frames')}",
                f"- Action names: `{traj.get('action_names')}`",
                f"- Episode length: `{traj.get('episode_length')}`",
                f"- Action norm: `{round_float_dict(traj.get('action_norm', {}))}`",
                f"- Delta norm: `{round_float_dict(traj.get('delta_norm', {}))}`",
                f"- Continuous delta norm: `{round_float_dict(traj.get('continuous_delta_norm', {}))}`",
                f"- Jerk norm: `{round_float_dict(traj.get('jerk_norm', {}))}`",
                f"- Continuous jerk norm: `{round_float_dict(traj.get('continuous_jerk_norm', {}))}`",
                f"- Gripper: `{round_float_dict(traj.get('gripper', {}))}`",
                "",
                "### Task Episode Counts",
                "",
            ]
        )
        for task, count in traj.get("task_episode_counts", {}).items():
            lines.append(f"- {task}: {count}")

        lines.extend(["", "### Top Delta Spike Candidates", ""])
        for item in traj.get("top_delta_spike_candidates", []):
            lines.append(f"- `{item['max_delta_norm']:.6f}`: `{item['parquet']}`")

        lines.extend(["", "### Top Continuous Delta Spike Candidates", ""])
        for item in traj.get("top_continuous_delta_spike_candidates", []):
            lines.append(f"- `{item['max_continuous_delta_norm']:.6f}`: `{item['parquet']}`")

    lines.extend(
        [
            "",
            "## Augmentation Implications",
            "",
            "- Use task counts to decide whether LIBERO pretraining needs task-balanced sampling.",
            "- Use delta/jerk spikes to find trajectories that should be filtered, downweighted, or inspected.",
            "- Use gripper switch statistics to decide whether close/open transitions need oversampling.",
            "- Use task instruction templates to build a reviewed paraphrase table.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def round_float_dict(values: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in values.items():
        if isinstance(value, float):
            result[key] = round(value, 6)
        else:
            result[key] = value
    return result


def fenced_json(value: Any) -> str:
    return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n```"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="LIBERO LeRobot dataset root.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/competition/libero_data_report.md"),
        help="Markdown report path.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional JSON summary path.",
    )
    parser.add_argument(
        "--limit-episodes",
        type=int,
        default=None,
        help="Limit parquet episodes for a quick smoke run.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only read metadata/tasks; skip parquet trajectory statistics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.data_root.resolve()
    if not dataset_root.exists():
        raise SystemExit(f"Dataset root does not exist: {dataset_root}")

    report: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "metadata": summarize_metadata(dataset_root),
        "language": analyze_language(dataset_root),
    }
    if not args.metadata_only:
        report["trajectory"] = analyze_trajectories(dataset_root, args.limit_episodes)

    write_markdown(report, args.output)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote report to {args.output}")
    if args.json_output:
        print(f"Wrote JSON summary to {args.json_output}")


if __name__ == "__main__":
    main()
