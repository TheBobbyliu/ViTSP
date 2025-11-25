#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Iterable, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a ShareGPT-style dataset from saved LLM records. "
            "Walks every run folder under llm_records/ and keeps only samples "
            "with objective_delta < 0 and no errors."
        )
    )
    parser.add_argument(
        "--records-root",
        default="llm_records",
        help="Root directory containing run folders (default: llm_records).",
    )
    parser.add_argument(
        "--tasks",
        required=True,
        help="One or more task names (matching the 'task_name' field) to extract, separated by ,",
    )
    parser.add_argument(
        "--output-name",
        required=True,
        help=(
            "Name for the dataset. Output will be written to "
            "datasets/<output-name>/<output-name>.json and images/"
        ),
    )
    return parser.parse_args()


def resolve_records_root(arg_path: str, repo_root: Path) -> Path:
    provided = Path(arg_path)
    if not provided.is_absolute():
        provided = (repo_root / provided).resolve()
    if provided.is_dir():
        return provided
    raise FileNotFoundError(f"Could not find records root '{arg_path}'.")


def extract_prompt_and_image(record: dict) -> tuple[str, Optional[str]]:
    prompt_text = (record.get("prompt") or "").strip()
    image_path = record.get("image_path")
    messages = record.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if message.get("role") != "user":
                continue
            text_parts = []
            for part in message.get("content", []):
                part_type = part.get("type")
                if part_type == "text":
                    text = (part.get("text") or "").strip()
                    if text:
                        text_parts.append(text)
                elif part_type in {"image_path", "image_url"} and not image_path:
                    image_path = part.get("path") or part.get("image_url")
            if text_parts:
                prompt_text = "\n".join(text_parts).strip()
            break
    return prompt_text, image_path


def iter_candidate_image_paths(
    hint_path: Optional[str], records_path: Path, record_id: int
) -> Iterable[Path]:
    if hint_path:
        raw = Path(hint_path)
        if raw.is_absolute():
            yield raw
        else:
            yield raw
            yield (Path.cwd() / raw).resolve()
        if raw.name:
            yield records_path / "images" / raw.name
    yield records_path / "images" / f"{record_id}.png"


def resolve_image_file(
    hint_path: Optional[str], records_path: Path, record_id: int
) -> Optional[Path]:
    seen = set()
    for candidate in iter_candidate_image_paths(hint_path, records_path, record_id):
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    records_root = resolve_records_root(args.records_root, repo_root)
    dataset_dir = repo_root / "datasets" / args.output_name
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    output_path = dataset_dir / f"{args.output_name}.json"

    target_tasks = set(args.tasks.split(','))
    dataset = []
    kept = 0
    skipped = 0

    for records_path in sorted(p for p in records_root.iterdir() if p.is_dir()):
        data_dir = records_path / "data"
        if not data_dir.is_dir():
            continue

        for json_file in sorted(data_dir.glob("*.json")):
            record = json.loads(json_file.read_text())
            task_name = record.get("task_name")
            if not task_name:
                continue
            task_name = task_name.split("/")[-1].split(".tsp")[0]
            if task_name not in target_tasks:
                continue

            if record.get("error"):
                skipped += 1
                continue

            raw_delta = record.get("objective_delta")
            try:
                delta = float(raw_delta)
            except (TypeError, ValueError):
                skipped += 1
                continue
            if math.isnan(delta) or delta >= 0:
                skipped += 1
                continue

            response = (record.get("response_text") or "").strip()
            if not response:
                skipped += 1
                continue

            prompt_text, image_hint = extract_prompt_and_image(record)
            # another sieve:
            if len(prompt_text) > 10000:
                print("Skip data because too long", len(prompt_text))
                skipped += 1
                continue
            image_file = resolve_image_file(
                image_hint, records_path, record.get("id", 0)
            )
            if not image_file:
                print(
                    f"[warn] unable to locate image for record {json_file.name}",
                    file=sys.stderr,
                )
                skipped += 1
                continue

            record_id = record.get("id")
            dest_name = (
                f"{records_path.name}_{record_id}.png"
                if record_id is not None
                else f"{records_path.name}_{json_file.stem}.png"
            )
            dest_path = images_dir / dest_name
            if not dest_path.exists():
                shutil.copy2(image_file, dest_path)

            image_rel_path = str(
                Path("datasets") / args.output_name / "images" / dest_name
            )
            conversations = [
                {
                    "from": "human",
                    "value": "<image>" + (f"\n{prompt_text}" if prompt_text else ""),
                },
                {"from": "gpt", "value": response},
            ]
            dataset.append(
                {
                    "id": f"sample_{len(dataset)}",
                    "images": [image_rel_path],
                    "conversations": conversations,
                }
            )
            kept += 1

    if not dataset:
        raise SystemExit("No samples matched the provided filters.")

    output_path.write_text(json.dumps(dataset, indent=2))
    print(
        f"Wrote {kept} samples (skipped {skipped}) to {output_path} with images in {images_dir}"
    )


if __name__ == "__main__":
    main()
