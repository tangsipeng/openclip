#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert FunASR JSON output into an SRT subtitle file."
    )
    parser.add_argument("input_json", help="Path to the FunASR JSON output.")
    parser.add_argument("output_srt", help="Path to the output SRT file.")
    parser.add_argument(
        "--max-line-length",
        type=int,
        default=20,
        help="Soft limit used when wrapping subtitle lines.",
    )
    parser.add_argument(
        "--lines-per-cue",
        type=int,
        default=2,
        help="Maximum number of lines per subtitle cue.",
    )
    return parser.parse_args()


def srt_timestamp(total_ms: int) -> str:
    total_ms = max(0, int(total_ms))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def wrap_text(text: str, max_line_length: int) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []

    lines: list[str] = []
    break_chars = set(" ，。！？；：,.!?;:、")
    remaining = text

    while remaining:
        if len(remaining) <= max_line_length:
            lines.append(remaining.strip())
            break

        split_at = max_line_length
        for idx in range(max_line_length, 0, -1):
            if remaining[idx - 1] in break_chars:
                split_at = idx
                break

        line = remaining[:split_at].strip()
        if not line:
            line = remaining[:max_line_length].strip()
            split_at = max_line_length

        lines.append(line)
        remaining = remaining[split_at:].lstrip()

    lines = [line for line in lines if line]
    leading_punc = set("，。！？；：,.!?;:、")
    for idx in range(1, len(lines)):
        while lines[idx] and lines[idx][0] in leading_punc:
            lines[idx - 1] += lines[idx][0]
            lines[idx] = lines[idx][1:].lstrip()
    return [line for line in lines if line]


def split_segment_to_cues(
    text: str,
    start_ms: int,
    end_ms: int,
    max_line_length: int,
    lines_per_cue: int,
) -> list[tuple[int, int, str]]:
    lines = wrap_text(text, max_line_length)
    if not lines:
        return []

    blocks = [
        lines[idx : idx + lines_per_cue]
        for idx in range(0, len(lines), lines_per_cue)
    ]
    block_texts = ["\n".join(block) for block in blocks]
    if len(block_texts) == 1:
        return [(start_ms, end_ms, block_texts[0])]

    total_duration = max(end_ms - start_ms, len(block_texts) * 500)
    weights = [max(len(block.replace("\n", "")), 1) for block in block_texts]
    total_weight = sum(weights)

    cues: list[tuple[int, int, str]] = []
    cursor = start_ms
    for idx, (block_text, weight) in enumerate(zip(block_texts, weights), start=1):
        if idx == len(block_texts):
            block_end = end_ms
        else:
            block_duration = max(int(total_duration * weight / total_weight), 500)
            block_end = min(end_ms, cursor + block_duration)
        cues.append((cursor, block_end, block_text))
        cursor = block_end
    return cues


def main() -> None:
    args = parse_args()
    input_json = Path(args.input_json)
    output_srt = Path(args.output_srt)

    payload = json.loads(input_json.read_text(encoding="utf-8"))
    segments = payload.get("segments", [])

    output_srt.parent.mkdir(parents=True, exist_ok=True)

    with output_srt.open("w", encoding="utf-8") as fh:
        index = 1
        for segment in segments:
            text = str(segment.get("text", "")).strip()
            start_ms = int(segment.get("start_ms", 0))
            end_ms = int(segment.get("end_ms", 0))
            if not text or end_ms <= start_ms:
                continue

            cues = split_segment_to_cues(
                text=text,
                start_ms=start_ms,
                end_ms=end_ms,
                max_line_length=args.max_line_length,
                lines_per_cue=args.lines_per_cue,
            )
            for cue_start, cue_end, cue_text in cues:
                if cue_end <= cue_start:
                    continue
                fh.write(f"{index}\n")
                fh.write(f"{srt_timestamp(cue_start)} --> {srt_timestamp(cue_end)}\n")
                fh.write(f"{cue_text}\n\n")
                index += 1


if __name__ == "__main__":
    main()
