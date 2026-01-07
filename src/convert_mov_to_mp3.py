#!/usr/bin/env python3
"""Convert MOV video files to high-quality MP3 audio on macOS."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract audio from a MOV file and encode to MP3 with the highest "
            "quality settings supported by ffmpeg."
        )
    )
    parser.add_argument("input", type=Path, help="Path to the input .mov file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional output .mp3 path (defaults to input name with .mp3)",
    )
    return parser


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit(
            "ffmpeg was not found. Install it with: brew install ffmpeg"
        )


def build_output_path(input_path: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path
    return input_path.with_suffix(".mp3")


def convert_mov_to_mp3(input_path: Path, output_path: Path) -> None:
    command = [
        "ffmpeg",
        "-i",
        str(input_path),
        "-vn",
        "-c:a",
        "libmp3lame",
        "-q:a",
        "0",
        "-map_metadata",
        "0",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path: Path = args.input
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    output_path = build_output_path(input_path, args.output)

    ensure_ffmpeg()
    convert_mov_to_mp3(input_path, output_path)

    print(f"Saved MP3 to: {output_path}")


if __name__ == "__main__":
    main()
