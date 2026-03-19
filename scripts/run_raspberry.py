#!/usr/bin/env python3
"""Удобный запуск inference без shell-обёртки."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_default_model() -> Path:
    ncnn_dir = PROJECT_ROOT / "models" / "deploy" / "best_ncnn_model"
    if (ncnn_dir / "model.ncnn.param").exists() and (ncnn_dir / "model.ncnn.bin").exists():
        return ncnn_dir
    return PROJECT_ROOT / "models" / "deploy" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RaspberryRoadSign inference")
    parser.add_argument("--source", default="0", help="0/1/.. или путь к видео")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output" / "raspberry_output.mp4",
        help="Путь к выходному видео",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=resolve_default_model(),
        help="Путь к .pt или папке best_ncnn_model",
    )
    parser.add_argument("--device", default="cpu", help="cpu/cuda/auto")
    parser.add_argument("--conf", type=float, default=0.35, help="Порог confidence")
    parser.add_argument("--show", action="store_true", help="Показывать окно OpenCV")
    parser.add_argument("--log-level", default="INFO", help="Уровень логов")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "main.py"),
        "--source",
        str(args.source),
        "--output",
        str(args.output),
        "--model",
        str(args.model),
        "--device",
        args.device,
        "--conf",
        str(args.conf),
        "--log-level",
        args.log_level,
    ]

    if args.show:
        cmd.append("--show")

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
