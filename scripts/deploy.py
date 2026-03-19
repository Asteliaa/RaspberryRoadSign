#!/usr/bin/env python3
"""Подготовка окружения для Raspberry"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VENV = PROJECT_ROOT / ".venv"
DEFAULT_REQUIREMENTS = PROJECT_ROOT / "requirements_raspberry.txt"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Setup RaspberryRoadSign Python environment")
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV, help="Путь к virtualenv")
    parser.add_argument(
        "--requirements",
        type=Path,
        default=DEFAULT_REQUIREMENTS,
        help="Путь к requirements файлу",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    venv = args.venv.resolve()
    requirements = args.requirements.resolve()

    if not requirements.exists():
        raise FileNotFoundError(f"requirements не найден: {requirements}")

    print(f"[INFO] Создаю/переиспользую venv: {venv}")
    run([sys.executable, "-m", "venv", str(venv)])

    pip_exe = venv / "bin" / "pip"
    python_exe = venv / "bin" / "python"
    if not pip_exe.exists() or not python_exe.exists():
        raise RuntimeError(f"Не удалось найти python/pip в окружении: {venv}")

    print("[INFO] Обновляю pip")
    run([str(pip_exe), "install", "--upgrade", "pip"])

    print(f"[INFO] Устанавливаю зависимости: {requirements}")
    run([str(pip_exe), "install", "-r", str(requirements)])

    print("[OK] Deploy complete")
    print("[NEXT] Запуск инференса:")
    print(
        f"  {python_exe} {PROJECT_ROOT / 'scripts' / 'run_raspberry.py'} "
        "--source 0 --output output/raspberry_output.mp4 --model models/deploy/best_ncnn_model --device cpu"
    )


if __name__ == "__main__":
    main()
