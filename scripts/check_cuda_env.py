#!/usr/bin/env python
from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys


def run_command(command: list[str]) -> dict[str, object]:
    executable = shutil.which(command[0])
    if executable is None:
        return {
            "available": False,
            "executable": None,
            "returncode": None,
            "stdout": "",
            "stderr": f"{command[0]} not found",
        }

    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "available": True,
        "executable": executable,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def main() -> None:
    pip_show_torch = run_command([sys.executable, "-m", "pip", "show", "torch"])
    nvidia_smi = run_command(["nvidia-smi"])
    nvcc_version = run_command(["nvcc", "--version"])

    info: dict[str, object] = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "which_python": shutil.which("python"),
        "which_pip": shutil.which("pip"),
        "pip_show_torch": pip_show_torch,
        "nvidia_smi": nvidia_smi,
        "nvcc_version": nvcc_version,
    }

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count()
        info.update(
            {
                "torch_import_error": None,
                "torch_version": torch.__version__,
                "torch_version_cuda": torch.version.cuda,
                "torch_cuda_is_available": cuda_available,
                "torch_cuda_device_count": device_count,
                "torch_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
                "torch_device_capability": torch.cuda.get_device_capability(0) if cuda_available else None,
                "torch_build_appears_cpu_only": torch.version.cuda is None or "+cpu" in torch.__version__,
            }
        )
    except Exception as exc:
        info.update(
            {
                "torch_import_error": f"{type(exc).__name__}: {exc}",
                "torch_version": None,
                "torch_version_cuda": None,
                "torch_cuda_is_available": False,
                "torch_cuda_device_count": 0,
                "torch_device_name": None,
                "torch_device_capability": None,
                "torch_build_appears_cpu_only": None,
            }
        )

    print(json.dumps(info, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
