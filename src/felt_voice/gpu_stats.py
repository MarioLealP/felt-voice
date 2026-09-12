"""GPU/VRAM headroom via nvidia-smi — felt-voice already runs on the
Nitro5 GPU node as a systemd service, so it's the natural place to expose
this rather than standing up anything new. Parsing is split out from the
subprocess call so it's unit testable without a real GPU.
"""

import subprocess

_QUERY_FIELDS = "name,utilization.gpu,memory.used,memory.total,temperature.gpu"


def parse_nvidia_smi_line(line: str) -> dict:
    """Parse one CSV line from `nvidia-smi --query-gpu=... --format=csv,noheader,nounits`."""
    name, utilization, memory_used, memory_total, temperature = (part.strip() for part in line.split(","))
    return {
        "name": name,
        "utilization_percent": int(utilization),
        "memory_used_mb": int(memory_used),
        "memory_total_mb": int(memory_total),
        "temperature_c": int(temperature),
    }


def get_gpu_stats() -> dict | None:
    """Returns None if nvidia-smi isn't available or fails (e.g. no GPU)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_QUERY_FIELDS}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

    line = result.stdout.strip().splitlines()[0]
    return parse_nvidia_smi_line(line)
