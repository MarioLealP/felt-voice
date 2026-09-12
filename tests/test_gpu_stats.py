from felt_voice.gpu_stats import parse_nvidia_smi_line


def test_parse_nvidia_smi_line():
    line = "NVIDIA GeForce RTX 3060 Laptop GPU, 12, 1261, 6144, 46"
    assert parse_nvidia_smi_line(line) == {
        "name": "NVIDIA GeForce RTX 3060 Laptop GPU",
        "utilization_percent": 12,
        "memory_used_mb": 1261,
        "memory_total_mb": 6144,
        "temperature_c": 46,
    }


def test_parse_nvidia_smi_line_idle():
    line = "NVIDIA GeForce RTX 3060 Laptop GPU, 0, 0, 6144, 38"
    result = parse_nvidia_smi_line(line)
    assert result["utilization_percent"] == 0
    assert result["memory_used_mb"] == 0
