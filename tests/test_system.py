from researchlab.system import parse_npu_smi, parse_nvidia_smi


def test_nvidia_status_includes_process_owner() -> None:
    gpus = "0, GPU-a, NVIDIA A100, 40960, 1024, 20, 51\n"
    processes = "GPU-a, 1234, python, 900\n"

    devices = parse_nvidia_smi(gpus, processes, {1234: "alice"})

    assert devices[0]["busy"] is True
    assert devices[0]["processes"][0]["user"] == "alice"


def test_ascend_status_parser_keeps_unknown_fields_safe() -> None:
    output = """
+----------------------------------------------------------------------------+
| 0 910B | OK | 23% | 1024 / 32768 |
| NPU Chip | Process id | Process name |
| 0 0 | 12345 | python |
+----------------------------------------------------------------------------+
"""
    devices = parse_npu_smi(output)

    assert devices[0]["backend"] == "npu"
    assert devices[0]["memory_total_mb"] == 32768
    assert devices[0]["busy"] is True

