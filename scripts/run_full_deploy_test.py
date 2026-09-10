#!/usr/bin/env python3
"""Run full deployment validation: unit smoke, ONNX host benchmark, ESP32 serial capture."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPLOY_DIR = ROOT / "outputs" / "export" / "conll2003_full" / "deploy"
SAMPLES = ROOT / "deploy" / "shared" / "sample_inputs.json"
ESP32_DIR = ROOT / "deploy" / "esp32-s3"
RESULTS_DIR = ROOT / "deploy" / "results"
ESP32_MODELS = ESP32_DIR / "models"


def _run(cmd: list[str], cwd: Path | None = None, timeout: int = 600) -> dict:
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd or ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return {
        "command": " ".join(cmd),
        "cwd": str(cwd or ROOT),
        "exit_code": proc.returncode,
        "duration_s": round(time.perf_counter() - t0, 2),
        "stdout": proc.stdout[-8000:] if proc.stdout else "",
        "stderr": proc.stderr[-4000:] if proc.stderr else "",
        "ok": proc.returncode == 0,
    }


def run_pytest(py: Path) -> dict:
    return _run([str(py), "-m", "pytest", "tests/test_smoke.py", "-q"])


def validate_deploy_bundle(deploy_dir: Path) -> dict:
    required = [
        "model_emissions.onnx",
        "char_vocab.json",
        "tags.json",
        "crf_transitions.npy",
        "model_config.json",
        "deploy_manifest.json",
    ]
    files = {}
    ok = True
    for name in required:
        path = deploy_dir / name
        exists = path.exists()
        files[name] = {"exists": exists, "bytes": path.stat().st_size if exists else 0}
        ok = ok and exists
    return {"ok": ok, "deploy_dir": str(deploy_dir), "files": files}


def run_onnx_host_benchmark(py: Path, deploy_dir: Path, samples: Path) -> dict:
    sys.path.insert(0, str(ROOT / "deploy" / "pi3b"))
    import psutil  # noqa: WPS433
    from edgefs_pi.benchmark import run_benchmark

    proc = psutil.Process()
    mem_before = proc.memory_info().rss
    summary = run_benchmark(deploy_dir=deploy_dir, sample_file=samples, warmup=3, repeat=10)
    mem_after = proc.memory_info().rss
    summary["platform"] = "host_cpu_onnx_simulation"
    summary["host_note"] = (
        "ONNX Runtime CPU benchmark on development PC; Pi 3B Plus numbers may differ."
    )
    summary["host_peak_rss_mb"] = round(max(mem_before, mem_after) / (1024 * 1024), 2)
    summary["model_size_kb"] = round(
        (deploy_dir / "model_emissions.onnx").stat().st_size / 1024, 1
    )
    return summary


def validate_espdl_bundle(deploy_dir: Path) -> dict:
    espdl = deploy_dir / "edgefs_model.espdl"
    validation = deploy_dir / "espdl_validation.json"
    ok = espdl.exists() and espdl.stat().st_size > 0
    result = {
        "ok": ok,
        "espdl_path": str(espdl),
        "espdl_kb": round(espdl.stat().st_size / 1024, 1) if ok else 0,
    }
    if validation.exists():
        result["validation"] = json.loads(validation.read_text(encoding="utf-8"))
        result["ok"] = result["ok"] and result["validation"].get("onnx_ok", False)
    return result


def run_esp32_test(port: str, skip_upload: bool, env: str = "esp32-s3-n16r8-stub") -> dict:
    steps: dict[str, dict] = {}

    if not skip_upload:
        steps["build_upload"] = _run(
            ["pio", "run", "-e", env, "-t", "upload", "--upload-port", port],
            cwd=ESP32_DIR,
            timeout=900,
        )
        if not steps["build_upload"]["ok"]:
            return {"ok": False, "port": port, "steps": steps}

    try:
        import serial
    except ImportError as exc:
        return {"ok": False, "error": f"pyserial missing: {exc}", "steps": steps}

    lines: list[str] = []
    try:
        ser = serial.Serial(port, 115200, timeout=0.5)
        ser.setDTR(False)
        ser.setRTS(True)
        time.sleep(0.1)
        ser.setRTS(False)
        time.sleep(0.3)
        deadline = time.time() + 12
        while time.time() < deadline:
            raw = ser.readline()
            if raw:
                lines.append(raw.decode("utf-8", errors="replace").rstrip())
        ser.close()
    except Exception as exc:
        return {"ok": False, "error": str(exc), "steps": steps, "serial_lines": lines}

    pattern = re.compile(
        r"sample=(?P<id>\S+) len=(?P<len>\d+) latency_us=(?P<lat>\d+) "
        r"tags\[0\]=(?P<tag>-?\d+) heap=(?P<heap>\d+) psram=(?P<psram>\d+)"
    )
    samples = []
    for line in lines:
        m = pattern.search(line)
        if m:
            samples.append(
                {
                    "id": m.group("id"),
                    "len": int(m.group("len")),
                    "latency_us": int(m.group("lat")),
                    "tag0": int(m.group("tag")),
                    "free_heap_bytes": int(m.group("heap")),
                    "free_psram_bytes": int(m.group("psram")),
                }
            )

    boot = {
        "flash_size_mb": next(
            (ln.split(":")[-1].strip() for ln in lines if "SPI Flash Size" in ln),
            None,
        ),
        "board_name": next(
            (ln.split("start (")[-1].rstrip(")") for ln in lines if "benchmark start" in ln),
            None,
        ),
    }

    ok = len(samples) >= 3 and any("benchmark complete" in ln for ln in lines)
    return {
        "ok": ok,
        "port": port,
        "steps": steps,
        "boot": boot,
        "samples": samples,
        "serial_lines": lines[-40:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Full deployment test with recorded report")
    parser.add_argument("--deploy-dir", default=str(DEPLOY_DIR))
    parser.add_argument("--esp32-port", default="COM3")
    parser.add_argument(
        "--esp32-env",
        default="esp32-s3-n16r8-stub",
        choices=("esp32-s3-n16r8-stub", "esp32-s3-n16r8-espdl"),
    )
    parser.add_argument("--skip-esp32-upload", action="store_true")
    parser.add_argument("--skip-espdl-quantize", action="store_true")
    parser.add_argument(
        "--output",
        default=str(RESULTS_DIR / "full_deploy_test.json"),
    )
    args = parser.parse_args()

    py = ROOT / ".venv" / "Scripts" / "python.exe"
    if not py.exists():
        py = Path(sys.executable)

    report: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": "windows_dev_pc",
        "sections": {},
    }

    print("== pytest smoke ==")
    report["sections"]["pytest_smoke"] = run_pytest(py)

    print("== deploy bundle ==")
    report["sections"]["deploy_bundle"] = validate_deploy_bundle(Path(args.deploy_dir))

    print("== ESP-DL export ==")
    espdl_section: dict = {"ok": False}
    try:
        if not args.skip_espdl_quantize:
            espdl_section["quantize"] = _run(
                [str(py), "scripts/quantize_to_espdl.py", "--backend", "torch"],
                cwd=ROOT,
                timeout=900,
            )
        _run([str(py), "scripts/gen_esp32_assets.py"], cwd=ROOT, timeout=120)
        espdl_section["validate"] = _run(
            [str(py), "scripts/validate_espdl_export.py"],
            cwd=ROOT,
            timeout=300,
        )
        espdl_section["bundle"] = validate_espdl_bundle(ESP32_MODELS)
        espdl_section["ok"] = (
            espdl_section.get("validate", {}).get("ok", False)
            and espdl_section["bundle"].get("ok", False)
            and (args.skip_espdl_quantize or espdl_section.get("quantize", {}).get("ok", False))
        )
    except Exception as exc:
        espdl_section = {"ok": False, "error": str(exc)}
    report["sections"]["espdl_export"] = espdl_section

    print("== ONNX host benchmark ==")
    try:
        report["sections"]["onnx_host"] = {
            "ok": True,
            "summary": run_onnx_host_benchmark(py, Path(args.deploy_dir), SAMPLES),
        }
    except Exception as exc:
        report["sections"]["onnx_host"] = {"ok": False, "error": str(exc)}

    print(f"== ESP32-S3 serial benchmark ({args.esp32_env}) ==")
    report["sections"]["esp32_s3"] = run_esp32_test(
        args.esp32_port,
        skip_upload=args.skip_esp32_upload,
        env=args.esp32_env,
    )

    section_ok = []
    for name, sec in report["sections"].items():
        if name == "onnx_host":
            section_ok.append(sec.get("ok", False))
        else:
            section_ok.append(sec.get("ok", False))
    report["overall_ok"] = all(section_ok)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport written: {out}")
    print(f"Overall: {'PASS' if report['overall_ok'] else 'FAIL'}")
    raise SystemExit(0 if report["overall_ok"] else 1)


if __name__ == "__main__":
    main()
