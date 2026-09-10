#!/usr/bin/env python3
"""Run tiered rigor checks for paper submission (P0 must / P1 should / P2 optional)."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "deploy" / "results"
sys.path.insert(0, str(ROOT / "scripts"))


@dataclass
class Check:
    id: str
    tier: str  # P0 | P1 | P2
    name: str
    passed: bool
    detail: str
    blocking: bool = True


@dataclass
class RigorReport:
    timestamp_utc: str
    checks: list[Check] = field(default_factory=list)

    @property
    def p0_pass(self) -> int:
        return sum(1 for c in self.checks if c.tier == "P0" and c.passed)

    @property
    def p0_total(self) -> int:
        return sum(1 for c in self.checks if c.tier == "P0")

    @property
    def p1_pass(self) -> int:
        return sum(1 for c in self.checks if c.tier == "P1" and c.passed)

    @property
    def p1_total(self) -> int:
        return sum(1 for c in self.checks if c.tier == "P1")

    def ok_for_submission(self) -> bool:
        return all(c.passed for c in self.checks if c.tier == "P0" and c.blocking)


def _load_json(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_experiment_steps(report: RigorReport, phase: str) -> None:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/verify_experiment_steps.py"), "--phase", phase],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    m = re.search(r"Summary: (\d+)/(\d+) passed", proc.stdout)
    ok, total = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    tier = "P0" if phase in ("all", "2") else "P1"
    report.checks.append(
        Check(
            id=f"EXP-{phase}",
            tier=tier,
            name=f"Experiment artifacts phase {phase}",
            passed=ok == total,
            detail=f"{ok}/{total} passed",
            blocking=(phase == "all"),
        )
    )


def check_deploy_f1_parity(report: RigorReport) -> None:
    host = _load_json(RESULTS / "host_deploy_eval.json")
    pi = _load_json(RESULTS / "pi3b_full_test.json")
    hf1 = (host or {}).get("metrics", {}).get("f1")
    pf1 = ((pi or {}).get("summary") or {}).get("metrics", {}).get("f1")
    if hf1 is None or pf1 is None:
        report.checks.append(
            Check("DEP-01", "P0", "Host vs Pi deploy F1 parity", False, "missing eval JSON")
        )
        return
    match = abs(hf1 - pf1) < 1e-4
    report.checks.append(
        Check(
            "DEP-01",
            "P0",
            "Host vs Pi deploy F1 parity",
            match,
            f"host={hf1:.4f} pi={pf1:.4f}",
        )
    )


def check_artifact_manifest(report: RigorReport) -> None:
    manifest_path = ROOT / "deploy/esp32-s3/models/artifact_manifest.json"
    manifest = _load_json(manifest_path)
    if not manifest:
        report.checks.append(
            Check("ESP-01", "P0", "ESP32 artifact manifest", False, "missing manifest")
        )
        return
    espdl = ROOT / "deploy/esp32-s3/models/edgefs_model.espdl"
    expected = manifest.get("primary_espdl_sha256")
    actual = _sha256(espdl) if espdl.exists() else None
    ok = expected == actual
    report.checks.append(
        Check(
            "ESP-01",
            "P0",
            "ESP32 artifact manifest SHA256",
            ok,
            f"tag={manifest.get('artifact_tag')} match={ok}",
        )
    )
    build_h = ROOT / "deploy/esp32-s3/include/edge_build_info.h"
    report.checks.append(
        Check(
            "ESP-02",
            "P1",
            "edge_build_info.h present",
            build_h.exists(),
            str(build_h.relative_to(ROOT)),
            blocking=False,
        )
    )


def check_espdl_validation(report: RigorReport) -> None:
    val = _load_json(ROOT / "deploy/esp32-s3/models/espdl_validation.json")
    if not val:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts/validate_espdl_export.py")],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        val = _load_json(ROOT / "deploy/esp32-s3/models/espdl_validation.json")
        if not val:
            report.checks.append(
                Check("ESP-03", "P1", "ESP-DL export validation", False, proc.stderr[-200:])
            )
            return
    ok = bool(val.get("ok"))
    report.checks.append(
        Check(
            "ESP-03",
            "P1",
            "ESP-DL export validation",
            ok,
            f"post_embed_ok={val.get('post_embed_ok')} onnx_ok={val.get('onnx_ok')}",
            blocking=False,
        )
    )


def check_esp32_full_eval(report: RigorReport) -> None:
    esp = _load_json(RESULTS / "esp32_full_eval.json")
    if not esp:
        report.checks.append(
            Check("ESP-04", "P0", "ESP32 full test eval", False, "missing esp32_full_eval.json")
        )
        return
    n = esp.get("num_sentences", 0)
    f1 = (esp.get("metrics") or {}).get("f1")
    ok = n >= 1722 and f1 is not None and f1 > 0
    tag = esp.get("artifact_tag")
    report.checks.append(
        Check(
            "ESP-04",
            "P0",
            "ESP32 full CoNLL test eval",
            ok,
            f"n={n} f1={f1:.4f}" if f1 else f"n={n}",
        )
    )
    report.checks.append(
        Check(
            "ESP-05",
            "P1",
            "ESP32 eval records artifact_tag",
            bool(tag),
            f"tag={tag or 'missing'}",
            blocking=False,
        )
    )


def check_deploy_gap_diagnosis(report: RigorReport, run_diagnose: bool) -> None:
    path = RESULTS / "deploy_gap_diagnosis.json"
    if run_diagnose or not path.exists():
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/diagnose_deploy_gap.py")],
            cwd=str(ROOT),
            capture_output=True,
        )
    diag = _load_json(path)
    if not diag:
        report.checks.append(
            Check("DEP-02", "P0", "Deploy gap diagnosis", False, "diagnosis failed")
        )
        return
    onnx_ok = "matches PyTorch" in (diag.get("verdicts") or {}).get("onnx_export", "")
    deploy = diag.get("deploy_roundtrip") or {}
    identical = deploy.get("predictions_identical", False)
    max_diff = deploy.get("max_emission_abs_diff", 999)
    report.checks.append(
        Check(
            "DEP-02",
            "P0",
            "ONNX export parity (PyTorch vs ORT)",
            onnx_ok and identical and max_diff < 1e-3,
            f"identical={identical} max_emission_diff={max_diff}",
        )
    )
    gaps = diag.get("gaps") or {}
    test_gap = abs(gaps.get("metrics_json_vs_local_checkpoint", 0))
    report.checks.append(
        Check(
            "DEP-03",
            "P0",
            "Training metrics vs local test.conll alignment",
            test_gap < 0.02,
            f"gap={test_gap:.4f} (>{0.02} => different test split or need remote pull)",
        )
    )
    report.checks.append(
        Check(
            "DEP-04",
            "P1",
            "Deploy gap report on disk",
            True,
            str(path.relative_to(ROOT)),
            blocking=False,
        )
    )


def check_paper_table3(report: RigorReport) -> None:
    t3 = (ROOT / "paper/tables/table3-deployment.tex").read_text(encoding="utf-8")
    summary = _load_json(RESULTS / "deploy_all_summary.json")
    if not summary:
        report.checks.append(
            Check("PAP-01", "P0", "Table 3 vs deploy summary", False, "no deploy_all_summary.json")
        )
        return
    pi_lat = summary.get("pi3b_full", {}).get("latency_ms_mean")
    esp_lat = summary.get("esp32_s3_int8", {}).get("latency_ms_mean")
    issues = []
    if pi_lat and "54.9" in t3 and abs(pi_lat - 54.9) > 0.5:
        issues.append(f"Pi latency table=54.9 measured={pi_lat:.1f}")
    if esp_lat and "1{,}205" not in t3 and "1205" not in t3:
        issues.append("ESP32 latency not 1205ms in table")
    ok = not issues
    report.checks.append(
        Check(
            "PAP-01",
            "P1",
            "Table 3 numbers vs measured summary",
            ok,
            "; ".join(issues) if issues else "consistent within tolerance",
            blocking=False,
        )
    )


def check_fewshot_seeds(report: RigorReport) -> None:
    for name, rel in [
        ("CoNLL 5w1s", "outputs/runs/conll2003_fewshot_5w1s/fewshot_summary.json"),
        ("CoNLL 5w5s", "outputs/runs/conll2003_fewshot_5w5s/fewshot_summary.json"),
    ]:
        data = _load_json(ROOT / rel)
        n = len(data) if isinstance(data, list) else 0
        report.checks.append(
            Check(
                f"FS-{name[:6]}",
                "P0",
                f"Few-shot {name} >=5 seeds",
                n >= 5,
                f"n_seeds={n}",
            )
        )


def check_fewshot_baselines(report: RigorReport) -> None:
    baselines = [
        "conll2003_fewshot_protobert_4w1s",
        "conll2003_fewshot_nnshot_4w1s",
        "conll2003_fewshot_structshot_4w1s",
    ]
    missing = []
    for run in baselines:
        if not (ROOT / "outputs/runs" / run / "fewshot_summary.json").exists():
            missing.append(run)
    report.checks.append(
        Check(
            "FS-BASE",
            "P0",
            "Few-shot baselines (ProtoBERT/NNShot/StructShot)",
            not missing,
            f"missing: {missing}" if missing else "all present",
        )
    )


def check_cluener_not_zero(report: RigorReport) -> None:
    path = ROOT / "outputs/runs/cluener_full/metrics.json"
    data = _load_json(path)
    f1 = float((data or {}).get("test", {}).get("f1", 0))
    report.checks.append(
        Check(
            "CLU-01",
            "P0",
            "CLUENER test F1 > 0 (IOB fix)",
            f1 > 0.05,
            f"test_f1={f1:.4f}",
        )
    )


def check_table1_no_placeholder(report: RigorReport) -> None:
    t1 = (ROOT / "paper/tables/table1-main-results.tex").read_text(encoding="utf-8")
    has_ph = "\\placeholder{XX}" in t1
    report.checks.append(
        Check(
            "PAP-02",
            "P0",
            "Table 1 no \\placeholder{XX}",
            not has_ph,
            "CLUENER column still placeholder" if has_ph else "ok",
        )
    )


def check_iob_schema(report: RigorReport) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from edgefs.data.schema import normalize_ner_tag

    cases = [
        ("B-company", "B-company"),
        ("I-PERSON", "I-PERSON"),
        ("B-LOC", "B-LOC"),
        ("O", "O"),
        ("X", "O"),
    ]
    failed = [f"{inp}->{normalize_ner_tag(inp)}" for inp, exp in cases if normalize_ner_tag(inp) != exp]
    report.checks.append(
        Check(
            "DATA-01",
            "P1",
            "IOB tag normalization preserves entity labels",
            not failed,
            "; ".join(failed) if failed else "CLUENER/OntoNotes B-/I- tags preserved",
            blocking=False,
        )
    )


def check_deploy_encode_regression(report: RigorReport) -> None:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "deploy/pi3b/tests/test_onnx_runner_encode.py")],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    report.checks.append(
        Check(
            "DEP-06",
            "P0",
            "Deploy char encoding matches training dataset",
            proc.returncode == 0,
            (proc.stdout or proc.stderr).strip()[-120:],
        )
    )


def check_summarize_deploy(report: RigorReport) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/summarize_deploy_results.py")],
        cwd=str(ROOT),
        capture_output=True,
    )
    ok = (RESULTS / "deploy_all_summary.json").exists()
    report.checks.append(
        Check("DEP-05", "P1", "Regenerate deploy_all_summary.json", ok, str(RESULTS / "deploy_all_summary.json"), blocking=False)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper rigor validation suite")
    parser.add_argument("--skip-diagnose", action="store_true", help="Reuse existing deploy_gap_diagnosis.json")
    parser.add_argument(
        "--output",
        default=str(RESULTS / "rigor_report.json"),
    )
    parser.add_argument("--tier", choices=("all", "P0", "P1"), default="all")
    args = parser.parse_args()

    report = RigorReport(timestamp_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat())

    check_experiment_steps(report, "1")
    check_experiment_steps(report, "2")
    check_deploy_f1_parity(report)
    check_artifact_manifest(report)
    check_esp32_full_eval(report)
    check_deploy_gap_diagnosis(report, run_diagnose=not args.skip_diagnose)
    check_fewshot_seeds(report)
    check_fewshot_baselines(report)
    check_cluener_not_zero(report)
    check_table1_no_placeholder(report)
    check_paper_table3(report)
    check_espdl_validation(report)
    check_iob_schema(report)
    check_deploy_encode_regression(report)
    check_summarize_deploy(report)

    if args.tier != "all":
        report.checks = [c for c in report.checks if c.tier == args.tier]

    out = Path(args.output)
    payload = {
        "timestamp_utc": report.timestamp_utc,
        "p0_pass": report.p0_pass,
        "p0_total": report.p0_total,
        "p1_pass": report.p1_pass,
        "p1_total": report.p1_total,
        "submission_ready": report.ok_for_submission(),
        "checks": [asdict(c) for c in report.checks],
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"=== Rigor Report ===")
    print(f"P0: {report.p0_pass}/{report.p0_total}  P1: {report.p1_pass}/{report.p1_total}")
    print(f"Submission ready (P0 blocking): {report.ok_for_submission()}")
    for c in report.checks:
        mark = "PASS" if c.passed else "FAIL"
        print(f"[{mark}] [{c.tier}] {c.id} {c.name}: {c.detail}")
    print(f"Wrote {out}")
    return 0 if report.ok_for_submission() else 1


if __name__ == "__main__":
    raise SystemExit(main())
