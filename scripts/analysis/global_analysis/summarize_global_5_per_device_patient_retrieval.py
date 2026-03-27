from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DeviceSpec:
    device_key: str
    device_name: str
    analysis_title: str
    subset_root: Path
    analysis_root: Path
    summary_csv: str
    patient_markdown: str
    anchoring_markdown: str
    build_wrapper: Path
    run_wrapper: Path


def project_root_from_here() -> Path:
    return Path(__file__).resolve().parents[3]


ROOT = project_root_from_here()

DEFAULT_SPECS = [
    DeviceSpec(
        device_key="philips_integris_h",
        device_name="P H I L I P S INTEGRIS H",
        analysis_title="Global Analysis 5",
        subset_root=ROOT / "input/global_analysis_5_per_device_patient_retrieval/philips_integris_h_unique_view_subset",
        analysis_root=ROOT / "outputs/analysis2_rep_analysis/global_5_per_device_patient_retrieval/philips_integris_h",
        summary_csv="summary_global_5_patient_retrieval.csv",
        patient_markdown="analysis_global_5_patient_retrieval.md",
        anchoring_markdown="analysis_global_5_cluster_anchoring_attribution.md",
        build_wrapper=ROOT / "scripts/analysis/global_analysis/run_global_5_build_philips_integris_h_unique_view_subset.sh",
        run_wrapper=ROOT / "scripts/analysis/global_analysis/run_global_5_single_device_patient_retrieval.sh",
    ),
    DeviceSpec(
        device_key="alluraxper",
        device_name="AlluraXper",
        analysis_title="Global Analysis 5-1",
        subset_root=ROOT / "input/global_analysis_5_per_device_patient_retrieval/alluraxper_unique_view_subset",
        analysis_root=ROOT / "outputs/analysis2_rep_analysis/global_5_per_device_patient_retrieval/alluraxper",
        summary_csv="summary_global_5_1_patient_retrieval.csv",
        patient_markdown="analysis_global_5_1_patient_retrieval.md",
        anchoring_markdown="analysis_global_5_1_cluster_anchoring_attribution.md",
        build_wrapper=ROOT / "scripts/analysis/global_analysis/run_global_5_1_build_alluraxper_unique_view_subset.sh",
        run_wrapper=ROOT / "scripts/analysis/global_analysis/run_global_5_1_single_device_patient_retrieval.sh",
    ),
    DeviceSpec(
        device_key="integris_allura_flat_detector",
        device_name="INTEGRIS Allura Flat Detector",
        analysis_title="Global Analysis 5-2",
        subset_root=ROOT / "input/global_analysis_5_per_device_patient_retrieval/integris_allura_flat_detector_unique_view_subset",
        analysis_root=ROOT / "outputs/analysis2_rep_analysis/global_5_per_device_patient_retrieval/integris_allura_flat_detector",
        summary_csv="summary_global_5_2_patient_retrieval.csv",
        patient_markdown="analysis_global_5_2_patient_retrieval.md",
        anchoring_markdown="analysis_global_5_2_cluster_anchoring_attribution.md",
        build_wrapper=ROOT / "scripts/analysis/global_analysis/run_global_5_2_build_integris_allura_flat_detector_unique_view_subset.sh",
        run_wrapper=ROOT / "scripts/analysis/global_analysis/run_global_5_2_single_device_patient_retrieval.sh",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize GA5 per-device patient retrieval family.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/analysis2_rep_analysis/global_5_per_device_patient_retrieval",
    )
    parser.add_argument(
        "--input-frame-root",
        type=Path,
        default=ROOT / "input/global_analysis_5_per_device_patient_retrieval",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_rows(csv_path: Path, spec: DeviceSpec) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            enriched = dict(row)
            enriched["device_key"] = spec.device_key
            enriched["device_name"] = spec.device_name
            enriched["analysis_title"] = spec.analysis_title
            enriched["subset_root"] = str(spec.subset_root)
            enriched["analysis_root"] = str(spec.analysis_root)
            rows.append(enriched)
        return rows


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: Iterable[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def float_str(row: dict[str, str], key: str) -> float:
    return float(row[key])


def build_markdown_lookup(specs: list[DeviceSpec]) -> dict[str, str]:
    return {spec.device_key: str(spec.analysis_root / spec.patient_markdown) for spec in specs}


def build_test_probe_table(
    rows: list[dict[str, str]],
    patient_markdown_lookup: dict[str, str],
) -> list[dict[str, str]]:
    out = []
    for row in rows:
        if row["target"] != "patient" or row["split"] != "test" or row["mode"] != "probe_linear":
            continue
        out.append({
            "device_key": row["device_key"],
            "device_name": row["device_name"],
            "analysis_title": row["analysis_title"],
            "backbone_name": row["backbone_name"],
            "mAP_mean": row["mAP_mean"],
            "mAP_std": row["mAP_std"],
            "recall_at_1_mean": row["recall_at_1_mean"],
            "recall_at_5_mean": row["recall_at_5_mean"],
            "queries_with_no_positive_mean": row["queries_with_no_positive_mean"],
            "patient_markdown": patient_markdown_lookup[row["device_key"]],
        })
    out.sort(key=lambda item: (item["device_key"], item["backbone_name"]))
    return out


def build_test_all_modes(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for row in rows:
        if row["target"] != "patient" or row["split"] != "test":
            continue
        out.append({
            "device_key": row["device_key"],
            "device_name": row["device_name"],
            "analysis_title": row["analysis_title"],
            "mode": row["mode"],
            "backbone_name": row["backbone_name"],
            "mAP_mean": row["mAP_mean"],
            "mAP_std": row["mAP_std"],
            "recall_at_1_mean": row["recall_at_1_mean"],
            "recall_at_5_mean": row["recall_at_5_mean"],
            "queries_with_no_positive_mean": row["queries_with_no_positive_mean"],
        })
    out.sort(key=lambda item: (item["device_key"], item["mode"], item["backbone_name"]))
    return out


def render_markdown(specs: list[DeviceSpec], test_all_modes: list[dict[str, str]], output_root: Path, input_frame_root: Path) -> str:
    lookup = {(r["device_key"], r["mode"], r["backbone_name"]): r for r in test_all_modes}
    lines: list[str] = []
    lines.append("# Global Analysis 5: Per-device Patient Retrieval")
    lines.append("")
    lines.append("## Frame")
    lines.append("")
    lines.append("- Unified framing for the following per-device patient retrieval analyses:")
    lines.append("  - `Global Analysis 5` -> `P H I L I P S INTEGRIS H`")
    lines.append("  - `Global Analysis 5-1` -> `AlluraXper`")
    lines.append("  - `Global Analysis 5-2` -> `INTEGRIS Allura Flat Detector`")
    lines.append("- Shared task: patient retrieval on `input/Stent-Contrast-unique-view` subsets, using the same retrieval/export/anchoring pipeline with only the device subset changed.")
    lines.append("")
    lines.append("## Experiment Roots")
    lines.append("")
    for spec in specs:
        lines.append(f"- `{spec.device_name}`")
        lines.append(f"  - subset: `{spec.subset_root}`")
        lines.append(f"  - analysis root: `{spec.analysis_root}`")
        lines.append(f"  - retrieval benchmark: `{spec.analysis_root / 'retrieval_benchmark'}`")
    lines.append("")
    lines.append("## Related Scripts")
    lines.append("")
    lines.append(f"- generic subset builder: `scripts/analysis/global_analysis/build_global_5_single_device_unique_view_subset.py`")
    lines.append(f"- generic experiment runner: `scripts/analysis/global_analysis/run_global_5_single_device_patient_retrieval.py`")
    for spec in specs:
        lines.append(f"- `{spec.device_name}` subset wrapper: `{spec.build_wrapper}`")
        lines.append(f"- `{spec.device_name}` experiment wrapper: `{spec.run_wrapper}`")
    lines.append(f"- per-device frame summary runner: `scripts/analysis/global_analysis/run_global_5_per_device_patient_retrieval_summary.sh`")
    lines.append("")
    lines.append("## Test Patient Retrieval Summary")
    lines.append("")
    lines.append("| Device | Mode | Backbone | mAP | R@1 | R@5 | Queries With No Positive |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    for spec in specs:
        for mode in ("probe_linear", "raw_frozen"):
            for backbone in ("imagenet", "cag"):
                row = lookup[(spec.device_key, mode, backbone)]
                lines.append(
                    f"| {spec.device_name} | {mode} | {backbone} | {float_str(row, 'mAP_mean'):.6f} | {float_str(row, 'recall_at_1_mean'):.6f} | {float_str(row, 'recall_at_5_mean'):.6f} | {float_str(row, 'queries_with_no_positive_mean'):.6f} |"
                )
    lines.append("")
    lines.append("## Structure")
    lines.append("")
    lines.append(f"- integrated output frame root: `{output_root}`")
    lines.append(f"- integrated input subset frame root: `{input_frame_root}`")
    lines.append("- Device-specific experiment results now live directly under the integrated output frame root.")
    lines.append("- Device-specific subset directories now live directly under the integrated input subset frame root.")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_root)
    ensure_dir(args.input_frame_root)

    all_rows: list[dict[str, str]] = []
    for spec in DEFAULT_SPECS:
        summary_path = spec.analysis_root / spec.summary_csv
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary CSV: {summary_path}")
        if not spec.subset_root.exists():
            raise FileNotFoundError(f"Missing subset root: {spec.subset_root}")
        all_rows.extend(load_rows(summary_path, spec))

    all_fieldnames = list(all_rows[0].keys())
    write_csv(args.output_root / "summary_global_5_per_device_patient_retrieval.csv", all_rows, all_fieldnames)

    test_probe_rows = build_test_probe_table(all_rows, build_markdown_lookup(DEFAULT_SPECS))
    write_csv(
        args.output_root / "summary_global_5_per_device_patient_retrieval_test_probe.csv",
        test_probe_rows,
        [
            "device_key",
            "device_name",
            "analysis_title",
            "backbone_name",
            "mAP_mean",
            "mAP_std",
            "recall_at_1_mean",
            "recall_at_5_mean",
            "queries_with_no_positive_mean",
            "patient_markdown",
        ],
    )

    test_all_modes = build_test_all_modes(all_rows)
    write_csv(
        args.output_root / "summary_global_5_per_device_patient_retrieval_test_all_modes.csv",
        test_all_modes,
        [
            "device_key",
            "device_name",
            "analysis_title",
            "mode",
            "backbone_name",
            "mAP_mean",
            "mAP_std",
            "recall_at_1_mean",
            "recall_at_5_mean",
            "queries_with_no_positive_mean",
        ],
    )

    markdown = render_markdown(DEFAULT_SPECS, test_all_modes, args.output_root, args.input_frame_root)
    (args.output_root / "analysis_global_5_per_device_patient_retrieval.md").write_text(markdown, encoding="utf-8")

    frame_meta = {
        "analysis_title": "Global Analysis 5: Per-device Patient Retrieval",
        "output_root": str(args.output_root),
        "input_frame_root": str(args.input_frame_root),
        "devices": [
            {
                "device_key": spec.device_key,
                "device_name": spec.device_name,
                "analysis_title": spec.analysis_title,
                "subset_root": str(spec.subset_root),
                "analysis_root": str(spec.analysis_root),
                "build_wrapper": str(spec.build_wrapper),
                "run_wrapper": str(spec.run_wrapper),
            }
            for spec in DEFAULT_SPECS
        ],
    }
    (args.output_root / "run_meta.json").write_text(json.dumps(frame_meta, indent=2), encoding="utf-8")

    subset_lines = [
        "# Global Analysis 5: Per-device Patient Retrieval Subsets",
        "",
        f"- frame root: `{args.input_frame_root}`",
        "- Each child path below is a real device-specific subset directory under the shared frame root.",
        "",
    ]
    for spec in DEFAULT_SPECS:
        subset_lines.append(f"- `{spec.device_name}`: `{args.input_frame_root / (spec.device_key + '_unique_view_subset')}`")
    (args.input_frame_root / "analysis_global_5_per_device_patient_retrieval_subsets.md").write_text("\n".join(subset_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
