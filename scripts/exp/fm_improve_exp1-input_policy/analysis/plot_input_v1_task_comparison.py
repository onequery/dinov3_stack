#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TASK_SPECS = {
    'global_4_1_patient_retrieval': {
        'target': 'patient',
        'title': 'GA4-1 Patient Retrieval',
        'metrics': [
            ('mAP_mean', 'mAP'),
            ('recall_at_1_mean', 'Recall@1'),
            ('recall_at_5_mean', 'Recall@5'),
        ],
    },
    'global_4_2_same_dicom_retrieval': {
        'target': 'same_dicom',
        'title': 'GA4-2 Same-DICOM Retrieval',
        'metrics': [
            ('mAP_mean', 'mAP'),
            ('recall_at_1_mean', 'Recall@1'),
            ('recall_at_5_mean', 'Recall@5'),
        ],
    },
    'global_4_3_view_classification': {
        'target': 'view_9way',
        'title': 'GA4-3 View Classification',
        'metrics': [
            ('accuracy_mean', 'Accuracy'),
            ('macro_f1_mean', 'Macro-F1'),
            ('balanced_accuracy_mean', 'Balanced Acc'),
        ],
    },
}

POLICY_ORDER = ['baseline_rgbtriplet', 'input_v1_cag_stats_normalization']
POLICY_LABELS = {
    'baseline_rgbtriplet': 'Baseline',
    'input_v1_cag_stats_normalization': 'input_v1',
}
POLICY_COLORS = {
    'baseline_rgbtriplet': '#4a4a4a',
    'input_v1_cag_stats_normalization': '#d96f2b',
}
READOUT_ORDER = ['raw', 'probe']
BACKBONE_ORDER = ['imagenet', 'cag']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot combined baseline vs input_v1 task comparisons.')
    parser.add_argument(
        '--summary-csv',
        default='outputs/fm_improve_exp1-input_policy/input_v1_cag_stats_normalization/downstream_only/reports/summary_input_v1_benchmark_metrics.csv',
    )
    parser.add_argument(
        '--output-dir',
        default='outputs/fm_improve_exp1-input_policy/input_v1_cag_stats_normalization/downstream_only/reports',
    )
    parser.add_argument('--split', default='test')
    return parser.parse_args()


def build_category_order() -> list[tuple[str, str]]:
    return [(backbone, readout) for backbone in BACKBONE_ORDER for readout in READOUT_ORDER]


def label_for_category(backbone: str, readout: str) -> str:
    backbone_label = 'ImageNet' if backbone == 'imagenet' else 'CAG'
    readout_label = 'Raw' if readout == 'raw' else 'Probe'
    return f'{backbone_label}\n{readout_label}'


def plot_task(df: pd.DataFrame, task_name: str, output_dir: Path, split: str) -> list[Path]:
    spec = TASK_SPECS[task_name]
    task_df = df[(df['task'] == task_name) & (df['target'] == spec['target']) & (df['split'] == split)].copy()
    if task_df.empty:
        return []

    categories = build_category_order()
    x = np.arange(len(categories))
    width = 0.34
    created = []

    fig, axes = plt.subplots(1, len(spec['metrics']), figsize=(5.4 * len(spec['metrics']), 5.0), constrained_layout=True)
    if len(spec['metrics']) == 1:
        axes = [axes]

    for ax, (metric_col, metric_label) in zip(axes, spec['metrics']):
        std_col = metric_col.replace('_mean', '_std')
        for idx, policy in enumerate(POLICY_ORDER):
            offset = (-0.5 + idx) * width
            vals = []
            errs = []
            for backbone, readout in categories:
                row = task_df[
                    (task_df['backbone_name'] == backbone)
                    & (task_df['readout_mode'] == readout)
                    & (task_df['input_policy'] == policy)
                ]
                if row.empty:
                    vals.append(np.nan)
                    errs.append(0.0)
                else:
                    vals.append(float(row.iloc[0][metric_col]) if pd.notna(row.iloc[0][metric_col]) else np.nan)
                    errs.append(float(row.iloc[0][std_col]) if std_col in row.columns and pd.notna(row.iloc[0][std_col]) else 0.0)
            ax.bar(
                x + offset,
                vals,
                width=width,
                yerr=errs,
                color=POLICY_COLORS[policy],
                alpha=0.92,
                label=POLICY_LABELS[policy],
                capsize=3,
                edgecolor='white',
                linewidth=0.8,
            )
        ax.set_title(metric_label)
        ax.set_xticks(x)
        ax.set_xticklabels([label_for_category(backbone, readout) for backbone, readout in categories])
        ax.grid(axis='y', alpha=0.25, linestyle='--')
        if any('recall' in m[0] or 'accuracy' in m[0] or 'f1' in m[0] or 'mAP' in m[0] for m in spec['metrics']):
            ax.set_ylim(bottom=0.0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle(f"{spec['title']} | {split} split | Baseline vs input_v1", fontsize=14, y=1.08)

    output_path = output_dir / f'fig_input_v1_{task_name}_combined_{split}.png'
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    created.append(output_path)

    export_rows = []
    for backbone, readout in categories:
        for policy in POLICY_ORDER:
            row = task_df[
                (task_df['backbone_name'] == backbone)
                & (task_df['readout_mode'] == readout)
                & (task_df['input_policy'] == policy)
            ]
            if row.empty:
                continue
            record = {
                'task': task_name,
                'target': spec['target'],
                'split': split,
                'backbone_name': backbone,
                'readout_mode': readout,
                'input_policy': policy,
            }
            for metric_col, _metric_label in spec['metrics']:
                record[metric_col] = row.iloc[0][metric_col]
                std_col = metric_col.replace('_mean', '_std')
                if std_col in row.columns:
                    record[std_col] = row.iloc[0][std_col]
            export_rows.append(record)
    export_df = pd.DataFrame(export_rows)
    export_path = output_dir / f'summary_input_v1_{task_name}_combined_{split}.csv'
    export_df.to_csv(export_path, index=False)
    created.append(export_path)
    return created


def main() -> None:
    args = parse_args()
    summary_csv = Path(args.summary_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(summary_csv)

    created_paths: list[Path] = []
    for task_name in TASK_SPECS:
        created_paths.extend(plot_task(df, task_name, output_dir, args.split))

    print('Created files:')
    for path in created_paths:
        print(path)


if __name__ == '__main__':
    main()
