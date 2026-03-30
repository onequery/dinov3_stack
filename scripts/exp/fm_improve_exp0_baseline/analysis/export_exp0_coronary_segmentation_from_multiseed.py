#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BACKBONE_ORDER = {'imagenet': 0, 'cag': 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Populate Exp0 coronary segmentation output from multiseed last-layer source.')
    parser.add_argument('--source-root', default='outputs/analysis2_rep-analysis/local_2_1_layerwise_segmentation_linear_probe_multiseed')
    parser.add_argument('--output-root', default='outputs/fm-imp-exp0_baseline/2_dense/1_coronary_segmentation')
    parser.add_argument('--selected-layer-id', type=int, default=12)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def maybe_clear(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)


def sort_backbones(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(_backbone_order=df['backbone_name'].map(BACKBONE_ORDER).fillna(999)).sort_values(
        ['_backbone_order', 'split', 'backbone_name']
    ).drop(columns=['_backbone_order']).reset_index(drop=True)


def save_bar_compare(summary_df: pd.DataFrame, output_path: Path) -> None:
    test_df = sort_backbones(summary_df.loc[summary_df['split'] == 'test'].copy())
    metrics = [
        ('miou', 'mIoU', 'miou_std'),
        ('dice', 'Dice', 'dice_std'),
    ]
    colors = {'imagenet': '#4C78A8', 'cag': '#F58518'}
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), constrained_layout=True)
    x = np.arange(len(test_df))
    labels = [row['backbone_name'] for _, row in test_df.iterrows()]
    for ax, (metric, title, std_col) in zip(axes, metrics):
        vals = test_df[metric].to_numpy(dtype=float)
        errs = test_df[std_col].to_numpy(dtype=float)
        bar_colors = [colors.get(name, '#888888') for name in labels]
        ax.bar(x, vals, yerr=errs, color=bar_colors, capsize=5, width=0.65, edgecolor='black', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(f'Test {title} (mean ± std)')
        ax.set_ylabel(title)
        ymin = max(0.0, float(np.min(vals - errs)) - 0.03)
        ymax = min(1.0, float(np.max(vals + errs)) + 0.03)
        ax.set_ylim(ymin, ymax)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        for xi, val, err in zip(x, vals, errs):
            ax.text(xi, val + err + 0.005, f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    fig.suptitle('Coronary Segmentation Linear Probe', fontsize=13)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def build_summary(source_root: Path, selected_layer_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    agg = pd.read_csv(source_root / 'summary_layerwise_segmentation_linear_probe.csv')
    raw = pd.read_csv(source_root / 'summary_layerwise_segmentation_linear_probe_raw.csv')
    agg_last = sort_backbones(agg.loc[agg['layer_id'] == selected_layer_id].copy())
    raw_last = raw.loc[raw['layer_id'] == selected_layer_id].copy()
    raw_last = raw_last.assign(_backbone_order=raw_last['backbone_name'].map(BACKBONE_ORDER).fillna(999)).sort_values(
        ['seed', '_backbone_order', 'split', 'backbone_name']
    ).drop(columns=['_backbone_order']).reset_index(drop=True)

    final_rows = []
    for _, row in agg_last.iterrows():
        final_rows.append({
            'backbone_name': row['backbone_name'],
            'split': row['split'],
            'best_lr': row['best_lr_mode'],
            'epochs_trained': row['epochs_trained_mean'],
            'num_images': int(row['num_images']),
            'pixel_acc': row['pixel_acc_mean'],
            'miou': row['miou_mean'],
            'dice': row['dice_mean'],
            'per_class_iou': row['per_class_iou_mean'],
            'per_class_dice': row['per_class_dice_mean'],
            'probe_params': int(row['probe_params']),
            'num_seeds': int(row['num_seeds']),
            'miou_std': row['miou_std'],
            'dice_std': row['dice_std'],
            'pixel_acc_std': row['pixel_acc_std'],
            'epochs_trained_std': row['epochs_trained_std'],
            'source_layer_id': int(row['layer_id']),
            'source_block_index': int(row['block_index']),
        })
    final_df = pd.DataFrame(final_rows)
    raw_out = raw_last[[
        'seed', 'backbone_name', 'split', 'best_lr', 'epochs_trained', 'num_images', 'pixel_acc', 'miou', 'dice',
        'per_class_iou', 'per_class_dice', 'probe_params', 'layer_id', 'block_index'
    ]].copy()
    return final_df, raw_out


def write_markdown(output_root: Path, summary_df: pd.DataFrame, source_root: Path, selected_layer_id: int) -> None:
    layer_block = int(summary_df['source_block_index'].iloc[0]) if not summary_df.empty else selected_layer_id - 1
    lines: list[str] = []
    lines.append('# Coronary Segmentation Linear Probe')
    lines.append('')
    lines.append('## Setup')
    lines.append('')
    lines.append('- benchmark dataset: `input/contrast_benchmark/2_dense/1_coronary_segmentation`')
    lines.append(f'- source baseline result: `{source_root}`')
    lines.append(f'- selected representation: `layer {selected_layer_id}` (block index `{layer_block}`)')
    lines.append('- aggregation: mean ± std over fixed seed set `{11, 22, 33}`')
    lines.append('- Exp0 keeps only the last-layer multiseed result and does not expose layer-wise comparison artifacts here.')
    lines.append('')
    lines.append('## Test Metrics')
    lines.append('')
    lines.append('| backbone | mIoU | Dice | Pixel Acc | num_seeds |')
    lines.append('| --- | --- | --- | --- | --- |')
    for _, row in sort_backbones(summary_df.loc[summary_df['split'] == 'test'].copy()).iterrows():
        lines.append(f"| {row['backbone_name']} | {row['miou']:.6f} ± {row['miou_std']:.6f} | {row['dice']:.6f} ± {row['dice_std']:.6f} | {row['pixel_acc']:.6f} ± {row['pixel_acc_std']:.6f} | {int(row['num_seeds'])} |")
    lines.append('')
    lines.append('## Valid Metrics')
    lines.append('')
    lines.append('| backbone | mIoU | Dice | Pixel Acc | num_seeds |')
    lines.append('| --- | --- | --- | --- | --- |')
    for _, row in sort_backbones(summary_df.loc[summary_df['split'] == 'valid'].copy()).iterrows():
        lines.append(f"| {row['backbone_name']} | {row['miou']:.6f} ± {row['miou_std']:.6f} | {row['dice']:.6f} ± {row['dice_std']:.6f} | {row['pixel_acc']:.6f} ± {row['pixel_acc_std']:.6f} | {int(row['num_seeds'])} |")
    lines.append('')
    lines.append('## Files')
    lines.append('')
    lines.append('- summary: `summary_segmentation_linear_probe.csv`')
    lines.append('- per-seed last-layer summary: `summary_segmentation_linear_probe_raw.csv`')
    lines.append('- figure: `fig_seg_linear_probe_bar_compare.png`')
    (output_root / 'analysis_segmentation_linear_probe.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()

    backup_path: Path | None = None
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f'Output root exists: {output_root}. Pass --overwrite to replace it.')
    if output_root.exists() and args.overwrite:
        backup_path = output_root.with_name(output_root.name + '_before_cleanup_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
        shutil.move(str(output_root), str(backup_path))

    ensure_dir(output_root)

    summary_df, raw_df = build_summary(source_root, args.selected_layer_id)
    summary_df.to_csv(output_root / 'summary_segmentation_linear_probe.csv', index=False)
    raw_df.to_csv(output_root / 'summary_segmentation_linear_probe_raw.csv', index=False)

    for name in [
        'image_manifest_train.csv', 'image_manifest_train.meta.json',
        'image_manifest_valid.csv', 'image_manifest_valid.meta.json',
        'image_manifest_test.csv', 'image_manifest_test.meta.json',
        'targets_train.pt', 'targets_train.meta.json',
        'targets_valid.pt', 'targets_valid.meta.json',
        'targets_test.pt', 'targets_test.meta.json',
    ]:
        shutil.copy2(source_root / name, output_root / name)

    save_bar_compare(summary_df, output_root / 'fig_seg_linear_probe_bar_compare.png')
    write_markdown(output_root, summary_df, source_root, args.selected_layer_id)

    note_lines = [
        '# Exp0 Reuse Note',
        '',
        'This directory is populated from the equivalent multiseed baseline output at',
        f'`{source_root}`.',
        '',
        'Policy:',
        f'- use only `layer {args.selected_layer_id}` (last layer) results',
        '- aggregate over seeds `{11, 22, 33}`',
        '- do not expose layer-wise comparison figures in the Exp0 dense benchmark root',
    ]
    (output_root / 'EXP0_REUSE_NOTE.md').write_text('\n'.join(note_lines) + '\n', encoding='utf-8')

    run_meta = {
        'analysis': 'coronary_segmentation_linear_probe_exp0_baseline',
        'benchmark_dataset_root': str(Path('input/contrast_benchmark/2_dense/1_coronary_segmentation').resolve()),
        'source_result_root': str(source_root),
        'source_result_type': 'local_2_1_layerwise_segmentation_linear_probe_multiseed',
        'selection_policy': 'last_layer_only',
        'selected_layer_id': args.selected_layer_id,
        'selected_block_index': int(summary_df['source_block_index'].iloc[0]) if not summary_df.empty else args.selected_layer_id - 1,
        'seed_set': [11, 22, 33],
        'aggregate_policy': 'mean_std_with_paired_delta_from_source_multiseed_run',
        'summary_path': str((output_root / 'summary_segmentation_linear_probe.csv').resolve()),
        'raw_summary_path': str((output_root / 'summary_segmentation_linear_probe_raw.csv').resolve()),
        'figure_path': str((output_root / 'fig_seg_linear_probe_bar_compare.png').resolve()),
        'backup_replaced_output_root': str(backup_path.resolve()) if backup_path else None,
        'created_at': datetime.now().isoformat(timespec='seconds'),
    }
    (output_root / 'run_meta.json').write_text(json.dumps(run_meta, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
