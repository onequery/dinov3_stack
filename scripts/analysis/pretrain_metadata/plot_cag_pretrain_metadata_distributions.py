#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, TextIO

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIELD_NAME_RE = re.compile(r"[^0-9A-Za-z._-]+")
SPLIT_ORDER = ['train', 'val', 'test']
SPLIT_COLORS = {'train': '#1f77b4', 'val': '#ff7f0e', 'test': '#2ca02c'}


class TeeStream:
    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def log(message: str) -> None:
    stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{stamp}] {message}', flush=True)


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f'{seconds:04.1f}s'
    total = int(round(seconds))
    hh = total // 3600
    mm = (total % 3600) // 60
    ss = total % 60
    return f'{hh:02d}:{mm:02d}:{ss:02d}'


def estimate_remaining(elapsed_seconds: float, done: int, total: int) -> float:
    if total <= 0 or done <= 0:
        return float('nan')
    avg = elapsed_seconds / max(done, 1)
    return max(0.0, avg * (total - done))


def sanitize_name(text: str) -> str:
    out = FIELD_NAME_RE.sub('_', str(text)).strip('_')
    return out or 'unnamed_field'


def normalize_splits(selected_splits: list[str] | None) -> list[str]:
    if not selected_splits:
        return list(SPLIT_ORDER)
    seen: set[str] = set()
    ordered: list[str] = []
    for split in SPLIT_ORDER:
        if split in selected_splits and split not in seen:
            ordered.append(split)
            seen.add(split)
    for split in selected_splits:
        if split not in seen:
            ordered.append(split)
            seen.add(split)
    return ordered


@dataclass(frozen=True)
class PlotRow:
    field_name: str
    field_type: str
    plot_type: str
    output_path: str
    note: str


def setup_logging(output_root: Path) -> tuple[Path, TextIO, TextIO, TextIO]:
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = output_root / f'plot_cag_pretrain_metadata_distributions_{stamp}.log'
    fh = open(log_path, 'a', encoding='utf-8', buffering=1)
    orig_out = sys.stdout
    orig_err = sys.stderr
    sys.stdout = TeeStream(orig_out, fh)
    sys.stderr = TeeStream(orig_err, fh)
    log(f'Console output is mirrored to log file: {log_path}')
    return log_path, fh, orig_out, orig_err


def restore_logging(fh: TextIO, orig_out: TextIO, orig_err: TextIO) -> None:
    sys.stdout = orig_out
    sys.stderr = orig_err
    fh.close()


def shorten_label(value: str, max_len: int = 48) -> str:
    text = str(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + '…'


def load_field_catalog(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def load_numeric_summary(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def split_categorical_summary_by_field(categorical_summary_path: Path, output_dir: Path, selected_splits: list[str], log_every_rows: int = 100000) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    handles: dict[str, TextIO] = {}
    writers: dict[str, csv.DictWriter] = {}
    field_to_path: dict[str, Path] = {}
    total_rows = 0
    start = time.time()
    with categorical_summary_path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = ['split', 'field_name', 'value', 'count', 'ratio']
        for row in reader:
            total_rows += 1
            if row.get('split') not in selected_splits:
                continue
            field_name = row['field_name']
            if field_name not in handles:
                field_path = output_dir / f'{sanitize_name(field_name)}.csv'
                fh = open(field_path, 'w', encoding='utf-8', newline='')
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                handles[field_name] = fh
                writers[field_name] = writer
                field_to_path[field_name] = field_path
            writers[field_name].writerow({key: row.get(key, '') for key in fieldnames})
            if log_every_rows > 0 and total_rows % log_every_rows == 0:
                log(f'[CATEGORICAL-SPLIT] rows={total_rows} | fields={len(field_to_path)} | elapsed={format_duration(time.time() - start)}')
    for fh in handles.values():
        fh.close()
    log(f'[CATEGORICAL-SPLIT] DONE | rows={total_rows} | fields={len(field_to_path)} | elapsed={format_duration(time.time() - start)}')
    return field_to_path


def plot_numeric_field(field_name: str, field_meta: pd.Series, field_df: pd.DataFrame, output_path: Path, selected_splits: list[str]) -> PlotRow:
    plot_df = field_df[field_df['field_name'] == field_name].copy()
    plot_df['split'] = pd.Categorical(plot_df['split'], categories=selected_splits, ordered=True)
    plot_df = plot_df.sort_values('split').reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    y_positions = np.arange(len(plot_df))
    for idx, row in enumerate(plot_df.itertuples(index=False)):
        color = SPLIT_COLORS.get(row.split, '#4c4c4c')
        ax.hlines(idx, row.p01, row.p99, color=color, linewidth=1.5, alpha=0.35)
        ax.hlines(idx, row.p05, row.p95, color=color, linewidth=3.0, alpha=0.55)
        ax.hlines(idx, row.p25, row.p75, color=color, linewidth=8.0, alpha=0.9)
        ax.plot(row.p50, idx, marker='o', markersize=8, color='black')
        ax.plot([row.min, row.max], [idx, idx], linestyle='None', marker='|', markersize=10, color=color, alpha=0.75)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df['split'].astype(str).tolist())
    ax.set_xlabel(field_name)
    split_label = selected_splits[0] if len(selected_splits) == 1 else ', '.join(selected_splits)
    ax.set_title(f'{field_name} | Quantile Distribution ({split_label})')
    ax.grid(axis='x', alpha=0.2)
    note = (
        f"missing_ratio={field_meta['missing_ratio']:.4f}, "
        f"present_count={int(field_meta['present_count'])}, "
        f"splits={','.join(selected_splits)}"
    )
    fig.text(0.01, 0.01, note, ha='left', va='bottom', fontsize=9)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return PlotRow(field_name=field_name, field_type='continuous', plot_type='quantile_interval', output_path=str(output_path), note=note)


def plot_categorical_field(field_name: str, field_meta: pd.Series, field_csv_path: Path, output_path: Path, top_n: int, all_n_threshold: int, selected_splits: list[str]) -> PlotRow:
    df = pd.read_csv(field_csv_path, low_memory=False, dtype={'value': str})
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, 'No categorical rows', ha='center', va='center')
        ax.axis('off')
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return PlotRow(field_name=field_name, field_type='categorical', plot_type='empty', output_path=str(output_path), note='empty')

    df['count'] = pd.to_numeric(df['count'], errors='coerce').fillna(0).astype(float)
    pivot = df.pivot_table(index='value', columns='split', values='count', aggfunc='sum', fill_value=0.0)
    for split in selected_splits:
        if split not in pivot.columns:
            pivot[split] = 0.0
    pivot = pivot[selected_splits]
    pivot['__total__'] = pivot.sum(axis=1)
    cardinality = int(pivot.shape[0])

    if cardinality <= all_n_threshold:
        plot_df = pivot.sort_values('__total__', ascending=True).drop(columns='__total__')
        plot_type = 'all_values_stacked_bar'
        fig_height = max(4.0, min(20.0, 0.22 * len(plot_df) + 2.0))
        fig, ax = plt.subplots(figsize=(11, fig_height))
        bottom = np.zeros(len(plot_df))
        y = np.arange(len(plot_df))
        for split in selected_splits:
            values = plot_df[split].to_numpy(dtype=float)
            ax.barh(y, values, left=bottom, color=SPLIT_COLORS[split], label=split, alpha=0.9)
            bottom += values
        ax.set_yticks(y)
        ax.set_yticklabels([shorten_label(v) for v in plot_df.index.tolist()])
        ax.set_xlabel('Count')
        split_label = selected_splits[0] if len(selected_splits) == 1 else ', '.join(selected_splits)
        ax.set_title(f'{field_name} | All Values ({split_label}, cardinality={cardinality})')
        ax.legend(loc='best')
        ax.grid(axis='x', alpha=0.2)
        split_text = ','.join(selected_splits)
        note = f'missing_ratio={field_meta["missing_ratio"]:.4f}, cardinality={cardinality}, splits={split_text}'
        fig.text(0.01, 0.01, note, ha='left', va='bottom', fontsize=9)
        fig.tight_layout(rect=[0, 0.03, 1, 1])
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return PlotRow(field_name=field_name, field_type='categorical', plot_type=plot_type, output_path=str(output_path), note=note)

    top_df = pivot.sort_values('__total__', ascending=False).head(top_n).sort_values('__total__', ascending=True)
    counts = pivot['__total__'].sort_values(ascending=False).to_numpy(dtype=float)
    plot_type = 'top_values_plus_rank'
    fig_height = max(6.0, min(18.0, 0.24 * len(top_df) + 2.5))
    fig, axes = plt.subplots(1, 2, figsize=(15, fig_height), gridspec_kw={'width_ratios': [1.6, 1.0]})
    ax0, ax1 = axes
    bottom = np.zeros(len(top_df))
    y = np.arange(len(top_df))
    for split in selected_splits:
        values = top_df[split].to_numpy(dtype=float)
        ax0.barh(y, values, left=bottom, color=SPLIT_COLORS[split], label=split, alpha=0.9)
        bottom += values
    ax0.set_yticks(y)
    ax0.set_yticklabels([shorten_label(v) for v in top_df.index.tolist()])
    ax0.set_xlabel('Count')
    split_label = selected_splits[0] if len(selected_splits) == 1 else ', '.join(selected_splits)
    ax0.set_title(f'{field_name} | Top {len(top_df)} Values ({split_label})')
    ax0.legend(loc='best')
    ax0.grid(axis='x', alpha=0.2)

    ranks = np.arange(1, len(counts) + 1)
    ax1.plot(ranks, counts, color='#444444', linewidth=1.5)
    ax1.set_yscale('log')
    ax1.set_xlabel('Value Rank')
    ax1.set_ylabel('Total Count (log scale)')
    ax1.set_title(f'{field_name} | Rank-Frequency (cardinality={cardinality})')
    ax1.grid(alpha=0.2)

    split_text = ','.join(selected_splits)
    note = f'missing_ratio={field_meta["missing_ratio"]:.4f}, cardinality={cardinality}, top_n={top_n}, splits={split_text}'
    fig.text(0.01, 0.01, note, ha='left', va='bottom', fontsize=9)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return PlotRow(field_name=field_name, field_type='categorical', plot_type=plot_type, output_path=str(output_path), note=note)


def save_inventory(rows: Iterable[PlotRow], output_path: Path) -> None:
    data = [r.__dict__ for r in rows]
    pd.DataFrame(data).sort_values(['field_type', 'field_name']).to_csv(output_path, index=False)


def save_markdown_inventory(plot_rows: list[PlotRow], output_path: Path) -> None:
    lines = ['# CAG Pretrain Metadata Distribution Plots', '']
    lines.append(f'- total_plots: `{len(plot_rows)}`')
    lines.append(f'- continuous_plots: `{sum(r.field_type == "continuous" for r in plot_rows)}`')
    lines.append(f'- categorical_plots: `{sum(r.field_type == "categorical" for r in plot_rows)}`')
    lines.append('')
    lines.append('## Output Directories')
    lines.append('')
    lines.append('- `continuous/`')
    lines.append('- `categorical/`')
    lines.append('- `_categorical_cache/`')
    lines.append('')
    lines.append('## Plot Inventory')
    lines.append('')
    inv = pd.DataFrame([r.__dict__ for r in plot_rows]).sort_values(['field_type', 'field_name'])
    if inv.empty:
        lines.append('_Empty._')
    else:
        lines.append('| field_name | field_type | plot_type | output_path | note |')
        lines.append('| --- | --- | --- | --- | --- |')
        for row in inv.itertuples(index=False):
            lines.append(f'| {row.field_name} | {row.field_type} | {row.plot_type} | {row.output_path} | {row.note} |')
    output_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot CAG pretraining metadata distributions.')
    parser.add_argument('--stats-root', type=Path, default=Path('outputs/cag_pretrain_metadata_stats'))
    parser.add_argument('--output-root', type=Path, default=None)
    parser.add_argument('--top-categorical-values', type=int, default=40)
    parser.add_argument('--all-categorical-threshold', type=int, default=50)
    parser.add_argument('--categorical-log-every-rows', type=int, default=100000)
    parser.add_argument('--splits', nargs='+', default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats_root = args.stats_root.resolve()
    output_root = args.output_root.resolve() if args.output_root is not None else (stats_root / 'plots')
    output_root.mkdir(parents=True, exist_ok=True)
    log_path, fh, orig_out, orig_err = setup_logging(output_root)
    try:
        selected_splits = normalize_splits(args.splits)
        log('Arguments: ' + str({
            'stats_root': str(stats_root),
            'output_root': str(output_root),
            'top_categorical_values': int(args.top_categorical_values),
            'all_categorical_threshold': int(args.all_categorical_threshold),
            'categorical_log_every_rows': int(args.categorical_log_every_rows),
            'splits': selected_splits,
            'log_path': str(log_path),
        }))

        field_catalog = load_field_catalog(stats_root / 'per_image' / 'field_catalog.csv')
        numeric_summary = load_numeric_summary(stats_root / 'summary_numeric_by_split.csv')
        numeric_summary = numeric_summary[numeric_summary['split'].isin(selected_splits)].copy()
        categorical_summary_path = stats_root / 'summary_categorical_all_values_by_split.csv'

        categorical_cache_dir = output_root / '_categorical_cache'
        continuous_dir = output_root / 'continuous'
        categorical_dir = output_root / 'categorical'
        continuous_dir.mkdir(parents=True, exist_ok=True)
        categorical_dir.mkdir(parents=True, exist_ok=True)

        stage_start = time.time()
        log('CATEGORICAL SPLIT START')
        field_to_categorical_csv = split_categorical_summary_by_field(
            categorical_summary_path,
            categorical_cache_dir,
            selected_splits=selected_splits,
            log_every_rows=int(args.categorical_log_every_rows),
        )
        log(f'CATEGORICAL SPLIT DONE | elapsed={format_duration(time.time() - stage_start)}')

        plot_rows: list[PlotRow] = []

        numeric_fields = field_catalog[field_catalog['field_type'] == 'continuous']['field_name'].tolist()
        num_start = time.time()
        log(f'NUMERIC PLOT START | fields={len(numeric_fields)}')
        for idx, field_name in enumerate(numeric_fields, start=1):
            field_meta = field_catalog[field_catalog['field_name'] == field_name].iloc[0]
            output_path = continuous_dir / f'{sanitize_name(field_name)}.png'
            plot_rows.append(plot_numeric_field(field_name, field_meta, numeric_summary, output_path, selected_splits))
            if idx == len(numeric_fields) or idx % 10 == 0:
                elapsed = time.time() - num_start
                remaining = estimate_remaining(elapsed, idx, len(numeric_fields))
                log(f'[NUMERIC PLOT] done={idx}/{len(numeric_fields)} | elapsed={format_duration(elapsed)} | remaining={format_duration(remaining) if math.isfinite(remaining) else "n/a"}')
        log(f'NUMERIC PLOT DONE | elapsed={format_duration(time.time() - num_start)}')

        categorical_fields = field_catalog[field_catalog['field_type'] == 'categorical']['field_name'].tolist()
        cat_start = time.time()
        log(f'CATEGORICAL PLOT START | fields={len(categorical_fields)}')
        for idx, field_name in enumerate(categorical_fields, start=1):
            field_meta = field_catalog[field_catalog['field_name'] == field_name].iloc[0]
            output_path = categorical_dir / f'{sanitize_name(field_name)}.png'
            field_csv = field_to_categorical_csv.get(field_name)
            if field_csv is None:
                continue
            plot_rows.append(
                plot_categorical_field(
                    field_name,
                    field_meta,
                    field_csv,
                    output_path,
                    top_n=int(args.top_categorical_values),
                    all_n_threshold=int(args.all_categorical_threshold),
                    selected_splits=selected_splits,
                )
            )
            if idx == len(categorical_fields) or idx % 10 == 0:
                elapsed = time.time() - cat_start
                remaining = estimate_remaining(elapsed, idx, len(categorical_fields))
                log(f'[CATEGORICAL PLOT] done={idx}/{len(categorical_fields)} | elapsed={format_duration(elapsed)} | remaining={format_duration(remaining) if math.isfinite(remaining) else "n/a"}')
        log(f'CATEGORICAL PLOT DONE | elapsed={format_duration(time.time() - cat_start)}')

        inventory_path = output_root / 'plot_inventory.csv'
        save_inventory(plot_rows, inventory_path)
        save_markdown_inventory(plot_rows, output_root / 'plot_inventory.md')
        log(f'INVENTORY DONE | plots={len(plot_rows)}')
    finally:
        restore_logging(fh, orig_out, orig_err)


if __name__ == '__main__':
    main()
