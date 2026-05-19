import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRACKED_METRICS = ['loss', 'loss_neg', 'auroc', 'auprc', 'minrp']
PLOTTED_METRICS = ['auroc', 'auprc', 'minrp']
BASE_COLUMNS = [
    'train_step',
    'epoch',
    'recent_mean_train_loss',
    'val_loss',
    'val_loss_neg',
    'val_auroc',
    'val_auprc',
    'val_minrp',
    'test_loss',
    'test_loss_neg',
    'test_auroc',
    'test_auprc',
    'test_minrp',
]


def _to_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def append_learning_curve_row(
    history,
    train_step,
    epoch,
    recent_mean_train_loss,
    val_res=None,
    test_res=None,
):
    row = {
        'train_step': train_step,
        'epoch': _to_float(epoch),
        'recent_mean_train_loss': _to_float(recent_mean_train_loss),
    }

    for prefix, results in [('val', val_res), ('test', test_res)]:
        if results is None:
            continue
        for metric in TRACKED_METRICS:
            if metric in results:
                row[f'{prefix}_{metric}'] = _to_float(results.get(metric))

    history.append(row)
    return row


def _history_to_frame(history):
    df = pd.DataFrame(history)
    for column in BASE_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
    extra_columns = [column for column in df.columns if column not in BASE_COLUMNS]
    return df[BASE_COLUMNS + sorted(extra_columns)]


def save_learning_curve_history(history, output_dir):
    df = _history_to_frame(history)
    path = os.path.join(output_dir, 'learning_curve_history.csv')
    df.to_csv(path, index=False)
    return df


def _plot_columns(df, columns, output_path, title, ylabel):
    fig, ax = plt.subplots(figsize=(10, 6))
    x_values = pd.to_numeric(df['train_step'], errors='coerce')
    plotted = False

    for column in columns:
        if column not in df.columns:
            continue
        y_values = pd.to_numeric(df[column], errors='coerce')
        valid = x_values.notna() & y_values.notna()
        if not valid.any():
            continue
        ax.plot(
            x_values[valid],
            y_values[valid],
            marker='o',
            linewidth=1.5,
            label=column,
        )
        plotted = True

    ax.set_title(title)
    ax.set_xlabel('train_step')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            'No learning-curve data yet',
            transform=ax.transAxes,
            ha='center',
            va='center',
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_learning_curves(history, output_dir):
    df = _history_to_frame(history)
    metric_columns = []
    for metric in PLOTTED_METRICS:
        for prefix in ['val', 'test']:
            column = f'{prefix}_{metric}'
            if column in df.columns and df[column].notna().any():
                metric_columns.append(column)

    _plot_columns(
        df,
        metric_columns,
        os.path.join(output_dir, 'learning_curve_metrics.png'),
        'Validation and Test Metrics',
        'metric value',
    )
    _plot_columns(
        df,
        ['recent_mean_train_loss', 'val_loss', 'test_loss'],
        os.path.join(output_dir, 'learning_curve_loss.png'),
        'Training, Validation, and Test Loss',
        'loss',
    )


def save_learning_curves(history, output_dir):
    df = save_learning_curve_history(history, output_dir)
    plot_learning_curves(history, output_dir)
    return df
