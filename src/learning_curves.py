import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRACKED_METRICS = ['loss', 'loss_neg', 'auroc', 'auprc', 'minrp']
PLOTTED_METRICS = ['auroc', 'auprc', 'minrp']
LOSS_PLOT_COLUMNS = ['recent_mean_train_loss', 'val_loss', 'test_loss']
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


def _log(logger, message):
    if logger is None:
        print(message)
    else:
        logger.write(message)


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
    logger=None,
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
    _log(logger, 'Learning-curve row appended: ' + str(row))
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


def _metric_plot_columns():
    return [
        f'{prefix}_{metric}'
        for metric in PLOTTED_METRICS
        for prefix in ['val', 'test']
    ]


def _numeric_valid_counts(df, columns):
    x_values = pd.to_numeric(df['train_step'], errors='coerce')
    valid_counts = {}
    for column in columns:
        if column not in df.columns:
            valid_counts[column] = 0
            continue
        y_values = pd.to_numeric(df[column], errors='coerce')
        valid_counts[column] = int((x_values.notna() & y_values.notna()).sum())
    return valid_counts


def _numeric_columns(df, columns):
    return [
        column
        for column, valid_count in _numeric_valid_counts(df, columns).items()
        if valid_count > 0
    ]


def _plot_columns(df, columns, output_path, title, ylabel, logger=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    x_values = pd.to_numeric(df['train_step'], errors='coerce')
    plotted = False
    valid_counts = _numeric_valid_counts(df, columns)
    _log(
        logger,
        'Learning-curve plot candidates: output_path=' + output_path
        + ', columns=' + str(columns)
        + ', numeric_valid_counts=' + str(valid_counts)
        + ', will_plot=' + str(any(valid_counts.values())),
    )

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
    _log(
        logger,
        'Learning-curve plot saved: output_path=' + output_path
        + ', plotted=' + str(plotted),
    )


def plot_learning_curves(history, output_dir, logger=None):
    df = _history_to_frame(history)
    metric_columns = _metric_plot_columns()

    _plot_columns(
        df,
        metric_columns,
        os.path.join(output_dir, 'learning_curve_metrics.png'),
        'Validation and Test Metrics',
        'metric value',
        logger=logger,
    )
    _plot_columns(
        df,
        LOSS_PLOT_COLUMNS,
        os.path.join(output_dir, 'learning_curve_loss.png'),
        'Training, Validation, and Test Loss',
        'loss',
        logger=logger,
    )


def save_learning_curves(history, output_dir, logger=None):
    df = save_learning_curve_history(history, output_dir)
    plot_learning_curves(history, output_dir, logger=logger)
    _log(
        logger,
        'Learning-curve save diagnostics: output_dir=' + output_dir
        + ', history_rows=' + str(len(history))
        + ', dataframe_columns=' + str(list(df.columns)),
    )
    if df.empty:
        _log(logger, 'Learning-curve DataFrame tail(3): <empty>')
    else:
        _log(logger, 'Learning-curve DataFrame tail(3):\n' + df.tail(3).to_string(index=False))
    _log(
        logger,
        'Learning-curve numeric plot columns: loss='
        + str(_numeric_columns(df, LOSS_PLOT_COLUMNS))
        + ', metrics=' + str(_numeric_columns(df, _metric_plot_columns())),
    )
    if not _numeric_columns(df, ['recent_mean_train_loss']):
        _log(
            logger,
            'Learning-curve history has no numeric recent_mean_train_loss; '
            'no post-training validation row is stored in this process.',
        )
    return df
