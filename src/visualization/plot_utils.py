"""
Shared visualization utilities for attention experiments.

Contains: Style setup, error-vs-budget curves, scatter plots, bar charts,
          log axis formatting, fig_to_base64, save_figure, METHOD_STYLES
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import LogLocator, FuncFormatter, MultipleLocator
from pathlib import Path
import base64
from io import BytesIO


# Method display configuration
METHOD_STYLES = {
    'TopK':             {'marker': 'o', 'label': 'Top-K',              'color': '#d62728', 'linestyle': '-'},
    'Uniform':          {'marker': 's', 'label': 'Uniform Sampling',   'color': '#ff7f0e', 'linestyle': '-'},
    'Oracle':           {'marker': '^', 'label': 'Oracle',             'color': '#2ca02c', 'linestyle': '-'},
    'JungleSampling':   {'marker': 'D', 'label': 'Jungle Sampling',   'color': '#9467bd', 'linestyle': '-'},
    'SimHash-SNIS':     {'marker': 'x', 'label': 'SimHash-SNIS',      'color': '#1f77b4', 'linestyle': 'none'},
    'CrossPoly-SNIS':   {'marker': '+', 'label': 'CrossPoly-SNIS',    'color': '#e377c2', 'linestyle': 'none'},
    'FullAttention':    {'marker': '*', 'label': 'Full Attention',     'color': '#7f7f7f', 'linestyle': '--'},
}

LAYER_TITLES = {
    'first_layer': 'First Layer (Layer 0)',
    'last_layer': 'Last Layer (Layer 31)',
}


def setup_style():
    """Configure matplotlib for publication-quality plots."""
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Verdana', 'Liberation Sans']
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10


def format_log_yaxis(ax):
    """Format log-scale y-axis with nice tick labels."""
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=20))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=[2, 3, 4, 5, 6, 7, 8, 9], numticks=200))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}' if y < 1 else f'{y:.1f}'))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda y, _: ''))


def plot_error_vs_budget(ax, budgets, means, stds, method_name, **kwargs):
    """
    Plot error-vs-budget curve with shaded variance.

    Args:
        ax: matplotlib axis
        budgets: list of budget values
        means: list of mean errors
        stds: list of std errors
        method_name: key into METHOD_STYLES
    """
    style = METHOD_STYLES.get(method_name, {'marker': 'o', 'color': 'gray', 'label': method_name, 'linestyle': '-'})
    color = kwargs.get('color', style['color'])
    label = kwargs.get('label', style['label'])

    budgets_arr = np.array(budgets)
    means_arr = np.array(means)
    stds_arr = np.array(stds)

    upper = means_arr + stds_arr
    lower = np.maximum(means_arr - stds_arr, 1e-6)

    ax.fill_between(budgets_arr, lower, upper, color=color, alpha=0.2, linewidth=0, zorder=1)
    ax.plot(budgets_arr, means_arr, marker=style['marker'], linestyle=style.get('linestyle', '-'),
            label=label, color=color, linewidth=2.5, markersize=6, alpha=0.95, zorder=3)


def plot_scatter_with_fit(ax, budgets, errors, method_name, **kwargs):
    """
    Plot scatter points for variable-budget methods (SNIS) with optional power-law fit.
    """
    style = METHOD_STYLES.get(method_name, {'marker': 'x', 'color': 'gray', 'label': method_name})
    color = kwargs.get('color', style['color'])
    label = kwargs.get('label', style['label'])

    ax.scatter(budgets, errors, marker=style['marker'], s=120, linewidths=2.5,
               color=color, alpha=0.8, label=label, zorder=5)

    # Optional power-law fit
    if len(budgets) >= 3:
        log_budgets = np.log(budgets)
        log_errors = np.log(errors)
        coeffs = np.polyfit(log_budgets, log_errors, 1)
        budget_range = np.linspace(min(budgets), max(budgets), 100)
        fitted_errors = np.exp(np.poly1d(coeffs)(np.log(budget_range)))
        ax.plot(budget_range, fitted_errors, '--', color=color, linewidth=2, alpha=0.6,
                label=f'{label} fit (error ~ budget^{coeffs[0]:.2f})')


def save_figure(fig, path, dpi=150):
    """Save figure with tight layout."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_to_base64(fig, dpi=150):
    """Convert matplotlib figure to base64 encoded PNG string."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"
