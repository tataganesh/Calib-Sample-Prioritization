import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import imageio


def plot_metrics(metric, metric_name):
    plt.plot(range(len(metric)), metric)
    plt.title(f'{metric_name}')
    plt.xlabel('Epochs')
    plt.ylabel(f'{metric_name}')
    # plt.legend()
    # plt.show()
    figdir = 'figs'
    if not os.path.isdir(figdir):
        os.makedirs(figdir)
    figpath = os.path.join(figdir, f'{metric_name}.jpg')
    plt.savefig(figpath)
    plt.close()

    return figpath


def plot_reliability_diagram(calib_rel_diag, metric_name, n_bins=10):
    _, calib_acc = calib_rel_diag
    # computations
    delta = 1.0 / n_bins
    x = torch.arange(0, 1, delta)
    mid = torch.linspace(delta/2, 1-delta/2, n_bins)
    error = torch.absolute(torch.subtract(mid, calib_acc))

    plt.rcParams['font.family'] = 'serif'
    # size and axis limits
    # plt.figure(figsize=(3, 3))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    # plot grid
    plt.grid(color='tab:grey', linestyle=(0, (1, 5)), linewidth=1, zorder=0)
    # plot bars and identity line
    plt.bar(x, calib_acc, color='b', width=delta, align='edge', edgecolor='k', label='Outputs', zorder=5)
    plt.bar(x, error, bottom=torch.min(calib_acc, mid), color='mistyrose', alpha=0.5, width=delta,
            align='edge', edgecolor='r', hatch='/', label='Gap', zorder=10)
    ident = [0.0, 1.0]
    plt.plot(ident, ident, linestyle='--', color='tab:grey', zorder=15)
    # labels and legend
    plt.ylabel('Accuracy')  # fontsize=13)
    plt.xlabel('Confidence')  # fontsize=13)
    plt.legend(loc='upper left', framealpha=1.0, fontsize='medium')
    plt.title('Reliability Diagram')
    plt.tight_layout()

    figdir = 'figs'
    if not os.path.isdir(figdir):
        os.makedirs(figdir)
    figpath = os.path.join(figdir, f'{metric_name}.jpg')
    plt.savefig(figpath)
    plt.close()

    return figpath


def plot_class_freqs(imp_labels_epoch, i, subset_size):
    plt.rcParams["figure.figsize"] = (10, 8)
    keys = imp_labels_epoch.keys()
    freqs = [imp_labels_epoch[j][i] for j in keys]
    cmap = plt.cm.get_cmap('hsv', 10)
    colors = [cmap(j) for j in range(10)]

    plt.title(f'Class Frequency at Epoch {i}    Total Size: {subset_size}')
    plt.bar(range(10), freqs, color=colors)

    for k in range(10):
        plt.text(k, freqs[k]+0.5, f'{freqs[k].item():.2f}', ha='center', weight='bold')

    plt.xlabel('Classes')
    plt.ylabel('Frequency (%)')
    plt.xticks(np.arange(10), keys)

    figname = f'Epoch_{i}.jpg'
    figdir = 'figs'
    if not os.path.isdir(figdir):
        os.makedirs(figdir)
    figpath = os.path.join(figdir, figname)
    plt.savefig(figpath)
    plt.close()

    return figpath


def create_gif(gif_figpaths):
    figdir = 'figs'
    subset_class_freqs_gif_path = os.path.join(figdir, 'subset_class_frequencies.gif')

    images = []
    for figpath in gif_figpaths:
        images.append(imageio.imread(figpath))
        # if not args.keep_figures:
        os.remove(figpath)
    imageio.mimsave(subset_class_freqs_gif_path, images, fps=1)
