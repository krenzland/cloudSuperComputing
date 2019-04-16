#!/usr/bin/env python3
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from plot_utils import *
from dataset_utils import *

from matplotlib import ticker


def plot_cosine_bubble(fig_path, solution_path, target, order=4):
    df, time, order, dim, dx, n_points = read_solution(solution_path, order=order, size=1000)
    print(time, order, dim, dx, n_points)

    # Fig 4 giraldo
    df_cut = cut_dataset(df, 'y', dim=2, cut_at=[None, None])
    fig, ax = plt.subplots(figsize=quadratic_figsize(scale=0.5, target=target))
    ax.plot(df_cut['y'], df_cut['potT']-300, c='black')
    ax.set_xlim(850, 1000)
    ax.set_ylim(-0.1, 0.6)
    ax.set_xlabel('$z [\si{m}]$')
    ax.set_ylabel("$\Theta\'$")
    fig.subplots_adjust(left=0.2)

    fig.savefig(fig_path + f'{target}_cosine_bubble_pertubation.pdf', transparent=True)

    fig, ax = plt.subplots(figsize=quadratic_figsize(0.5, target=target))
    df['potTPertubation'] = df['potT'] - 300
    levels = np.arange(-0.05, 0.525+0.025, 0.025)
    levels = levels[levels!=0]
    _,_, c = plot_contour(df=df, fig=fig, ax=ax, var='potTPertubation', num_sample=1000, levels=levels, axis_off='on')

    sns.despine(fig, ax)
    ax.set_xlabel('$x [\si{m}]$')
    ax.set_ylabel('$z [\si{m}]$')
    ax.text(100,300, r'$t = \SI{700}{\s}$')

    fig.subplots_adjust(left=0.2, right=0.87, bottom=0.2)
    cax = fig.add_axes([0.25, 0.2, .6, .05])
    cb = fig.colorbar(mappable=c,cax=cax, orientation='horizontal')
    cb.locator = ticker.MaxNLocator(nbins=5)
    cb.update_ticks()

    fig.savefig(fig_path + f'{target}_cosine_bubble_contour_t700.pdf', transparent=True)


def main():
    set_plot_defaults()
    fig_path = 'output/'
    plot_cosine_bubble(fig_path=fig_path,
                       solution_path='/home/lukas/tmp/benchmark_results_cosine/solution_cartesian-10.vtu',
                       target='paper')

if __name__ == '__main__':
    main()
