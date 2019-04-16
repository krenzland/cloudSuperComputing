#!/usr/bin/env python3
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from plot_utils import *
from dataset_utils import *

from matplotlib import ticker


def plot_two_bubbles(fig_path, solution_path, target, order=4):
    df, time, order, dim, dx, n_points = read_solution(solution_path, order=order, size=1000)
    print(time, order, dim, dx, n_points)
    
    fig, ax = plt.subplots(figsize=quadratic_figsize(0.5, target=target))
    df['potTPertubation'] = df['potT'] - 300
    levels = np.arange(-0.05, 0.45+0.05, 0.05)
    levels = levels[levels!=0]
    plot_contour(df=df, fig=fig, ax=ax, var='potTPertubation', num_sample=100, levels=levels, axis_off='on', colored=False,
                linewidths=0.5)
    sns.despine(fig, ax)
    ax.set_xlabel('$x [\si{m}]$')
    ax.set_ylabel('$z [\si{m}]$')
    ax.text(100,100, r'$t = \SI{' + str(time) + r'}{\s}$')
    fig.subplots_adjust(left=0.2, bottom=0.15)
    fig.savefig(fig_path + f'{target}_two_bubbles_contour_t{time}.pdf', transparent=True)

def main():
    set_plot_defaults()
    fig_path = 'output/'
    plot_two_bubbles(fig_path=fig_path,
                       solution_path='/home/lukas/Documents/MA-results/AMR/solution_cartesian-10.vtu',
                       target='paper')

if __name__ == '__main__':
    main()
