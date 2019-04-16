#!/usr/bin/env python3
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from plot_utils import *
from dataset_utils import *

def plot_cavity(fig_path, solution_path, target):
    # TODO: Use current solution
    df, time, order, dim, dx, n_points = read_solution(solution_path, order=3, size=1, limiting=True)
    print(time, order, dim, dx, n_points)

    sns.set_palette(sns.color_palette('Paired'))

    # Compare with reference data from High-Re Solutions 
    # for Incompressible Flow Using the Navier-Stokes Equations and a Multigrid Method
    ref_y, ref_v_x = np.array([
        1.0000, 1.00000,
        0.9766, 0.84123,
        0.9688, 0.78871,
        0.9609, 0.73722,
        0.9531, 0.68717,
        0.8516, 0.23151,
        0.7344, 0.00332,
        0.6172,-0.13641,
        0.5000,-0.20581,
        0.4531,-0.21090,
        0.2813,-0.15662,
        0.1719,-0.10150,
        0.1016,-0.06434,
        0.0703,-0.04775,
        0.0625,-0.04192,
        0.0547,-0.03717,
        0.0000,-0.00000
    ]).reshape(-1,2).T
    ref_y -= 0.5

    ref_x, ref_v_y = np.array([
        1.0000, 0.00000,
        0.9688,-0.05906,
        0.9609,-0.07391,
        0.9531,-0.08864,
        0.9453,-0.10313,
        0.9063,-0.16914,
        0.8594,-0.22445,
        0.8047,-0.24533,    
        0.5000, 0.05454,
        0.2344, 0.17527,
        0.2266, 0.17507,
        0.1563, 0.16077,
        0.0938, 0.12317,
        0.0781, 0.10890,
        0.0703, 0.10091,
        0.0625, 0.09233,    
        0.0000, 0.00000

    ]).reshape(-1,2).T
    ref_x -= 0.5


    df_cut = cut_dataset(df, 'y', dim=2, cut_at=[None, None])
    fig, ax = plt.subplots(figsize=figsize(0.5, target=target))
    ax.plot(df_cut['y'], df_cut['v_x'], label='$v_x(z)$')

    ax.scatter(ref_y, ref_v_x, marker='x',  s=50., lw=1)

    df_cut = cut_dataset(df, 'x', dim=2, cut_at=[None, None])
    ax.plot(df_cut['x'], df_cut['v_y'], label='$v_z(x)$')
    ax.scatter(ref_x, ref_v_y, marker='x', s=50., lw=1)

    ax.set_xlabel('$x, z$')
    ax.set_ylabel('$v_x(z), v_z(x)$')
    fig.legend(loc='upper left', bbox_to_anchor=(0.2,1))
    sns.despine(fig, ax)
    fig.subplots_adjust(left=0.2, bottom=0.2)

    xticks = [-0.5, -0.25,0,0.25,0.5]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)

    fig.savefig(fig_path + f'{target}_lid_driven_cavity.pdf', transparent=True)

    # Contour
    fig, ax = plt.subplots(figsize=quadratic_figsize(scale=0.5, target=target))
    start_points = np.array([[-0.48, -0.48],
                            [-0.46, 0.46],
                            [0.48, -0.48],
                            [0.45,-0.46]
                            ])
    plot_stream_lines(df=df, fig=fig, ax=ax, num_sample=1000, density=10, minlength=0.05, maxlength=20, interpolator='cubic',
                    modified_streamline=False, start_points=start_points, linewidth=0.7)

    n = 12
    x = np.array([-0.4, 0.0, 0.4])#np.linspace(-0.4, 0.4, n)
    y = x
    xx, yy = np.meshgrid(x,y)
    start_points = np.vstack((xx.ravel(), yy.ravel())).T

    plot_stream_lines(df=df, fig=fig, ax=ax, num_sample=1000, density=1, minlength=1.5, maxlength=10, interpolator='cubic',
                    modified_streamline=True, start_points=start_points, arrowsize=10e-100, linewidth=0.7)


    ax.set_aspect(adjustable='box-forced', aspect='equal')
    fig.subplots_adjust(left=0.2)

    ticks = [-0.5, -0.25,0,0.25,0.5]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)

    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks)
    # https://github.com/matplotlib/matplotlib/issues/8388 ?
    fig.savefig(fig_path + f'{target}_lid_driven_cavity_stream.pdf', transparent=True)

def main():
    set_plot_defaults()
    fig_path='output/'
    plot_cavity(fig_path=fig_path,
                solution_path='/home/lukas/Documents/MA-results/lid-driven-cavity/solution-3.vtu',
                target='paper')
        
if __name__ == '__main__':
    main()
    
