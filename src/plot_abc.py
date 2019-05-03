#!/usr/bin/env python3
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from plot_utils import *
from dataset_utils import *

# Analytical solutions
    
def analytical_solution_p_abc(x,y,z,time,mu):
    C = 100/1.4
    decay = np.exp(-2 * mu * time)
    return C - (np.cos(x) * np.sin(y) + np.sin(x) * np.cos(z) + np.sin(z) * np.cos(y)) * decay

def analytical_solution_u_abc(x,y,z,time,mu):
    decay = np.exp(-1 * mu * time)
    return (np.sin(z) + np.cos(y)) * decay

def analytical_solution_v_abc(x,y,z,time,mu):
    decay = np.exp(-1 * mu * time)
    return (np.sin(x) + np.cos(z)) * decay

def analytical_solution_w_abc(x,y,z,time,mu):
    decay = np.exp(-1 * mu * time)
    return (np.sin(y) + np.cos(x)) * decay

def analytical_solution_abc(x,y,z,time,mu,var):
    if var == 'p':
        return analytical_solution_p_abc(x,y,z,time,mu)
    elif var == 'u':
        return analytical_solution_u_abc(x,y,z,time,mu)
    elif var == 'v':
        return analytical_solution_v_abc(x,y,z,time,mu)
    else:
        return analytical_solution_w_abc(x,y,z,time,mu)

def plot_abc(fig_path, solution_path, target, order=2):
    df, time, order, dim, dx, n_points = read_solution(solution_path, order=order, dimensions=3)
    print(df.columns)
    time, order, dim, dx, n_points
    
    analytical_solution = analytical_solution_abc
    coord = 'x'
    var = 'p'

    fs = figsize(1.0, target=target)
    fs = fs[0], 0.5*fs[1]

    fig, axs = plt.subplots(ncols=3, figsize=fs, sharey=True)

    markers = ['o', 'x', 'X']
    marker = markers[1]

    for ax, coord in zip(axs, ['x', 'y', 'z']):
    #_, _, c = plot_contour(df_cut, fig, ax, var, levels=4, coords=others, num_sample=500,colored=False)
        coord_cut, p_approx_cut, p_ana_cut, vel_approx_cut, vel_ana_cut = eval_cut(df, coord, time, mu=0.01,
                                                                        analytical_solution=analytical_solution,
                                                                        dim=3, every_nth=4)
        ax.plot(coord_cut, p_ana_cut,  c='gray', linestyle='solid', zorder=-1)
        ax.scatter(coord_cut, p_approx_cut, c='black', marker=marker, label=coord, s=1., lw=6, facecolors='none',)
        if coord == 'x':
            ax.set_ylabel('$p$')

        ax.set_xlabel(str(coord))
        ax.set_xticks([-np.pi, 0, np.pi])
        ax.set_xticklabels(['$-\pi$', '$0$', '$\pi$'])
        sns.despine(fig,ax)
        fig.subplots_adjust(left=0.1, bottom=0.2, top=0.95, right=0.95)

    fig.savefig(fig_path + '{target}_abc_flow_pressure.pdf'.format(target=target), transparent=True)

    fig, axs = plt.subplots(ncols=3, figsize=figsize(0.5, target=target), sharey=True)

    for ax, coord, vel_var in zip(axs, ['x', 'y', 'z'], ['v', 'v', 'v']):
    #_, _, c = plot_contour(df_cut, fig, ax, var, levels=4, coords=others, num_sample=500,colored=False)
        coord_cut, p_approx_cut, p_ana_cut, vel_approx_cut, vel_ana_cut = eval_cut(df, coord, time, mu=0.01,
                                                                        analytical_solution=analytical_solution,
                                                                        dim=dim, every_nth=6, vel_var='v')
        ax.plot(coord_cut, vel_ana_cut,  c='gray', linestyle='solid', zorder=-1)
        ax.scatter(coord_cut, vel_approx_cut, c='black', marker=marker, label=coord, s=1., lw=6, facecolors='none',)
        if coord == 'x':
            ax.set_ylabel('$v_x$')

        ax.set_xlabel(str(coord))
        ax.set_xticks([-np.pi, 0, np.pi])
        ax.set_xticklabels(['$-\pi$', '$0$', '$\pi$'])
        sns.despine(fig,ax)
        fig.subplots_adjust(left=0.2, bottom=0.31, top=0.95, right=0.95)

    fig.savefig(fig_path + '{target}_abc_flow_velocity.pdf'.format(target=target), transparent=True)  

def main():
    set_plot_defaults()
    fig_path='output/'
    plot_abc(fig_path=fig_path,
                      solution_path='/work_fast/krenz/abc_flow/solution_cartesian-0.vtu',
                      target='paper')
        
if __name__ == '__main__':
    main()
    
