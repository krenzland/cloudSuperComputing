#!/usr/bin/env python3
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from plot_utils import *
from dataset_utils import *

# Analytical solutions
def analytical_solution_p_tg(x,y,time,mu):
    C = 100/1.4
    decay = np.exp(-4 * mu * time)
    return C + 1/4 * (np.cos(2*x) + np.cos(2*y)) * decay

def analytical_solution_u_tg(x,y,time,mu):
    decay = np.exp(-2 * mu * time)
    return np.sin(x) * np.cos(y) * decay

def analytical_solution_v_tg(x,y,time,mu):
    decay = np.exp(-2 * mu * time)
    return -np.cos(x) * np.sin(y) * decay

def analytical_solution_taylor_green(x,y,z,time,mu,var):
    if var == 'p':
        return analytical_solution_p_tg(x,y,time,mu)
    elif var == 'u':
        return analytical_solution_u_tg(x,y,time,mu)
    else:
        return analytical_solution_v_tg(x,y,time,mu)

def plot_taylor_green(fig_path, solution_path, target, order=5):
    df, time, order, dim, dx, n_points = read_solution(solution_path, order=order, size=2*np.pi)
    print(time, order, dim, dx, n_points)

    scenario = 'taylor-green' if dim == 2 else 'abc'
    analytical_solution = analytical_solution_taylor_green if scenario == 'taylor-green' else analytical_solution_abc

    fig, ax = plt.subplots(figsize=figsize(scale=0.5, target=target))
    if dim == 2:
        coords = ['x', 'y']
        markers = ['o', 'x']   
    else:
        coords = ['x', 'y', 'z']
        markers = ['o', 'x', 'X']
        #coords = ['x']
        #markers = ['x']

    # Velocities
    for coord, marker in zip(coords, markers):
        coord_cut, p_approx_cut, p_ana_cut, vel_approx_cut, vel_ana_cut = eval_cut(df, coord, time, mu=0.01,
                                                                    analytical_solution=analytical_solution,
                                                                                dim=dim)
        ax.plot(coord_cut, vel_ana_cut, c='gray', linestyle='solid', zorder=-1, label=None)
        coord_label = 'z' if coord == 'y' else coord
        other = 'x' if coord == 'y' else 'z'
        #ax.scatter(coord_cut, vel_approx_cut, c='black', marker=marker, s=1., lw=6, facecolors='none',
        #          label=f'$v_{coord_label} ({other})$')
        ax.plot(coord_cut, vel_approx_cut, 'o', c='black', marker=marker,lw=6, mfc='none',
                label=f'$v_{coord_label} ({other})$')

    ax.set_xlabel('$x, z$')
    ax.set_ylabel('$v_x(z), v_z(x)$')
    ax.set_xticks([0, np.pi, 2*np.pi])
    ax.set_xticklabels([0, '$\pi$', '$2\pi$'])
    fig.subplots_adjust(left=0.2, bottom=0.2, right=0.85)
    fig.legend(loc='upper right', bbox_to_anchor=(1.01,1), handletextpad=10e-100, )
    sns.despine(fig,ax)
    fig.savefig(fig_path + f'{target}_taylor_green_vel.pdf', transparent=True)

    fig, ax = plt.subplots(figsize=figsize(scale=0.5, target=target))
    coord_cut, p_approx_cut, p_ana_cut, vel_approx_cut, vel_ana_cut = eval_cut(df, 'x', time, mu=0.01,
                                                                    analytical_solution=analytical_solution,
                                                                            dim=dim)
    ax.plot(coord_cut, p_ana_cut,  c='gray', linestyle='solid', zorder=-1)
    ax.scatter(coord_cut, p_approx_cut, c='black', marker=marker, label=coord, s=1., lw=6, facecolors='none',)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$p(z)$')
    ax.set_xticks([0, np.pi, 2*np.pi])
    ax.set_xticklabels([0, '$\pi$', '$2\pi$'])
    sns.despine(fig,ax)
    fig.subplots_adjust(left=0.2, bottom=0.2)
    fig.savefig(fig_path + f'{target}_taylor_green_pressure.pdf', transparent=True)

def main():
    set_plot_defaults()
    fig_path='output/'
    plot_taylor_green(fig_path=fig_path,
                      solution_path='/home/lukas/Documents/MA-results/taylor-green/solution-2.vtu',
                      target='paper')
        
if __name__ == '__main__':
    main()
    
