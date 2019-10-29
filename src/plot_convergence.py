#!/usr/bin/env python3
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

from plot_utils import *
from dataset_utils import *

from matplotlib import ticker

def plot_convergence(fig_path, solution_path, target):
    df = pd.read_csv(solution_path)
    orders = np.unique(df['order'])
    fig, ax = plt.subplots(figsize=figsize(1.0, target=target))
    colors = sns.color_palette("Paired")

    for i, order in enumerate(orders):
        cur_color = colors[i % len(colors)]
        df_sub = df.loc[df['order'] == order, :]
        x = df_sub['hmin']
        y = df_sub['error']**0.5 # TODO: Is this correct?
        ax.loglog(x, y, label="$P={}$".format(order), c=cur_color)
        ax.scatter(x, y, c=cur_color, marker='x', label=None)

        model = LinearRegression(fit_intercept=True)
        model.fit(np.log(x.values.reshape(-1, 1)), np.log(y))
    
        print("Emp. conv. {} for theoretical order {}".format(model.coef_[0], order+1))

        model.coef_[0] = order + 1
        y_pred = np.exp(model.predict(np.log(x.values.reshape(-1, 1))))
        ax.loglog(x, y_pred, linestyle='dashed', c=cur_color, label=None)

    ax.set_xticks(np.unique(df['hmin']))
    ax.set_xticklabels(["$10/3^{}$".format(i+1) for i in range(1,5)][::-1])
    fig.gca().invert_xaxis()
    fig.legend(loc='lower left', bbox_to_anchor=(0.095,0.16))
    ax.set_xlabel('Grid spacing')
    ax.set_ylabel('$L_2$ error')
    sns.despine(fig, ax)

    #fig.subplots_adjust(left=0.2, bottom=0.31, top=0.95, right=0.95)
    fig.subplots_adjust(left=0.1, bottom=0.175, right=0.98, top=0.97)
    fig.savefig(fig_path + '{target}_convergence.pdf'.format(target=target),
                transparent=True)
    

def main():
    set_plot_defaults()

    fig_path = 'output/'
    plot_convergence(fig_path=fig_path,
                     solution_path='/work_fast/krenz/src/cloudSuperComputing/tmp/errors.csv',
                     target='paper')

if __name__ == '__main__':
    main()
