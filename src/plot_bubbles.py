#!/usr/bin/env python3
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from plot_utils import *
from dataset_utils import *

def plot_bubbles(fig_path, solution_path, target):
    pass

def main():
    set_plot_defaults()
    fig_path='output/'
    plot_cavity(fig_path=fig_path,
                solution_path='/home/lukas/Documents/MA-results/lid-driven-cavity/solution-3.vtu',
                target='paper')
        
if __name__ == '__main__':
    main()
    
