import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata


import streamplot_mod # fixes some issues with streamplot

def figsize(scale=1.0, target='thesis'):
    width_map = {
        'thesis': 418.25555,
        'beamer_169' : 398.3386,
        'beamer_43': 307.28987,
        'paper': 347.12354 #llncs
    }
    latex_width = width_map[target]
    fig_width = latex_width/72.27 # inches
    fig_height = fig_width * (np.sqrt(5)-1.0)/2.0
    return [fig_width*scale, fig_height*scale]
    

def quadratic_figsize(scale=1.0, target='thesis'):
    fs = figsize(scale=scale, target=target)
    size = (fs[0], fs[0])
    return size
    
def set_plot_defaults():
    sns.set_style('white')
    #sns.set_palette(sns.color_palette('viridis'))
    sns.set_palette(sns.color_palette('Paired'))

    params = {
        'font.size': 11,
        'font.family': 'serif',
        'font.serif': [],
        'axes.labelsize': 11,
        'legend.fontsize': 11,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.figsize': figsize(scale=1.0),
        'text.usetex': True,
        'pgf.texsystem': 'pdflatex',
        'text.latex.unicode': True
        }
    plt.rcParams.update(params)
    plt.rcParams['text.latex.preamble'].extend([
        r'\usepackage{amsmath}'
        r'\usepackage[separate-uncertainty]{siunitx}'],)

def plot_stream_lines(df, fig, ax, num_sample=500, density=2, minlength=0.01, maxlength=2, arrowsize=10e-100, start_points=None, interpolator='cubic',
                     modified_streamline=False, linewidth=None):
    x = df['x']
    y = df['y']
    u = df['v_x']
    v = df['v_y']
    
    xx = np.linspace(x.min(), x.max(), num_sample)
    yy = np.linspace(y.min(), y.max(), num_sample)

    X, Y = np.meshgrid(xx,yy)

    uu = griddata((x,y), u, (X,Y))
    vv = griddata((x,y), v, (X,Y))

    fs = figsize(scale=1.0)
    size = (fs[0], fs[0])

    #ax.axis('off')
    
    if modified_streamline:
        streamplot_mod.streamplot(axes=ax, x=xx, y=yy, u=uu, v=vv, color='black', minlength=minlength, maxlength=maxlength,
                                 density=density, arrowsize=arrowsize, integration_direction='both',
                                 start_points=start_points, linewidth=linewidth)
    else:
        ax.streamplot(x=xx, y=yy, u=uu, v=vv, start_points=start_points, color='black', minlength=minlength, maxlength=maxlength,
                      density=density, arrowsize=arrowsize, linewidth=linewidth)    
    #ax.set(adjustable='box-forced', aspect='equal')
    
    ax.set_xlim(left=xx.min(), right=xx.max())
    ax.set_ylim(bottom=yy.min(), top=yy.max())

    ax.set_xlabel('$x$')
    ax.set_ylabel('$z$') 
    
    sns.despine(fig, ax)
    
    return fig, ax

def plot_contour(df, fig, ax, var, num_sample=500, levels=5, axis_off='off', colored=True, 
                 linewidths=plt.rcParams['lines.linewidth'], coords=None):
    if not coords:
        coords = ['x', 'y']
    x = df[coords[0]]
    y = df[coords[1]]
    r = df[var]

    xx = np.linspace(x.min(), x.max(), num_sample)
    yy = np.linspace(y.min(), y.max(), num_sample)

    rr = griddata((x,y), r, (xx[None,:], yy[:,None]), method='cubic')

    fs = figsize(scale=1.0)
    size = (fs[0], fs[0])

    ax.axis(axis_off)
    
    ax.set_xlabel('$x$')
    ax.set_ylabel('$z$') 
    
    if colored:
        c = ax.contour(xx, yy, rr, levels=levels, cmap='viridis', linewidths=linewidths)
    else:
        c = ax.contour(xx, yy, rr, levels=levels, colors='black', linewidths=linewidths)
    
    return fig, ax, c

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
