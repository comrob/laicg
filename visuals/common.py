import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
from typing import List
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

SUBPLOTS = List[List[plt.Axes]]


class FigProvider:
    def __init__(self):
        self._i = 0

    def __call__(self, *args, **kwargs) -> plt.Figure:
        self._i += 1
        return plt.figure(self._i)

    def show(self, *args, **kw):
        plt.show(*args, **kw)

    def close_all(self):
        plt.clf()
        plt.cla()
        plt.close('all')
        self._i = 0


def colors(total_n, cmap="gist_ncar"):
    return [mpl.colormaps[cmap](i/total_n) for i in range(total_n)]


def multiplot(figr, t, signal, labels: list=None, dist=3, override_yticklabels=None,
              title=None, y_label=None, x_label=None, global_color=None, pivot_norm_signal=None, distance_metric=False):
    if title is not None:
        figr.set_title(title, fontsize=18)
    if pivot_norm_signal is None:
        pivot_norm_signal = signal

    if not distance_metric:
        normed = list(map(lambda xp: (xp[0] - xp[1].mean())/(xp[1].std() + 0.00001), [(signal[:, s], pivot_norm_signal[:, s]) for s in range(signal.shape[1])]))
    else:
        normed = list(map(lambda xp: xp[0]/(xp[1].std() + 0.00001), [(signal[:, s], pivot_norm_signal[:, s]) for s in range(signal.shape[1])]))

    if labels is None:
        labels = [str(i+1) for i in range(signal.shape[1])]

    ticks = []
    for i in range(signal.shape[1]):
        if global_color is None:
            figr.plot(t, normed[i] + i * (1 + dist))
        else:
            figr.plot(t, normed[i] + i * (1 + dist), color=global_color, alpha=0.9)

        m = np.mean(i * (1 + dist))
        figr.axhline(y=m, color='k', linestyle='--', alpha=0.5)
        ticks.append(m)


    figr.spines['right'].set_visible(False)
    figr.spines['top'].set_visible(False)
    figr.set_yticks(ticks)
    if override_yticklabels is None:
        override_yticklabels = [i for i in labels]
    figr.set_yticklabels(override_yticklabels)
    if y_label is not None:
        figr.set_ylabel(y_label, fontsize=16)
    if x_label is not None:
        figr.set_xlabel(x_label, fontsize=16)


def plot_embeddings_evolution(figs: SUBPLOTS, embeddings, embeddings_t=None, **kwargs):
    num, dim, granularity = embeddings.shape
    for i in range(dim):
        for j in range(granularity):
            fgr = figs[i][j]
            if embeddings_t is None:
                fgr.plot(embeddings[:, i, j], **kwargs)
            else:
                fgr.plot(embeddings_t[:, i, j], embeddings[:, i, j], **kwargs)


def subplots_styling(figs: SUBPLOTS, inner_x_ticks_off=True, legend_on=True,
                     xlabels: List[str] = None, ylabels: List[str] = None,
                     titles: List[str] = None
                     ):
    if inner_x_ticks_off:
        for i in range(len(figs)-1):
            for fig in figs[i]:
                fig.set_xticklabels([])

    if legend_on:
        figs[-1][-1].legend()

    if xlabels is not None:
        for i, fig in enumerate(figs[-1]):
            fig.set_xlabel(xlabels[i])

    if titles is not None:
        for i, fig in enumerate(figs[0]):
            fig.set_title(titles[i])

    if ylabels is not None:
        for i, row in enumerate(figs):
            row[0].set_ylabel(ylabels[i])


def subplots(fig: plt.Figure, rows, cols, **kwargs) -> SUBPLOTS:
    _figs = fig.subplots(rows, cols, **kwargs)
    if rows == 1 and cols > 1:
        return [_figs]
    if rows > 1 and cols == 1:
        return [[_f] for _f in _figs]
    if rows == 1 and cols == 1:
        return [[_figs]]
    return _figs


def standalone_legend(fig, labels_args, ax=None, prop={}):
    pls = []
    if ax is None:
        ax = fig.subplot(111, aspect='equal')
        plt.subplots_adjust(left=.99, bottom=0.11, right=1., top=0.88, wspace=0.200, hspace=0.200)

    for lab_arg in labels_args:
        pl = ax.plot([0.1, 0.2], [0.1, 0.2], **lab_arg)
        pls.append(pl)


    leg = ax.legend(ncol=len(labels_args), prop=prop)
    leg.get_frame().set_linewidth(0.0)

    for pl in pls:
        pl[0].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])


def confidence_ellipse(ax, x, y, n_std=3.0, edgecolor = 'red', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor='none', edgecolor=edgecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)