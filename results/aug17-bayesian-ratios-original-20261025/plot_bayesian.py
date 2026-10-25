"""Replot the archived posterior and original ML comparison without refitting.

Run ``python plot_bayesian.py`` next to posterior.npz and source/. All posterior
summaries use every saved draw. A fixed random subset is used only for drawing
very large point clouds; it never changes histograms, intervals, or correlations.
The ML ellipse is an asymptotic normal confidence contour, not a posterior region.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

_CACHE = Path(tempfile.gettempdir()) / 'xemat-aug17-bayesian-plot-cache'
_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(_CACHE / 'matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(_CACHE))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np
from scipy.stats import chi2, norm

ROOT = Path(__file__).resolve().parent
POSTERIOR_COLOR = '#3C78A8'
ML_COLOR = '#C45A20'
MAP_COLOR = '#202020'
SCATTER_SEED = 2026102517
CHI2_95 = float(chi2.ppf(.95, 2))
ISOTOPES = ('Xe135', 'Xe133', 'Xe131m', 'Xe133m')
LABELS = (r'^{135}\mathrm{Xe}', r'^{133}\mathrm{Xe}',
          r'^{131\mathrm{m}}\mathrm{Xe}', r'^{133\mathrm{m}}\mathrm{Xe}')


def style(font_size):
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
        'font.size': font_size, 'mathtext.fontset': 'cm', 'text.usetex': False,
        'axes.labelsize': font_size + .5, 'axes.titlesize': font_size + .5,
        'xtick.labelsize': font_size - .4, 'ytick.labelsize': font_size - .4,
        'axes.linewidth': .7, 'xtick.major.width': .7, 'ytick.major.width': .7,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


def ellipse_points(center, covariance, n=1001):
    """Parametrize (a-a_hat)^T K^-1 (a-a_hat) = chi2(2,.95)."""
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if np.any(eigenvalues <= 0):
        raise ValueError('ML statistical covariance block is not positive definite')
    angle = np.linspace(0., 2*np.pi, n)
    circle = np.stack((np.cos(angle), np.sin(angle)))
    return center[:, None] + eigenvectors @ (np.sqrt(CHI2_95 * eigenvalues)[:, None] * circle)


def ratio_delta(ml, covariance, numerator, denominator=1):
    """Full four-concentration covariance, including the cross term."""
    gradient = np.zeros(4)
    gradient[numerator] = 1. / ml[denominator]
    gradient[denominator] = -ml[numerator] / ml[denominator]**2
    return dict(mean=float(ml[numerator]/ml[denominator]),
                sd=float(np.sqrt(gradient @ covariance @ gradient)),
                gradient=gradient.tolist())


def load_data(posterior_path, source_path):
    with np.load(posterior_path, allow_pickle=False) as saved:
        concentrations = np.asarray(saved['concentrations_mBq_m3'])
        ratios = np.asarray(saved['ratios'])
        if concentrations.ndim != 3 or concentrations.shape[-1] != 4:
            raise ValueError('Expected concentration axes: chain, draw, isotope=4')
        if ratios.shape != concentrations.shape[:2] + (2,):
            raise ValueError('Expected ratio axes: chain, draw, ratio=2')
        if not np.all(np.isfinite(concentrations)) or np.any(concentrations < 0):
            raise ValueError('Posterior concentrations must be finite and nonnegative')
        np.testing.assert_allclose(ratios[...,0], concentrations[...,2]/concentrations[...,1], rtol=1e-13)
        np.testing.assert_allclose(ratios[...,1], concentrations[...,3]/concentrations[...,1], rtol=1e-13)
        c = concentrations.reshape(-1,4)
        r = ratios.reshape(-1,2)
        shape = concentrations.shape[:2]
    with np.load(source_path, allow_pickle=False) as source:
        np.testing.assert_array_equal(source['isotope_order'][1:], ISOTOPES)
        ml = source['poisson_concentrations_mBq_m3'][1:].copy()
        map_estimate = source['poisson_nonnegative_concentrations_mBq_m3'][1:].copy()
        covariance = source['poisson_concentration_covariance_statistical'][1:,1:].copy()
    return c, r, shape, ml, map_estimate, covariance


def configure_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(color='#EBEBEB', linewidth=.6)
    ax.spines[['right','top']].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))


def concentration_label(index):
    return rf'$a_V({LABELS[index]})$ / mBq m$^{{-3}}$'


def joint_figure(c, ml, map_estimate, covariance, *, width, height, max_scatter):
    fig, axes = plt.subplots(1, 2, figsize=(width, height), sharex=True)
    fig.subplots_adjust(left=.105, right=.985, bottom=.28, top=.88, wspace=.32)
    indices = np.arange(len(c))
    if len(c) > max_scatter:
        indices = np.sort(np.random.default_rng(SCATTER_SEED).choice(len(c), max_scatter, replace=False))
    pair_rows = []
    x_bounds = []
    for panel, (ax, j) in enumerate(zip(axes, (2,3))):
        configure_axis(ax)
        pair = [1,j]
        block = covariance[np.ix_(pair,pair)]
        points = ellipse_points(ml[pair], block)
        ax.scatter(c[indices,1], c[indices,j], s=2.0, alpha=.16,
                   edgecolors='none', color=POSTERIOR_COLOR, zorder=2,
                   rasterized=False)
        ax.plot(points[0], points[1], color=ML_COLOR, lw=1.7, zorder=3)
        ax.plot(ml[1], ml[j], color=ML_COLOR, marker='+', markersize=9,
                markeredgewidth=1.7, linestyle='none', zorder=5)
        ax.plot(map_estimate[1], map_estimate[j], color=MAP_COLOR, marker='D',
                markersize=5.1, markerfacecolor='white', markeredgewidth=1.2,
                linestyle='none', zorder=4)
        posterior_r = float(np.corrcoef(c[:,1], c[:,j])[0,1])
        ml_r = float(block[0,1] / np.sqrt(block[0,0]*block[1,1]))
        ax.text(.04,.96, rf'$r_{{\rm post}}={posterior_r:.3f}$'+'\n'+rf'$r_{{\rm ML}}={ml_r:.3f}$',
                transform=ax.transAxes, va='top', ha='left', fontsize=9,
                bbox=dict(facecolor='white', edgecolor='none', alpha=.88, pad=2), zorder=6)
        ax.set_xlabel(concentration_label(1), labelpad=5)
        ax.set_ylabel(concentration_label(j), labelpad=5)
        ax.set_title(f"({'ab'[panel]}) " + rf'${LABELS[1]}$ and ${LABELS[j]}$', loc='left', pad=9)
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        ax.xaxis.set_major_formatter(formatter)
        lo = min(c[:,j].min(), points[1].min(), map_estimate[j])
        hi = max(c[:,j].max(), points[1].max(), map_estimate[j])
        pad = .06*(hi-lo)
        ax.set_ylim(lo-pad, hi+pad)
        x_bounds.extend([c[:,1].min(),c[:,1].max(),points[0].min(),points[0].max()])
        if lo < 0:
            ax.axhline(0, color='#8F8F8F', lw=.7, ls=':', zorder=1)
        pair_rows.append(dict(isotopes=[ISOTOPES[k] for k in pair],
            ml_center_mBq_m3=ml[pair].tolist(), map_center_mBq_m3=map_estimate[pair].tolist(),
            ml_stat_covariance_mBq2_m6=block.tolist(),
            posterior_correlation=posterior_r, ml_stat_correlation=ml_r,
            ellipse_chi2_df2_95=CHI2_95, ellipse_points_mBq_m3=points.T.tolist(),
            y_limits=ax.get_ylim()))
    dx = max(x_bounds)-min(x_bounds)
    axes[0].set_xlim(min(x_bounds)-.05*dx,max(x_bounds)+.05*dx)
    handles = [Line2D([],[],marker='o',linestyle='none',color=POSTERIOR_COLOR,
                       markersize=4,label='Posterior draws'),
               Line2D([],[],color=ML_COLOR,lw=1.7,label='95% ML normal-approximation ellipse'),
               Line2D([],[],color=ML_COLOR,marker='+',linestyle='none',markersize=8,
                       markeredgewidth=1.5,label='Free Poisson ML'),
               Line2D([],[],color=MAP_COLOR,marker='D',linestyle='none',markersize=5,
                       markerfacecolor='white',label='MAP (archived nonnegative ML)')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.53,.025),
               frameon=False,ncol=2,fontsize=9,handlelength=2.2,columnspacing=1.8,labelspacing=.7)
    metadata=dict(pairs=pair_rows,posterior_draw_count=len(c),
                  scatter_draw_count=len(indices),scatter_seed=SCATTER_SEED,
                  scatter_flat_indices=indices.tolist(),x_limits=axes[0].get_xlim(),
                  covariance_kind='statistical only; no reduced-deviance rescaling',
                  ellipse_interpretation='ML normal approximation, not a posterior credible region')
    return fig,metadata


def ratio_figure(r, ml, covariance, *, width, height, bins):
    fig,axes=plt.subplots(1,2,figsize=(width,height))
    fig.subplots_adjust(left=.105,right=.985,bottom=.28,top=.88,wspace=.32)
    rows=[]
    for panel,(ax,j) in enumerate(zip(axes,(2,3))):
        configure_axis(ax)
        draws=r[:,panel]
        lower,median,upper=np.quantile(draws,[.025,.5,.975])
        delta=ratio_delta(ml,covariance,j)
        # Show the continuous normal density on both sides of zero: no clipping
        # or renormalization to the physical half-line is applied.
        lo=min(float(draws.min()),delta['mean']-4.5*delta['sd'])
        hi=max(float(draws.max()),delta['mean']+4.5*delta['sd'])
        pad=.025*(hi-lo)
        grid=np.linspace(lo-pad,hi+pad,1601)
        edges=np.linspace(float(draws.min()),float(draws.max()),bins+1)
        heights,_,_=ax.hist(draws,bins=edges,density=True,color=POSTERIOR_COLOR,
                           alpha=.43,edgecolor='white',linewidth=.25,zorder=2)
        ax.axvspan(lower,upper,color=POSTERIOR_COLOR,alpha=.08,zorder=1)
        ax.plot(grid,norm.pdf(grid,delta['mean'],delta['sd']),color=ML_COLOR,lw=1.7,zorder=4)
        ax.axvline(median,color=POSTERIOR_COLOR,linestyle='-',lw=1.4,zorder=5)
        ax.axvline(lower,color=POSTERIOR_COLOR,linestyle=(0,(4,3)),lw=1.1,zorder=3)
        ax.axvline(upper,color=POSTERIOR_COLOR,linestyle=(0,(4,3)),lw=1.1,zorder=3)
        if lo < 0: ax.axvline(0,color='#999999',ls=':',lw=.7,zorder=1)
        ax.set_xlim(lo-pad,hi+pad)
        ax.set_ylim(bottom=0)
        ax.set_title(f"({'ab'[panel]}) " + rf'${LABELS[j]}/{LABELS[1]}$',loc='left',pad=9)
        ax.set_xlabel(rf'$r_{panel+1}=a_V({LABELS[j]})/a_V({LABELS[1]})$',labelpad=5)
        ax.set_ylabel('Probability density',labelpad=5)
        formatter=ScalarFormatter(useMathText=True,useOffset=False)
        formatter.set_powerlimits((-3,-3))
        ax.xaxis.set_major_formatter(formatter)
        rows.append(dict(ratio_name=f'r{panel+1}', numerator=ISOTOPES[j],denominator=ISOTOPES[1],
            posterior_mean=float(draws.mean()),posterior_median=float(median),
            posterior_sd=float(draws.std(ddof=1)),credible_interval_equal_tail_95=[float(lower),float(upper)],
            delta_method=delta,normal_probability_below_zero=float(norm.cdf(-delta['mean']/delta['sd'])),
            hist_bin_edges=edges.tolist(),hist_density=heights.tolist(),hist_integral=float(np.sum(heights*np.diff(edges))),
            normal_plot_range=[float(grid[0]),float(grid[-1])],
            normal_probability_in_plot_range=float(norm.cdf((grid[-1]-delta['mean'])/delta['sd'])-norm.cdf((grid[0]-delta['mean'])/delta['sd']))))
    handles=[Patch(facecolor=POSTERIOR_COLOR,alpha=.43,label='Posterior density'),
             Line2D([],[],color=ML_COLOR,lw=1.7,label='ML delta-method normal approximation'),
             Line2D([],[],color=POSTERIOR_COLOR,lw=1.4,label='Posterior median'),
             Line2D([],[],color=POSTERIOR_COLOR,lw=1.1,linestyle=(0,(4,3)),label='Equal-tailed 95% credible interval')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.53,.025),frameon=False,
               ncol=2,fontsize=9,handlelength=2.2,columnspacing=1.7,labelspacing=.7)
    return fig,dict(ratios=rows,posterior_draw_count=len(r),normal_tail_truncated_at_zero=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--posterior',type=Path,default=ROOT/'posterior.npz')
    parser.add_argument('--source',type=Path,default=ROOT/'source'/'measurement_arrays.npz')
    parser.add_argument('--output-dir',type=Path,default=ROOT)
    parser.add_argument('--width',type=float,default=7.4)
    parser.add_argument('--height',type=float,default=4.2)
    parser.add_argument('--font-size',type=float,default=10)
    parser.add_argument('--dpi',type=int,default=300)
    parser.add_argument('--bins',type=int,default=60)
    parser.add_argument('--max-scatter',type=int,default=20000)
    args=parser.parse_args()
    if args.bins<2 or args.max_scatter<1: parser.error('bins >=2 and max-scatter >=1 required')
    style(args.font_size)
    c,r,shape,ml,map_estimate,cov=load_data(args.posterior,args.source)
    joint,joint_meta=joint_figure(c,ml,map_estimate,cov,width=args.width,height=args.height,max_scatter=args.max_scatter)
    ratio,ratio_meta=ratio_figure(r,ml,cov,width=args.width,height=args.height,bins=args.bins)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for fig,name in ((joint,'bayesian_joint_concentrations'),(ratio,'bayesian_ratios')):
        fig.savefig(args.output_dir/(name+'.pdf'),metadata={'Title':name.replace('_',' '),'Creator':'XeMat plot_bayesian.py'})
        fig.savefig(args.output_dir/(name+'.png'),dpi=args.dpi)
        plt.close(fig)
        print('Saved',name+'.pdf','and',name+'.png',flush=True)
    metadata=dict(posterior_shape_chain_draw=list(shape),isotope_order=list(ISOTOPES),
        input_sha256={str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.posterior,args.source)},
        concentration_units='mBq m^-3',ratio_units='dimensionless',joint=joint_meta,ratios=ratio_meta,
        style=dict(font_family='DejaVu Serif',mathtext_fontset='cm',width_inches=args.width,
                   height_inches=args.height,font_size=args.font_size,png_dpi=args.dpi,vector_pdf=True))
    (args.output_dir/'plot_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
