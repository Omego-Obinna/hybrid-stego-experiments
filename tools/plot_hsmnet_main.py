"""Render the publication HSMNet figure from saved CSV tables only."""
from pathlib import Path
import argparse
import csv

ROOT=Path(__file__).resolve().parents[1]
INPUTS=ROOT/'experiments/studies/cpm_v2_hsmnet_matched_1200/records'

def rows(name):
    with (INPUTS/name).open(newline='') as f:return list(csv.DictReader(f))

METHODS=['cpm','suniward','mipod','wow','hill']
PAY=[.1,.2,.4]

def make_figure():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5,
                         'axes.labelsize': 10, 'axes.titlesize': 10.5,
                         'xtick.labelsize': 9, 'ytick.labelsize': 9,
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.edgecolor': '#82909c', 'axes.linewidth': .6,
                         'text.color': '#20303d', 'axes.labelcolor': '#20303d',
                         'xtick.color': '#45545f', 'ytick.color': '#45545f'})
    colors = dict(zip(METHODS, ['#172c45', '#007f86', '#b36b00', '#ae3e6c', '#587846']))
    markers = ['o', 's', '^']
    detection = {(x['method'], float(x['payload_bpp'])): x for x in rows('detectability.csv')}
    contrasts = {(x['baseline'], float(x['payload_bpp'])): x for x in rows('auc_paired_contrasts.csv')}
    fig = plt.figure(figsize=(7.15, 4.0), facecolor='white')
    ax = fig.add_axes([.085, .275, .345, .60])
    bl = fig.add_axes([.625, .275, .175, .60])
    br = fig.add_axes([.818, .275, .155, .60], sharey=bl)
    for a in [ax, bl, br]:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.set_axisbelow(True)
        a.tick_params(length=3, width=.6)
    ax.yaxis.grid(color='#dce3e8', linewidth=.6)
    ax.axhline(.5, color='#7a8994', linewidth=.8, linestyle=(0, (4, 3)))
    offsets = np.linspace(-.15, .15, 5)
    for mi, method in enumerate(METHODS):
        xx = np.arange(3) + offsets[mi]
        yy = [float(detection[method, p]['auc']) for p in PAY]
        ax.plot(xx, yy, color=colors[method], lw=1.0, alpha=.85, zorder=2)
        for x, y, p in zip(xx, yy, PAY):
            r = detection[method, p]
            lo, hi = float(r['auc_ci_low']), float(r['auc_ci_high'])
            # Exact endpoints: percentile intervals can exclude the estimate.
            ax.vlines(x, lo, hi, color=colors[method], lw=1.0, zorder=3)
            ax.hlines([lo, hi], x-.025, x+.025, color=colors[method], lw=.9)
            ax.plot(x, y, 'o', color=colors[method], ms=3.5, mec='white', mew=.35, zorder=4)
    ax.set(xlim=(-.36, 2.38), ylim=(.495, .600),
           xticks=range(3), xticklabels=['0.10', '0.20', '0.40'],
           yticks=[.50, .52, .54, .56, .58, .60],
           xlabel='Application payload (bpp)', ylabel='Held-out AUC')
    ax.text(-.19, 1.08, '(a)', transform=ax.transAxes, fontsize=11, fontweight='bold')
    ax.text(0, 1.08, 'AUC across payloads', transform=ax.transAxes, fontsize=10.5, fontweight='bold')
    bases = METHODS[1:]
    for bi, base in enumerate(bases):
        center = 3-bi
        for pi, p in enumerate(PAY):
            r = contrasts[base, p]
            d, lo, hi = [float(r[k]) for k in ['auc_cpm_minus_baseline', 'adjusted_ci_low', 'adjusted_ci_high']]
            y = center + [.22, 0, -.22][pi]
            for a in [bl, br]:
                a.hlines(y, lo, hi, color=colors[base], lw=1.15, zorder=3)
                a.plot([lo, hi], [y, y], '|', color=colors[base], ms=4, mew=.8, zorder=3)
                a.plot(d, y, marker=markers[pi], color=colors[base], ms=4.1, mec='white', mew=.35, zorder=4)
    for a in [bl, br]:
        a.set_ylim(-.55, 3.55)
        a.set_yticks([3, 2, 1, 0])
        a.grid(axis='y', color='#edf0f3', linewidth=.8)
    bl.set_xlim(-.082, -.009)
    br.set_xlim(-.0048, .0015)
    bl.set_yticklabels(['S-UNIWARD', 'MiPOD*', 'WOW', 'HILL'])
    br.tick_params(axis='y', left=False, labelleft=False)
    bl.spines['right'].set_visible(False)
    br.spines['left'].set_visible(False)
    bl.set_xticks([-.08, -.05, -.02])
    br.set_xticks([-.004, 0])
    fmt = FuncFormatter(lambda x, pos: '0' if abs(x) < 1e-10 else f'{x:.3f}'.rstrip('0'))
    bl.xaxis.set_major_formatter(fmt)
    br.xaxis.set_major_formatter(fmt)
    br.axvline(0, color='#7a8994', lw=.8, linestyle=(0, (4, 3)))
    # Clearly marked axis break keeps the small, nonzero contrasts legible.
    for a, x in [(bl, 1), (br, 0)]:
        for y in [0, 1]:
            a.plot([x-.017, x+.017], [y-.012, y+.012], transform=a.transAxes,
                   color='#82909c', lw=.7, clip_on=False)
    bl.text(-.30, 1.08, '(b)', transform=bl.transAxes, fontsize=11, fontweight='bold')
    bl.text(.04, 1.08, 'Adjusted paired differences', transform=bl.transAxes,
            fontsize=10.5, fontweight='bold')
    fig.text(.80, .172, 'CPM minus baseline AUC', ha='center', fontsize=10)
    fig.text(.80, .126, 'Negative favours CPM; axis break', ha='center', fontsize=8, color='#687986')
    labels = ['CPM', 'S-UNIWARD', 'MiPOD*', 'WOW', 'HILL']
    handles = [Line2D([0], [0], color=colors[m], marker='o', lw=1, ms=3.5, label=l)
               for m, l in zip(METHODS, labels)]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(.48, -.18),
              frameon=False, ncol=2, columnspacing=1.0, handlelength=1.4,
              fontsize=8.5, borderaxespad=0)
    ph = [Line2D([0], [0], color='#45545f', marker=mk, linestyle='none', ms=4,
                 label=f'{p:.2f} bpp') for p, mk in zip(PAY, markers)]
    fig.legend(handles=ph, loc='lower center', bbox_to_anchor=(.80, .042),
               frameon=False, ncol=3, columnspacing=.8, handletextpad=.3, fontsize=8.5)
    fig.text(.085, .018, '* MiPOD: recorded local variance approximation. Intervals condition on the fixed models.',
             fontsize=8, color='#687986')
    out = MAN / 'figures_revision/matched_hsmnet_comparison.pdf'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor='white', metadata={'Title': 'Matched HSMNet five-method image-carrier comparison', 'Author': 'Manuscript figure from checked saved tables'})
    fig.savefig(ARCHIVE / 'matched_hsmnet_comparison.png', dpi=220, facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,default=ROOT/'local_runs/figures')
    args=parser.parse_args()
    ARCHIVE=args.out_dir.resolve(); MAN=ARCHIVE
    ARCHIVE.mkdir(parents=True,exist_ok=True)
    make_figure()
    print(ARCHIVE/'figures_revision/matched_hsmnet_comparison.pdf')
