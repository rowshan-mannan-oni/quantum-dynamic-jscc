"""Overlay the figure sweeps of two or more arms on one pair of axes.

make_figures.py evaluates a single checkpoint, so comparing arms means running it
once per arm and combining the csv files it leaves behind. This reads those and
writes:

    comparison_rate_psnr.png   rate vs SNR and PSNR vs SNR, one panel each
    comparison_data.csv        the numbers behind both panels

Rate and PSNR are kept in separate panels rather than sharing a twin axis: two
scales on one frame make the crossing point of the curves an artefact of the
scaling, and the whole question here is where the arms actually separate.

Example:
    python compare_arms.py --runs "Classical=./figures/classical,Quantum policy=./figures/quantum"
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')            # no display on Kaggle / headless machines
import matplotlib.pyplot as plt

# Categorical slots 1 and 2, and the ink colours for text/grid. One hue per arm,
# assigned in order and never cycled; a derived curve reuses its parent's hue as a
# dashed line rather than taking a slot of its own.
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
INK = '#20201d'
MUTED = '#6b6a63'


def read_fig4(run_dir):
    """The adaptive sweep: SNR, rate and PSNR for one arm."""
    path = os.path.join(run_dir, 'fig4_data.csv')
    with open(path, newline='') as fh:
        rows = list(csv.DictReader(fh))
    return {'snr': [float(r['snr_db']) for r in rows],
            'cpp': [float(r['avg_cpp']) for r in rows],
            'psnr': [float(r['avg_psnr_db']) for r in rows]}


def read_fig5_at(run_dir, cpp, tol=1e-3):
    """One fixed-rate curve from an arm's rate sweep, or None if it has no such rate."""
    path = os.path.join(run_dir, 'fig5_data.csv')
    if not os.path.exists(path):
        return None
    with open(path, newline='') as fh:
        rows = [r for r in csv.DictReader(fh) if abs(float(r['cpp']) - cpp) < tol]
    if not rows:
        return None
    rows.sort(key=lambda r: float(r['snr_db']))
    return {'snr': [float(r['snr_db']) for r in rows],
            'psnr': [float(r['avg_psnr_db']) for r in rows]}


def collapsed_rate(arm, tol=1e-4):
    """The single rate an arm always picks, or None if it actually adapts.

    A policy that has collapsed to a constant rate is not comparable to an
    adaptive one on the adaptive curve, so the caller uses this to pull in the
    matching fixed-rate reference instead.
    """
    lo, hi = min(arm['cpp']), max(arm['cpp'])
    return lo if hi - lo < tol else None


def label_last(ax, x, y, text, color, dy=0):
    """Direct label at the right end of a curve, so identity is never colour alone.

    dy nudges a label clear of one that would otherwise sit on top of it, which
    happens whenever two curves converge at the right-hand edge.
    """
    ax.annotate(text, xy=(x[-1], y[-1]), xytext=(4, dy), textcoords='offset points',
                va='center', ha='left', fontsize=8, color=MUTED)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--runs', type=str, required=True,
                        help='comma separated Label=results_dir pairs, in plotting order')
    parser.add_argument('--results_dir', type=str, default='./figures',
                        help='where the comparison figure and csv are written')
    parser.add_argument('--fixed_ref', type=float, default=None,
                        help='overlay the first arm at this fixed rate as a reference. '
                             'Default: auto, using the rate of any arm whose policy '
                             'has collapsed to a constant')
    parser.add_argument('--title', type=str, default='Classical vs quantum policy')
    opt = parser.parse_args()

    arms = []
    for pair in opt.runs.split(','):
        label, _, run_dir = pair.partition('=')
        arms.append((label.strip(), read_fig4(run_dir.strip())))
    os.makedirs(opt.results_dir, exist_ok=True)

    # A collapsed arm transmits at one rate whatever the channel does, so its
    # adaptive curve is really a fixed-rate curve. Pulling the first arm's curve
    # at that same rate is what makes the PSNR panel a like-for-like comparison.
    ref_cpp = opt.fixed_ref
    if ref_cpp is None:
        for _, arm in arms[1:]:
            ref_cpp = collapsed_rate(arm)
            if ref_cpp is not None:
                break
    ref = None
    if ref_cpp is not None:
        base_dir = opt.runs.split(',')[0].partition('=')[2].strip()
        ref = read_fig5_at(base_dir, ref_cpp)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    for (label, arm), color in zip(arms, SERIES):
        ax1.plot(arm['snr'], arm['cpp'], '-o', color=color, lw=2, ms=7, label=label)
        label_last(ax1, arm['snr'], arm['cpp'], label, color)
    ax1.set_xlabel('SNR (dB)', color=INK)
    ax1.set_ylabel('Average rate (CPP)', color=INK)
    ax1.set_title('Rate chosen vs channel quality', color=INK, fontsize=11)
    ax1.grid(alpha=.25)

    for (label, arm), color in zip(arms, SERIES):
        ax2.plot(arm['snr'], arm['psnr'], '-o', color=color, lw=2, ms=7, label=label)
        label_last(ax2, arm['snr'], arm['psnr'], label, color)
    if ref is not None:
        ref_label = '%s @ fixed CPP=%.3f' % (arms[0][0], ref_cpp)
        ax2.plot(ref['snr'], ref['psnr'], '--s', color=SERIES[0], lw=2, ms=6,
                 alpha=.75, label=ref_label)
        label_last(ax2, ref['snr'], ref['psnr'], 'fixed CPP=%.3f' % ref_cpp,
                   SERIES[0], dy=-11)
    ax2.set_xlabel('SNR (dB)', color=INK)
    ax2.set_ylabel('Average PSNR (dB)', color=INK)
    ax2.set_title('Reconstruction quality vs channel quality', color=INK, fontsize=11)
    ax2.grid(alpha=.25)
    ax2.legend(fontsize=8, frameon=False, loc='upper left')

    for ax in (ax1, ax2):
        ax.tick_params(colors=MUTED)
        for spine in ax.spines.values():
            spine.set_color('#d9d8d0')
        ax.margins(x=.12)

    fig.suptitle(opt.title, fontsize=13, color=INK)
    fig.tight_layout()
    out_png = os.path.join(opt.results_dir, 'comparison_rate_psnr.png')
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

    header = ['snr_db']
    for label, _ in arms:
        header += ['%s_cpp' % label, '%s_psnr_db' % label]
    if ref is not None:
        header.append('%s_psnr_db_at_cpp_%.3f' % (arms[0][0], ref_cpp))
    rows = []
    for i, snr in enumerate(arms[0][1]['snr']):
        row = [snr]
        for _, arm in arms:
            row += ['%.4f' % arm['cpp'][i], '%.4f' % arm['psnr'][i]]
        if ref is not None:
            row.append('%.4f' % ref['psnr'][i])
        rows.append(row)
    out_csv = os.path.join(opt.results_dir, 'comparison_data.csv')
    with open(out_csv, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)

    print('wrote %s\nwrote %s' % (out_png, out_csv))
    for label, arm in arms:
        rate = collapsed_rate(arm)
        note = 'constant rate %.3f — policy is not adapting' % rate if rate is not None \
            else 'rate %.3f -> %.3f across the sweep' % (arm['cpp'][0], arm['cpp'][-1])
        print('  %-16s %s' % (label + ':', note))


if __name__ == '__main__':
    main()