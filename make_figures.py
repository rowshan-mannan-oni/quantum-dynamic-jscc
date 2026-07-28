"""Reproduce the paper's figures from a trained checkpoint.

test_dyna.py only prints scalars, so this script does the evaluation sweeps and
writes the figures plus the raw numbers behind them:

    fig4_rate_psnr_vs_snr.png   average rate (CPP) and PSNR against SNR
    fig5_attainable_psnr.png    PSNR at each fixed rate, adaptive points overlaid
    fig6_per_class.png          per-class rate and PSNR at one SNR
    reconstructions.png         sample reconstructions at several SNRs
    fig4_data.csv, fig5_data.csv, fig6_data.csv

Everything is driven through the repo's own model and options, so a config
change (including swapping in a quantum module) is picked up here automatically.

Example:
    python make_figures.py --gpu_ids 0 --select hard --lambda_reward 1.5e-3
"""
import csv
import os

import matplotlib
matplotlib.use('Agg')            # no display on Kaggle / headless machines
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from models import create_model
from options.test_options import TestOptions

CLASSES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
           'Dog', 'Frog', 'Horse', 'Ship', 'Truck']


class FigureOptions(TestOptions):
    """Test options plus the few knobs the figure sweeps need."""

    def initialize(self, parser):
        parser = TestOptions.initialize(self, parser)
        parser.add_argument('--results_dir', type=str, default='./figures',
                            help='where figures and csv files are written')
        parser.add_argument('--snr_list', type=str, default='0,2,4,6,8,10,12,14,16,18,20',
                            help='comma separated SNRs (dB) to sweep')
        parser.add_argument('--fig6_snr', type=int, default=10,
                            help='SNR (dB) used for the per-class figure')
        parser.add_argument('--eval_batch', type=int, default=250,
                            help='batch size for evaluation')
        parser.add_argument('--num_workers', type=int, default=2,
                            help='dataloader workers; safe here because this '
                                 'script has a __main__ guard')
        parser.add_argument('--dataroot', type=str, default='./data')
        return parser


def psnr_uint8(fake, real):
    """PSNR on 8-bit images, matching the computation in test_dyna.py."""
    f = ((fake.detach().cpu().float().numpy().transpose(0, 2, 3, 1) + 1) / 2 * 255).astype(np.uint8)
    r = ((real.detach().cpu().float().numpy().transpose(0, 2, 3, 1) + 1) / 2 * 255).astype(np.uint8)
    diff = np.mean((np.float64(f) - np.float64(r)) ** 2, (1, 2, 3))
    diff = np.clip(diff, 1e-10, None)
    return 10 * np.log10((255 ** 2) / diff)


@torch.no_grad()
def evaluate(model, loader, opt, snr, cpp_per_group, forced_active=None):
    """Run the model over the test set at one SNR, optionally at a fixed rate."""
    model.opt.SNR = snr                      # forward() reads this in eval mode
    model.forced_active = forced_active
    psnr, groups, labels, seen = [], [], [], 0
    for images, label in loader:
        if seen >= opt.num_test:
            break
        model.set_input(images)
        model.forward()
        psnr.append(psnr_uint8(model.fake, model.real_B))
        groups.append(model.hard_mask.sum(-1).cpu().numpy())
        labels.append(label.numpy())
        seen += images.shape[0]
    model.forced_active = None
    psnr = np.concatenate(psnr)[:opt.num_test]
    groups = np.concatenate(groups)[:opt.num_test]
    labels = np.concatenate(labels)[:opt.num_test]
    return {'psnr': psnr, 'groups': groups, 'cpp': groups * cpp_per_group, 'label': labels}


def write_csv(path, header, rows):
    with open(path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def figure_4(results, snrs, out_dir, lambda_reward):
    cpp = [r['cpp'].mean() for r in results]
    psnr = [r['psnr'].mean() for r in results]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(snrs, cpp, 'o-', color='tab:blue', label='Average rate (CPP)')
    ax1.set_xlabel('SNR (dB)')
    ax1.set_ylabel('Average rate (CPP)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(alpha=.3)
    ax2 = ax1.twinx()
    ax2.plot(snrs, psnr, 's--', color='tab:red', label='Average PSNR')
    ax2.set_ylabel('Average PSNR (dB)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    plt.title(f'Adaptive rate and PSNR vs SNR (alpha={lambda_reward})')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig4_rate_psnr_vs_snr.png'), dpi=120)
    plt.close(fig)

    write_csv(os.path.join(out_dir, 'fig4_data.csv'),
              ['snr_db', 'avg_cpp', 'avg_psnr_db', 'avg_groups'],
              [[s, f'{r["cpp"].mean():.4f}', f'{r["psnr"].mean():.4f}',
                f'{r["groups"].mean():.4f}'] for s, r in zip(snrs, results)])
    return cpp, psnr


def figure_5(model, loader, opt, snrs, cpp_per_group, adaptive_psnr, out_dir):
    curves, rows = {}, []
    for k in range(opt.G_s + 1):
        psnrs = []
        for snr in snrs:
            res = evaluate(model, loader, opt, snr, cpp_per_group, forced_active=k)
            psnrs.append(res['psnr'].mean())
            rows.append([k, f'{(opt.G_n + k) * cpp_per_group:.4f}', snr, f'{psnrs[-1]:.4f}'])
        curves[k] = psnrs
        print(f'  fixed rate: {k} selective groups '
              f'(CPP={(opt.G_n + k) * cpp_per_group:.3f}) done')

    fig = plt.figure(figsize=(7, 5))
    for k, psnrs in curves.items():
        plt.plot(snrs, psnrs, '-', label=f'Fixed CPP={(opt.G_n + k) * cpp_per_group:.3f}')
    plt.plot(snrs, adaptive_psnr, 'k*--', ms=10, label='Adaptive (proposed)')
    plt.xlabel('SNR (dB)')
    plt.ylabel('PSNR (dB)')
    plt.grid(alpha=.3)
    plt.legend()
    plt.title('Attainable PSNR: fixed rates vs adaptive')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig5_attainable_psnr.png'), dpi=120)
    plt.close(fig)

    write_csv(os.path.join(out_dir, 'fig5_data.csv'),
              ['forced_selective_groups', 'cpp', 'snr_db', 'avg_psnr_db'], rows)


def figure_6(result, snr, out_dir, lambda_reward):
    cls_cpp = np.array([result['cpp'][result['label'] == c].mean() for c in range(10)])
    cls_psnr = np.array([result['psnr'][result['label'] == c].mean() for c in range(10)])
    order = np.argsort(-cls_cpp)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(range(10), cls_cpp[order], color='tab:blue', alpha=.8)
    ax1.set_xticks(range(10))
    ax1.set_xticklabels([CLASSES[i] for i in order], rotation=45, ha='right')
    ax1.set_ylabel('Rate (CPP)', color='tab:blue')
    ax1.set_ylim(min(cls_cpp) * 0.95, max(cls_cpp) * 1.05)
    ax2 = ax1.twinx()
    ax2.plot(range(10), cls_psnr[order], 'r*-', ms=10)
    ax2.set_ylabel('PSNR (dB)', color='tab:red')
    plt.title(f'Per-class rate and PSNR (SNR={snr} dB, alpha={lambda_reward})')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig6_per_class.png'), dpi=120)
    plt.close(fig)

    write_csv(os.path.join(out_dir, 'fig6_data.csv'),
              ['class', 'avg_cpp', 'avg_psnr_db'],
              [[CLASSES[i], f'{cls_cpp[i]:.4f}', f'{cls_psnr[i]:.4f}'] for i in order])
    print(f'  per-class PSNR std: {cls_psnr.std():.3f} dB')


@torch.no_grad()
def reconstructions(model, testset, opt, cpp_per_group, out_dir, n_show=8, snrs=(0, 10, 20)):
    loader = torch.utils.data.DataLoader(testset, batch_size=n_show, shuffle=True)
    images, labels = next(iter(loader))
    to_img = lambda t: np.clip((t.cpu().float().numpy().transpose(1, 2, 0) + 1) / 2, 0, 1)

    fig, axes = plt.subplots(1 + len(snrs), n_show, figsize=(1.5 * n_show, 1.7 * (1 + len(snrs))))
    for j in range(n_show):
        axes[0, j].imshow(to_img(images[j]))
        axes[0, j].axis('off')
        axes[0, j].set_title(CLASSES[labels[j]], fontsize=8)
    axes[0, 0].text(-0.3, 0.5, 'Original', transform=axes[0, 0].transAxes,
                    rotation=90, va='center', ha='center', fontsize=9)

    for i, snr in enumerate(snrs):
        model.opt.SNR = snr
        model.set_input(images)
        model.forward()
        groups = model.hard_mask.sum(-1).cpu().numpy()
        for j in range(n_show):
            axes[i + 1, j].imshow(to_img(model.fake[j]))
            axes[i + 1, j].axis('off')
            axes[i + 1, j].set_title(f'CPP={groups[j] * cpp_per_group:.3f}', fontsize=7)
        axes[i + 1, 0].text(-0.3, 0.5, f'SNR {snr} dB', transform=axes[i + 1, 0].transAxes,
                            rotation=90, va='center', ha='center', fontsize=9)

    plt.suptitle('Original vs reconstructions (title = rate chosen for that image)')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'reconstructions.png'), dpi=120)
    plt.close(fig)


def main():
    opt = FigureOptions().parse()
    opt.name = (f'C{opt.C_channel}_L2_{opt.lambda_L2}_re_{opt.lambda_reward}_{opt.select}')
    os.makedirs(opt.results_dir, exist_ok=True)
    snrs = [int(s) for s in opt.snr_list.split(',')]

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root=opt.dataroot, train=False,
                                           download=True, transform=transform)
    loader = torch.utils.data.DataLoader(testset, batch_size=opt.eval_batch, shuffle=False,
                                         num_workers=opt.num_workers)

    model = create_model(opt)
    model.setup(opt)
    model.eval()

    # One group carries C_channel*h*w/(G_n+G_s) symbols; CPP counts channel uses
    # per source pixel, hence the factor 2 for the complex channel.
    spatial = 32 // (2 ** opt.n_downsample)
    group_len = opt.C_channel * spatial * spatial / (opt.G_n + opt.G_s)
    cpp_per_group = group_len / (2 * 32 * 32)

    print(f'\nSweeping SNR for the adaptive model ({opt.num_test} images each)...')
    results = []
    for snr in snrs:
        res = evaluate(model, loader, opt, snr, cpp_per_group)
        results.append(res)
        print(f'  SNR {snr:3d} dB | CPP {res["cpp"].mean():.3f} | '
              f'PSNR {res["psnr"].mean():.2f} dB | groups {res["groups"].mean():.2f}')

    _, adaptive_psnr = figure_4(results, snrs, opt.results_dir, opt.lambda_reward)

    print('\nSweeping each fixed rate...')
    figure_5(model, loader, opt, snrs, cpp_per_group, adaptive_psnr, opt.results_dir)

    print(f'\nPer-class breakdown at SNR {opt.fig6_snr} dB...')
    if opt.fig6_snr in snrs:
        fig6_res = results[snrs.index(opt.fig6_snr)]
    else:
        fig6_res = evaluate(model, loader, opt, opt.fig6_snr, cpp_per_group)
    figure_6(fig6_res, opt.fig6_snr, opt.results_dir, opt.lambda_reward)

    print('\nRendering sample reconstructions...')
    reconstructions(model, testset, opt, cpp_per_group, opt.results_dir)

    print(f'\nDone. Figures and csv files written to {opt.results_dir}')


if __name__ == '__main__':
    main()
