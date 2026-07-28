"""Convert a Kaggle-notebook checkpoint into the layout expected by this repo.

The notebook (Dynamic_JSCC_Kaggle.ipynb) saves every sub-network in a single
file as {'SE': ..., 'CE': ..., 'G': ..., 'P': ...}, while base_model.py expects
one file per network, named '<epoch>_net_<NAME>.pth', inside
<checkpoints_dir>/<name>/.

Each state dict is loaded into the corresponding network from models/networks.py
with strict=True before anything is written, so a silent architecture mismatch
cannot slip through.

Example:
    python convert_kaggle_weights.py --src Weight/final.pth --lambda_reward 1.5e-3
"""
import argparse
import os
import torch

from models import networks
from options.base_options import run_name

NET_NAMES = ['SE', 'CE', 'G', 'P']


def build_networks(args):
    """Instantiate the repo's networks on CPU (no DataParallel wrapper)."""
    return {
        'SE': networks.define_SE(input_nc=args.input_nc, ngf=args.ngf, max_ngf=args.max_ngf,
                                 n_downsample=args.n_downsample, norm=args.norm, gpu_ids=[]),
        'CE': networks.define_CE(ngf=args.ngf, max_ngf=args.max_ngf, n_downsample=args.n_downsample,
                                 C_channel=args.C_channel, norm=args.norm, gpu_ids=[]),
        'G': networks.define_dynaG(output_nc=args.output_nc, ngf=args.ngf, max_ngf=args.max_ngf,
                                   n_downsample=args.n_downsample, C_channel=args.C_channel,
                                   n_blocks=args.n_blocks, norm=args.norm, gpu_ids=[]),
        'P': networks.define_dynaP(ngf=args.ngf, max_ngf=args.max_ngf,
                                   n_downsample=args.n_downsample, gpu_ids=[],
                                   N_output=args.G_s + 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--src', type=str, default='Weight/final.pth',
                        help='checkpoint saved by the Kaggle notebook')
    parser.add_argument('--checkpoints_dir', type=str, default='./Checkpoints')
    parser.add_argument('--epoch', type=str, default='latest',
                        help="tag used in the filename; test_dyna.py loads 'latest' by default")
    parser.add_argument('--force', action='store_true',
                        help='overwrite existing checkpoint files')

    # These must match the run that produced the weights; they also determine the
    # folder name, which test_dyna.py rebuilds from the same fields.
    parser.add_argument('--C_channel', type=int, default=16)
    parser.add_argument('--lambda_L2', type=float, default=1.0)
    parser.add_argument('--lambda_reward', type=float, default=1.5e-3)
    parser.add_argument('--select', type=str, default='hard')
    parser.add_argument('--G_s', type=int, default=4)
    parser.add_argument('--input_nc', type=int, default=3)
    parser.add_argument('--output_nc', type=int, default=3)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--max_ngf', type=int, default=256)
    parser.add_argument('--n_downsample', type=int, default=2)
    parser.add_argument('--n_blocks', type=int, default=2)
    parser.add_argument('--norm', type=str, default='batch')
    args = parser.parse_args()

    ckpt = torch.load(args.src, map_location='cpu')
    if not isinstance(ckpt, dict) or not set(NET_NAMES).issubset(ckpt.keys()):
        raise SystemExit(f'{args.src} is not a notebook checkpoint; '
                         f'expected a dict containing {NET_NAMES}')

    # Derived by the same function the other scripts use, so a converted
    # checkpoint always lands where they will look for it.
    out_dir = os.path.join(args.checkpoints_dir, run_name(args))
    os.makedirs(out_dir, exist_ok=True)

    targets = {n: os.path.join(out_dir, f'{args.epoch}_net_{n}.pth') for n in NET_NAMES}
    existing = [p for p in targets.values() if os.path.exists(p)]
    if existing and not args.force:
        raise SystemExit('Refusing to overwrite existing checkpoints (pass --force):\n  ' +
                         '\n  '.join(existing))

    nets = build_networks(args)

    # Validate everything before writing anything.
    for net_name in NET_NAMES:
        nets[net_name].load_state_dict(ckpt[net_name], strict=True)
    print('\nAll four state dicts loaded with strict=True - architectures match.')

    for net_name in NET_NAMES:
        torch.save(nets[net_name].state_dict(), targets[net_name])
        n_param = sum(p.numel() for p in nets[net_name].parameters())
        print(f'  {net_name:2s} -> {targets[net_name]}  ({n_param / 1e6:.6f} M params)')

    print(f'\nDone. Run inference with:\n'
          f'  python test_dyna.py --gpu_ids 0 --select {args.select} '
          f'--lambda_reward {args.lambda_reward} --SNR 10 --num_test 1000 --num_test_channel 1')


if __name__ == '__main__':
    main()