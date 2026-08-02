"""Check that TorchQuantum actually works here before wiring it into the model.

Run this first on any new machine (and on Kaggle):

    python -m quantum.smoke_test
    python -m quantum.smoke_test --bench      # also time the candidate sites

Importing TorchQuantum successfully is not enough. What matters is that a
batched circuit runs on the GPU and that gradients reach the circuit
parameters, because a silent failure there would look like a model that simply
does not learn.

Two levels of check run by default. The first exercises `HybridVQC` on its own.
The second builds every registered site through the same `build_swappable` hook
training uses, at the shapes and batch sizes a real run hits, for all three
arms - so it catches a site that is registered but mis-wired, an arm whose
shapes do not line up, and a parameter budget that has drifted away from its
control. Skip it with --no-sites if `models/` is unavailable.
"""
import argparse
import time
from types import SimpleNamespace

import torch

from quantum.layers import HybridVQC, VariationalCircuit

ITERS_PER_EPOCH = 390        # 50000 CIFAR-10 images / batch 128
BATCH = 128                  # --batch_size default, so the checks see real sizes


def check(device):
    print(f'torch {torch.__version__} | device {device}')
    import torchquantum
    print(f'torchquantum {getattr(torchquantum, "__version__", "unknown")}\n')

    n_qubits, n_layers, bsz = 8, 2, 256
    block = HybridVQC(in_features=64, out_features=16,
                      n_qubits=n_qubits, n_layers=n_layers).to(device)

    x = torch.randn(bsz, 64, device=device)
    out = block(x)
    loss = out.pow(2).mean()
    loss.backward()

    circuit_params = list(block.circuit.parameters())
    with_grad = [p for p in circuit_params if p.grad is not None and p.grad.abs().sum() > 0]
    grad_norm = torch.stack([p.grad.norm() for p in circuit_params
                             if p.grad is not None]).norm().item()

    print(f'output shape          {tuple(out.shape)}  (expected ({bsz}, 16))')
    print(f'circuit parameters    {block.n_quantum_parameters()}')
    print(f'receiving gradient    {len(with_grad)}/{len(circuit_params)}')
    print(f'gradient norm         {grad_norm:.6g}')
    print(f'pre-linear has grad   {block.pre.weight.grad is not None}')

    assert out.shape == (bsz, 16), 'unexpected output shape'
    assert grad_norm > 0, 'no gradient reached the circuit parameters'

    expectations = block.circuit(torch.randn(bsz, n_qubits, device=device))
    assert expectations.abs().max() <= 1.0 + 1e-5, 'measurements outside [-1, 1]'
    print(f'measurement range     [{expectations.min():.3f}, {expectations.max():.3f}]')
    print('\nSMOKE TEST PASSED')


def build_arm(site, arm, device, n_qubits, vqc_layers):
    """Build the sub-network containing <site>, for one arm, through the real hook.

    Goes through `define_*` rather than constructing the module directly, so
    what is checked is the path training uses: the hook, the wrapper it applies,
    the site factory, and `init_net`'s re-initialisation of the linear layers on
    the way past.

    Returns the network, the swapped module inside it, a callable that builds
    this site's inputs and runs the forward pass, and the output shape expected
    of it. The site owns its own inputs because they differ - the decoder site
    is fed a 16-channel latent where the others take a 256-channel feature map.

    The swapped module is returned separately because it is what the parameter
    budget is about. Counting the whole network instead would bury a 2.2k module
    inside the channel encoder's 2.9M and report every arm as identical, which
    is precisely the drift the count is supposed to catch.

    Add a branch when adding a site. The unknown-site raise at the end is
    deliberate: a site that registers itself but has no entry here would
    otherwise be silently untested.
    """
    from models import networks

    opt = SimpleNamespace(quantum_site='none', matched_site='none',
                          n_qubits=n_qubits, vqc_layers=vqc_layers)
    if arm == 'quantum':
        opt.quantum_site = site
    elif arm == 'matched':
        opt.matched_site = site

    gpu_ids = [device.index or 0] if device.type == 'cuda' else []
    common = dict(ngf=64, max_ngf=256, n_downsample=2, gpu_ids=gpu_ids, opt=opt)

    def unwrap(net):
        return net.module if isinstance(net, torch.nn.DataParallel) else net

    def inputs(channels):
        return (torch.randn(BATCH, channels, 8, 8, device=device),
                torch.rand(BATCH, 1, device=device) * 20)

    if site == 'policy':
        net = networks.define_dynaP(N_output=5, **common)
        # (hard_mask, soft_mask, logits) - the logits are what the site produces
        return (net, unwrap(net).model_gate,
                lambda: net(*inputs(256), 5)[2], (BATCH, 5))
    if site == 'projection':
        net = networks.define_CE(C_channel=16, norm='batch', **common)
        return net, unwrap(net).projection, lambda: net(*inputs(256)), (BATCH, 16, 8, 8)
    if site == 'decoder':
        net = networks.define_dynaG(output_nc=3, C_channel=16, n_blocks=2,
                                    norm='batch', **common)
        # Runs the whole decoder, so the swapped module is checked in place
        # rather than in isolation: a shape it gets wrong surfaces here.
        return net, unwrap(net).mask_conv, lambda: net(*inputs(16)), (BATCH, 3, 32, 32)

    raise KeyError(f'site {site!r} is registered but has no entry in build_arm, '
                   f'so it would go untested')


def check_wrapper(device):
    """PerPosition must be exactly a 1x1 convolution when given a linear map.

    The wrapper folds (N,C,H,W) into (N*H*W, C) and back. Getting the permute
    and reshape the wrong way round transposes positions against channels
    without changing any shape, so nothing raises - the model simply trains on
    scrambled spatial structure and reports a worse number. Pinning it against
    Conv2d(1x1), which is the same map by definition, is what turns that into a
    failure instead of a quiet result.
    """
    import torch.nn as nn
    from models.networks import PerPosition

    c_in, c_out = 12, 5
    linear = nn.Linear(c_in, c_out).to(device)
    conv = nn.Conv2d(c_in, c_out, kernel_size=1).to(device)
    with torch.no_grad():
        conv.weight.copy_(linear.weight.view(c_out, c_in, 1, 1))
        conv.bias.copy_(linear.bias)

    x = torch.randn(4, c_in, 8, 6, device=device)      # H != W catches a transpose
    got, want = PerPosition(linear)(x), conv(x)

    assert got.shape == want.shape, f'PerPosition gave {got.shape}, 1x1 conv {want.shape}'
    assert torch.allclose(got, want, atol=1e-5), \
        f'PerPosition disagrees with a 1x1 conv by {(got - want).abs().max():.3g}'
    print(f'\nPerPosition == Conv2d(1x1)   max diff {(got - want).abs().max():.2g}'
          f'   on {tuple(x.shape)}')


def check_sites(device, n_qubits, vqc_layers):
    """Every registered site, every arm: shapes line up and gradients arrive."""
    import quantum
    from models.networks import SWAPPABLE_SITES

    print(f'\n{"=" * 78}\nsites, through the real build_swappable hook '
          f'(q={n_qubits}, L={vqc_layers}, batch={BATCH})\n{"=" * 78}')

    registered = quantum.available_sites()
    missing = [s for s in registered if s not in SWAPPABLE_SITES]
    assert not missing, f'registered with no model hook: {missing}'

    for site in registered:
        print(f'\n{site}')
        print(f'  {"arm":<10}{"module params":>14}{"circuit":>9}  '
              f'{"output":<18}{"circuit grad":>14}')
        counts = {}
        for arm in ('classical', 'matched', 'quantum'):
            net, module, forward, expected = build_arm(site, arm, device,
                                                       n_qubits, vqc_layers)
            out = forward()
            out.pow(2).mean().backward()

            assert tuple(out.shape) == expected, \
                f'{site}/{arm} produced {tuple(out.shape)}, expected {expected}'

            total = sum(p.numel() for p in module.parameters())
            counts[arm] = total

            circuit = [p for n, p in module.named_parameters() if 'circuit.' in n]
            if arm == 'quantum':
                assert circuit, f'{site}/quantum built no circuit parameters'
                grads = [p.grad for p in circuit if p.grad is not None]
                assert len(grads) == len(circuit), \
                    f'{site}/quantum: only {len(grads)}/{len(circuit)} angles got a gradient'
                norm = torch.stack([g.norm() for g in grads]).norm().item()
                assert norm > 0, \
                    f'{site}/quantum: gradient reached the angles but is exactly zero'
                grad = f'{norm:.3g}'
            else:
                grad = '-'

            print(f'  {arm:<10}{total:>14,}{len(circuit):>9}  '
                  f'{str(tuple(out.shape)):<18}{grad:>14}')

        # The control exists to answer "you just shrank the layer", which it
        # only does if it is actually the same size as the circuit it stands in
        # for. 10% is loose enough for the arms' differing internals and tight
        # enough that a real drift fails here rather than in review.
        q, m = counts['quantum'], counts['matched']
        skew = abs(q - m) / max(q, m)
        assert skew < 0.10, (f'{site}: matched control is {skew:.0%} away from the '
                             f'quantum arm ({m:,} vs {q:,}); it is no longer a control')
        print(f'  matched/quantum differ by {skew:.2%}, '
              f'both {counts["classical"] / q:.1f}x smaller than classical')

    print('\nSITE CHECKS PASSED')


def bench(device):
    """Time a forward+backward at the batch size each site would use."""
    def timed(bsz, n_qubits, n_layers, iters=5):
        circuit = VariationalCircuit(n_qubits, n_layers).to(device)
        x = torch.randn(bsz, n_qubits, device=device, requires_grad=True)
        for _ in range(2):                                   # warm up CUDA
            circuit(x).sum().backward()
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iters):
            circuit(x).sum().backward()
        torch.cuda.synchronize()
        return (time.time() - start) / iters * 1000

    print(f'\n{"site":<32}{"batch":>8}{"qubits":>8}{"layers":>8}'
          f'{"ms/step":>10}{"+min/epoch":>12}')
    print('-' * 78)
    for label, bsz, n_qubits, n_layers in [
            ('policy / modulation', 128, 4, 2),
            ('policy / modulation', 128, 8, 2),
            ('projection (64 positions)', 8192, 4, 2),
            ('projection (64 positions)', 8192, 6, 2),
            ('projection (64 positions)', 8192, 8, 2),
            ('projection (64 positions)', 8192, 8, 4)]:
        ms = timed(bsz, n_qubits, n_layers)
        print(f'{label:<32}{bsz:>8}{n_qubits:>8}{n_layers:>8}'
              f'{ms:>10.1f}{ms * ITERS_PER_EPOCH / 60000:>12.1f}')
    print('\nCompare against the classical baseline measured with train_dyna.py.')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bench', action='store_true', help='also time each candidate site')
    parser.add_argument('--no-sites', dest='sites', action='store_false',
                        help='skip the per-site checks, which need the models package')
    parser.add_argument('--n_qubits', type=int, default=8, help='circuit width to check the sites at')
    parser.add_argument('--vqc_layers', type=int, default=2, help='circuit depth to check the sites at')
    parser.add_argument('--cpu', action='store_true', help='force CPU')
    parser.add_argument('--gpu', type=int, default=0, help='which GPU to use, matching --gpu_ids in the training scripts')
    args = parser.parse_args()

    if args.cpu or not torch.cuda.is_available():
        device = torch.device('cpu')
        print('WARNING: running on CPU. Simulation is far slower there.\n')
    else:
        count = torch.cuda.device_count()
        if not 0 <= args.gpu < count:
            raise SystemExit('--gpu %d does not exist: this machine has %d GPU(s), so valid '
                             'ids are 0..%d' % (args.gpu, count, count - 1))
        device = torch.device('cuda', args.gpu)
        # bench() calls torch.cuda.synchronize(), which acts on the current
        # device, so timings would be taken against the wrong GPU without this.
        torch.cuda.set_device(device)

    check(device)
    if args.sites:
        check_wrapper(device)
        check_sites(device, args.n_qubits, args.vqc_layers)
    if args.bench:
        if device.type != 'cuda':
            print('\nSkipping benchmark: needs a GPU.')
        else:
            bench(device)


if __name__ == '__main__':
    main()
