"""Check that TorchQuantum actually works here before wiring it into the model.

Run this first on any new machine (and on Kaggle):

    python -m quantum.smoke_test
    python -m quantum.smoke_test --bench      # also time the candidate sites

Importing TorchQuantum successfully is not enough. What matters is that a
batched circuit runs on the GPU and that gradients reach the circuit
parameters, because a silent failure there would look like a model that simply
does not learn.
"""
import argparse
import time

import torch

from quantum.layers import HybridVQC, VariationalCircuit

ITERS_PER_EPOCH = 390        # 50000 CIFAR-10 images / batch 128


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
    if args.bench:
        if device.type != 'cuda':
            print('\nSkipping benchmark: needs a GPU.')
        else:
            bench(device)


if __name__ == '__main__':
    main()
