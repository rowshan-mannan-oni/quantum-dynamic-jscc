"""Per-epoch measurements of what a swapped-in module is doing while it trains.

`loss_log.txt` records the objective, which tells you *that* a run went wrong but
never *why*. These are the numbers that separate the explanations, and they only
exist while training is happening - gradients are gone the moment the run ends,
so a checkpoint cannot be interrogated for them afterwards.

Written to `metrics.csv` in the run folder, one row per epoch:

    rate_entropy        bits over the rates the policy chose. Falling to 0 means
                        it has stopped adapting and become a fixed-rate codec.
    <site>_saturation   fraction of the angle encoder with |tanh| > 0.99. This is
                        the failure that made the first policy run collapse: 82%
                        pinned at 20 dB, and the circuit returned a constant.
    <site>_measure      mean |<Z>| of the measurements. Near 1 means the state is
                        stuck at the poles and the readout carries little.
    <site>_grad_norm    gradient norm on the circuit angles. Decaying towards 0
                        while the loss is still high is the barren-plateau
                        signature that plan section 11 asks to be watched for.
    <site>_grad_var     variance of those gradients across angles. Vanishing
                        variance is the sharper barren-plateau test - a circuit
                        can hold a healthy norm while every angle receives
                        essentially the same uninformative push.

Nothing here touches the graph, the loss or any weight: hooks read forward
activations and the recorded gradients are read after the optimizer step. A run
with diagnostics is the same run without them, which is what lets these be on by
default rather than something to remember to enable.

Sites are discovered by structure rather than by name - any module carrying a
`.pre` linear is a swapped-in arm - so a new site is covered the moment it is
registered, with nothing to update here.
"""
import csv
import math
import os

import torch
import torch.nn as nn

SATURATED = 0.99          # |tanh| past this contributes nothing to the angle


class Diagnostics:
    """Accumulates over an epoch's training batches, writes one row per epoch."""

    def __init__(self, model, opt, run_dir):
        self.path = os.path.join(run_dir, 'metrics.csv')
        self.n_rates = opt.G_s + 1          # the policy picks 0..G_s selective groups
        self.enabled = True                 # switched off around validation
        self.sites = {}
        self.circuit_params = {}
        self._handles = []

        for net_name in ('netSE', 'netCE', 'netG', 'netP'):
            net = getattr(model, net_name, None)
            if net is None:
                continue
            base = net.module if isinstance(net, nn.DataParallel) else net
            for mod_path, module in base.named_modules():
                if not isinstance(getattr(module, 'pre', None), nn.Linear):
                    continue
                label = net_name[3:].lower()        # netCE -> ce
                self.sites[label] = module
                self._handles.append(
                    module.pre.register_forward_hook(self._saturation_hook(label)))
                circuit = getattr(module, 'circuit', None)
                if circuit is not None:
                    self._handles.append(
                        circuit.register_forward_hook(self._measure_hook(label)))
                    self.circuit_params[label] = [
                        p for n, p in module.named_parameters() if 'circuit.' in n]

        self.columns = ['epoch', 'stage', 'lr', 'train_loss', 'rate_entropy']
        for label in sorted(self.sites):
            self.columns.append(f'{label}_saturation')
            if label in self.circuit_params:
                self.columns += [f'{label}_measure', f'{label}_grad_norm',
                                 f'{label}_grad_var']
        self._reset()

        if self.sites:
            print('diagnostics: watching %s -> %s'
                  % (', '.join(sorted(self.sites)), self.path))

    # ---------------------------------------------------------------- hooks

    def _saturation_hook(self, label):
        def hook(_module, _inputs, output):
            if self.enabled:
                frac = (torch.tanh(output.detach()).abs() > SATURATED).float().mean()
                self._add(f'{label}_saturation', frac.item())
        return hook

    def _measure_hook(self, label):
        def hook(_module, _inputs, output):
            if self.enabled:
                self._add(f'{label}_measure', output.detach().abs().mean().item())
        return hook

    # ------------------------------------------------------------ recording

    def _reset(self):
        self._sums = {}
        self._counts = {}
        self.rate_hist = torch.zeros(self.n_rates, dtype=torch.long)

    def _add(self, key, value):
        self._sums[key] = self._sums.get(key, 0.0) + value
        self._counts[key] = self._counts.get(key, 0) + 1

    def on_batch(self, model):
        """Call after optimize_parameters, while .grad still holds this step."""
        if not self.enabled:
            return

        count = getattr(model, 'count', None)
        if count is not None:
            idx = count.detach().flatten().round().long().clamp(0, self.n_rates - 1)
            self.rate_hist += torch.bincount(idx.cpu(), minlength=self.n_rates)

        for label, params in self.circuit_params.items():
            grads = [p.grad.detach().flatten() for p in params if p.grad is not None]
            if not grads:
                continue
            g = torch.cat(grads)
            self._add(f'{label}_grad_norm', g.norm().item())
            self._add(f'{label}_grad_var', g.var(unbiased=False).item())

    def on_epoch_end(self, epoch, stage, lr, train_loss):
        row = {'epoch': epoch, 'stage': stage, 'lr': lr,
               'train_loss': round(train_loss, 6),
               'rate_entropy': round(self._entropy(), 4)}
        for key in self.columns[5:]:
            if self._counts.get(key):
                row[key] = '%.6g' % (self._sums[key] / self._counts[key])
            else:
                row[key] = ''

        new = not os.path.exists(self.path)
        with open(self.path, 'a', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=self.columns)
            if new:
                writer.writeheader()
            writer.writerow(row)

        self._reset()
        return row

    def _entropy(self):
        """Shannon entropy in bits of the rates chosen this epoch.

        Zero means every image got the same rate, which is a fixed-rate codec
        wearing an adaptive one's architecture. log2(n_rates) is the maximum.
        """
        total = self.rate_hist.sum().item()
        if total == 0:
            return 0.0
        p = self.rate_hist.double() / total
        p = p[p > 0]
        return float(-(p * p.log2()).sum())

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []
