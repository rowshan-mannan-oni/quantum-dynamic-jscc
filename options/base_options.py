import argparse
import os
import random

import numpy as np
import torch
import models


def parse_sites(value):
    """Split a comma separated --quantum_site / --matched_site value."""
    if not value or value == 'none':
        return []
    return [site.strip() for site in value.split(',') if site.strip()]


def run_name(opt):
    """Checkpoint folder name for a run.

    Every script rebuilds this from the options rather than taking a path, so it
    has to be derived in exactly one place. It previously appeared as a literal
    expression in four files, which is how a run once wrote to 'L2_1' and was
    then looked for under 'L2_1.0'.

    The classical name is unchanged, so existing checkpoints still resolve. Any
    setting that produces a different model extends it, so a quantum run cannot
    overwrite the classical baseline it is being compared against.
    """
    name = 'C%s_L2_%s_re_%s_%s' % (opt.C_channel, opt.lambda_L2,
                                   opt.lambda_reward, opt.select)

    quantum = parse_sites(getattr(opt, 'quantum_site', 'none'))
    matched = parse_sites(getattr(opt, 'matched_site', 'none'))
    if quantum:
        name += '_q%s-q%dl%d' % ('+'.join(quantum), opt.n_qubits, opt.vqc_layers)
    elif matched:
        name += '_m%s-q%d' % ('+'.join(matched), opt.n_qubits)

    seed = getattr(opt, 'seed', None)
    if seed is not None:
        name += '_s%d' % seed
    return name


def set_seed(seed):
    """Seed every RNG the training path draws from.

    Enough to make two arms see the same initialisation and data order. It does
    not force deterministic cuDNN kernels, which would cost throughput for
    run-to-run bit-exactness this comparison does not need.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """ 

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./Checkpoints', help='models are saved here')

        # model parameters
        parser.add_argument('--model', type=str, default='DynaAWGN', help='chooses which model to use. [DynaAWGN]')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--max_ngf', type=int, default=256, help='maximal # of gen filters in the last conv layer')
        parser.add_argument('--norm', type=str, default='batch', help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--n_downsample', type=int, default=2, help='number of downsample layers')
        parser.add_argument('--n_blocks', type=int, default=2, help='number of residual blocks')
        parser.add_argument('--C_channel', type=int, default=16, help='number of channels of the encoder output')
        parser.add_argument('--G_n', type=int, default=4, help='number of non-selective groups')
        parser.add_argument('--G_s', type=int, default=4, help='number of selective groups')
        parser.add_argument('--select', type=str, default='hard', help='using hard or soft mask [hard | soft]')
        parser.add_argument('--quantum_site', type=str, default='none', help="module(s) to replace with a quantum one, comma separated; 'none' keeps the model fully classical. See quantum/modules.py for the registered sites")
        parser.add_argument('--matched_site', type=str, default='none', help="module(s) to replace with a classical control of the same shape and parameter budget as the quantum module. Answers whether a difference came from the circuit or just from the smaller layer. Mutually exclusive with --quantum_site")
        parser.add_argument('--n_qubits', type=int, default=8, help='qubits in the variational circuit, and the bottleneck width of the matched control. Simulation cost grows as 2**n_qubits')
        parser.add_argument('--vqc_layers', type=int, default=2, help='depth of the variational circuit. Deeper is not safer: random deep circuits have exponentially vanishing gradients')
        parser.add_argument('--seed', type=int, default=None, help='random seed. Unset reproduces the original unseeded behaviour; set it for any run being compared against another, and vary it to separate a real effect from run-to-run noise')
        parser.add_argument('--SNR_MAX', type=int, default=20, help='maximum SNR')
        parser.add_argument('--SNR_MIN', type=int, default=0, help='minimum SNR')
        parser.add_argument('--lambda_reward', type=float, default=1.5e-3, help='weight for efficiency loss')
        parser.add_argument('--lambda_L2', type=float, default=1.0, help='weight for MSE loss')

        # dataset parameters
        parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
        parser.add_argument('--num_workers', type=int, default=0, help='dataloader worker processes. Must stay 0 on Windows: the scripts run at module scope with no __main__ guard, so spawned workers re-import them. Use 2-4 on Linux')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        
        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        #model_name = opt.model
        #model_option_setter = models.get_option_setter(model_name)
        #parser = model_option_setter(parser, self.isTrain)
        #opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        #dataset_name = opt.dataset_mode
        #dataset_option_setter = data.get_option_setter(dataset_name)
        #parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print the options, and return the text so a run can record it."""
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
        return message

    def save_options(self, opt):
        """Write the options next to the checkpoints as a record of the run.

        Called once the checkpoint folder exists. With several arms differing
        only by a flag, the folder name alone is a thin provenance record.
        """
        with open(os.path.join(opt.checkpoints_dir, opt.name, 'opt.txt'), 'w') as fh:
            fh.write(opt._options_text + '\n')

    def validate_sites(self, opt):
        """Reject site selections that cannot be honoured, before any training."""
        quantum = parse_sites(opt.quantum_site)
        matched = parse_sites(getattr(opt, 'matched_site', 'none'))

        if quantum and matched:
            raise SystemExit(
                '--quantum_site and --matched_site are mutually exclusive: the '
                'matched control exists to isolate the circuit, so a run cannot '
                'be both arms at once.')

        if not quantum:
            return

        try:
            import quantum as quantum_pkg
        except ImportError as exc:
            raise SystemExit('--quantum_site %s needs the quantum package and its '
                             'dependencies: %s\nSee quantum/README.md.'
                             % (opt.quantum_site, exc))

        unknown = [s for s in quantum if s not in quantum_pkg.available_sites()]
        if unknown:
            raise SystemExit('unknown quantum site(s) %s; available: %s'
                             % (unknown, quantum_pkg.available_sites()))

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        self.validate_sites(opt)

        # Derived in one place so every script agrees on where a run lives
        opt.name = run_name(opt)

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        if opt.seed is not None:
            set_seed(opt.seed)

        opt._options_text = self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
