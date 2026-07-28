"""Quantum extensions for Dynamic JSCC.

Everything quantum lives in this package. The classical pipeline in `models/`
is shared by both arms of the comparison, which is what makes the comparison
fair: the only difference between a classical and a hybrid run is which module
one factory returns.

A "site" is a place in the pipeline where a classical module can be swapped for
a quantum one. Sites register themselves here, and `--quantum_site` selects one
at run time; `none` (the default) leaves the model entirely classical.

To remove the quantum work entirely, delete this package. The only thing left
behind is the `build_site` call at each hook, which falls back to the classical
module whenever the site is not enabled.
"""
from quantum import layers                                   # noqa: F401
from quantum.layers import HybridVQC, VariationalCircuit     # noqa: F401

# name -> factory(opt, **kwargs) -> nn.Module
_SITES = {}


def register_site(name):
    """Register a factory that builds the quantum module for <name>."""
    def decorator(factory):
        if name in _SITES:
            raise ValueError(f'quantum site {name!r} is already registered')
        _SITES[name] = factory
        return factory
    return decorator


def available_sites():
    """Site names that can be passed to --quantum_site (besides 'none')."""
    return sorted(_SITES)


def build_site(name, opt, **kwargs):
    """Build the quantum module registered under <name>.

    Raises KeyError with the list of valid names rather than silently falling
    back, so a typo in --quantum_site cannot quietly train a classical model
    and be written up as a quantum result.
    """
    if name not in _SITES:
        raise KeyError(f'unknown quantum site {name!r}; '
                       f'available: {["none"] + available_sites()}')
    return _SITES[name](opt, **kwargs)


def is_enabled(opt):
    """True when the options ask for a quantum module somewhere."""
    return getattr(opt, 'quantum_site', 'none') != 'none'


# Importing the module registers the sites it defines. Kept last so the
# registry helpers above already exist.
from quantum import modules                                  # noqa: E402,F401
