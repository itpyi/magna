"""Fixed noise plan of a compiled task, exported for the CUDA sampler."""
from copy import deepcopy

import numpy as np

from . import _native


class CudaExperiment:
    """Packed Pauli-frame columns of every noise action of one compiled task.

    Creation checks the task's noise channels and output features. Decoder
    histories use an additional plan owned by the CUDA sampler.
    """

    def __init__(self, compiled):
        if compiled.options != compiled._execution_options:
            raise ValueError('compiled options changed; recompile before creating a CUDA experiment')
        self.compiled = compiled
        self._options_snapshot = deepcopy(compiled.options)
        self.handle, self.metadata = _native.make_cuda_experiment(compiled._execution.handle)
        self.words = int(self.metadata['words'])
        self.width = 2 * self.words
        self.actions = int(self.metadata['actions'])
        self.columns = np.frombuffer(self.metadata['columns'], dtype='<u8').reshape(self.actions, self.width)
        self.base_frame = np.frombuffer(self.metadata['base_frame'], dtype='<u8')

    def validate(self):
        """Reject option edits or native reconfiguration of this fixed plan.

        CUDA plans snapshot a completed compilation. Unlike the default CPU
        runner they do not apply mutable option dictionaries at run time.
        Recompile and construct a new experiment after changing options.
        """
        if self.compiled.options != self._options_snapshot:
            raise ValueError('compiled options changed; recompile and recreate the CUDA experiment')
        _native.cuda_noise_validate(self.handle)
