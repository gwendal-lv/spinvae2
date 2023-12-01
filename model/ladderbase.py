"""
Base classes and methods for a ladder encoder or decoder.
"""
from abc import abstractmethod
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn


def parse_latent_extract_architecture(full_architecture: str):
    arch_args = full_architecture.split('_')
    base_arch_name = arch_args[0].lower()
    if base_arch_name != 'none':
        num_layers = int(arch_args[1].replace('l', ''))
    else:
        num_layers = 0
    arch_args_dict = {
        'k1x1': False, 'k3x3': False,  # Only these 2 kernel sizes are available
        'posenc': False,  # Positional encodings can be added inside some architectures
        'gated': False,  # (Self-)gating ("light attention") mechanisms can be added to some architectures
        'att': False,  # SAGAN-like self-attention
    }
    for arch_arg in arch_args[2:]:
        if arch_arg in arch_args_dict.keys():
            arch_args_dict[arch_arg] = True  # Authorized arguments
        else:
            raise ValueError("Unvalid encoder argument '{}' from architecture '{}'".format(arch_arg, full_architecture))
    return {'name': base_arch_name, 'n_layers': num_layers, 'args': arch_args_dict}


class LadderBase(nn.Module):
    def __init__(self, audio_arch, latent_arch, dim_z: int, z_CNN_shape: Optional[List[int]] = None):
        super().__init__()
        self.audio_arch, self.latent_arch, self.dim_z, self.z_CNN_shape = audio_arch, latent_arch, dim_z, z_CNN_shape

        if self.audio_arch['name'].startswith('specladder'):
            self.audio_is_a_CNN = True
        elif self.audio_arch['name'].startswith('ast'):  # audio spectrogram transformer
            self.audio_is_a_CNN = False
        else:
            raise AssertionError(f"audio arch name {self.audio_arch['name']} cannot be identified as a CNN or not")

    @property
    def audio_is_a_transformer(self):
        return not self.audio_is_a_CNN  # Only 2 possible architectures at the moment

    @staticmethod
    def flatten_CNN_output(z: torch.Tensor):
        """ Flattens all dimensions but the batch dimension. """
        assert len(z.shape) == 4, "This method should be used to flatten 2D feature maps"
        return z.flatten(start_dim=1)

    def unflatten_CNN_input(self, z_flat: torch.Tensor):
        """ Transforms a N x dimZ flattened latent vector into a N x C x H x W tensor
            where C, H, W were the latent CNN encoder output shape """
        assert self.audio_is_a_CNN and self.z_CNN_shape is not None
        return z_flat.view(*(z_flat.shape[0:1] + tuple(self.z_CNN_shape[1:])))

    @property
    def z_chunks_size(self):
        """ Returns the size of chunks of related latent coordinates, e.g. 1 for a pure 1D-vector z, or 4 for
        latent vectors obtained from flattened 2x2 feature maps. """
        if not self.audio_is_a_CNN:
            return 1
        else:
            assert len(self.z_CNN_shape) == 4, "z_CNN is expected to be 2d feature maps"
            return np.prod(self.z_CNN_shape[2:])


    @abstractmethod
    def get_custom_group_module(self, group_name: str) -> nn.Module:
        """
        Returns a module of parameters corresponding to a given group (e.g. 'audio', 'latent', ...).
        That means that even if a group is split into different Modules, they have to be stored in,
        e.g., a single ModuleList.
        """
        pass

    @property
    def _summary_col_names(self):
        if self.audio_is_a_CNN:
            return "input_size", "kernel_size", "output_size", "num_params", "mult_adds"
        else:
            return "input_size", "output_size", "num_params", "mult_adds"



class AudioPassthroughLatentModule(nn.Module):
    def __init__(self):
        """ Dummy module which only returns the audio hidden representation. Useful to test some
            architectures during pre-training. TODO maybe remove after debug. """
        super().__init__()
        self.dummy_conv = nn.Conv1d(1, 1, 1)  # To ensure that the latent optimizer has at least 1 registered parameter

    def forward(self, x_audio: torch.tensor,
                u_preset: Optional[torch.Tensor] = None, midi_notes: Optional[torch.Tensor] = None):
        return x_audio


