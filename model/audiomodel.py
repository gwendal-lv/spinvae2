from typing import Tuple, Optional, Sequence

import torch
import torch.nn as nn



def parse_audio_architecture(full_architecture: str):
    """ Parses an argument used to describe the encoder and decoder audio architectures (e.g. speccnn8l_big_res) """
    # Decompose architecture to retrieve number of layers, options, ...
    arch_args = full_architecture.split('_')
    base_arch_name = arch_args[0]  # type: str
    del arch_args[0]
    # Default block args values (won't all be set by all architectures)
    n_layers_per_block, hidden_dim = None, None
    if base_arch_name.startswith('speccnn'):
        n_blocks = int(base_arch_name.replace('speccnn', '').replace('l', ''))
        n_layers_per_block = 1
    elif base_arch_name.startswith('sprescnn'):
        n_blocks = None
    elif base_arch_name.startswith('specladder'):  # Is also a CNN
        blocks_args = base_arch_name.replace('specladder', '').split('x')
        n_blocks, n_layers_per_block = int(blocks_args[0]), int(blocks_args[1])
    elif base_arch_name.startswith('ast'):  # e.g. ast6h256 indicate ast, 6 layers, hidden token dimension 256
        blocks_args = base_arch_name.replace('ast', '').split('h')
        n_blocks, hidden_dim = int(blocks_args[0]), int(blocks_args[1])
    else:
        raise AssertionError("Base architecture not available for given arch '{}'".format(base_arch_name))
    # Check arch args
    arch_args_dict = {
        'adain': False, 'big': False, 'bigger': False, 'res': False, 'att': False,
        'depsep5x5': False, 'swish': False, 'wn': False,
        'p16x16': False, 'p256x1': False,  # audio transformer patch size (16x16 is the default)
    }
    for arch_arg in arch_args:
        if arch_arg in arch_args_dict.keys():
            arch_args_dict[arch_arg] = True  # Authorized arguments
        else:
            raise ValueError("Unvalid encoder argument '{}' from architecture '{}'".format(arch_arg, full_architecture))
    return {'name': base_arch_name,
            'n_blocks': n_blocks, 'n_layers_per_block': n_layers_per_block, 'hidden_dim': hidden_dim,
            'args': arch_args_dict}



class AudioBase(nn.Module):
    def __init__(self, dim_z: int, arch, original_audio_shape: Sequence[int]):
        """ Base class for audio encoders and decoders, based on CNNs or ViTs. """
        super().__init__()
        self.dim_z, self.arch, self.original_audio_shape = dim_z, arch, tuple(original_audio_shape)

        # ViT-specific options
        self.patches: Optional[AudioBase.ImagePatchesProperties] = None
        if self.is_transformer_based:
            # Default patch_size is 16x16
            patch_size = (16, 16)  # H (frequency bins), W (time steps)
            if self.arch['args']['p16x16']:
                patch_size = (16, 16)
            elif self.arch['args']['p256x1']:
                patch_size = (256, 1)
            try:
                if self.original_audio_shape[2:] == (257, 251):
                    if patch_size == (16, 16):
                        # We'll discard the highest freq bin, and pad input on the right (note release) side
                        self.patches = AudioBase.ImagePatchesProperties(patch_size, 16, 16)
                    elif patch_size == (256, 1):  # vertical patches = spectrogram time frames
                        # We'll discard the highest freq bin, don't need to pad on the right
                        self.patches = AudioBase.ImagePatchesProperties(patch_size, 1, 251)
                    else:
                        raise AssertionError()
                else:
                    raise AssertionError()
            except AssertionError:
                raise NotImplementedError(f"patch_size {patch_size} with audio shape "
                                          f"{self.original_audio_shape[2:]} is currently unavailable")

    class ImagePatchesProperties:
        def __init__(self, patch_size: Tuple[int, int], n_rows: int, n_cols: int):
            self.H = patch_size[0]  # Height (in pixels) of a single patch
            self.W = patch_size[1]  # Width (in pixels) of a single patch
            self.n_rows, self.n_cols = n_rows, n_cols
            self.count = self.n_rows * self.n_cols

        @property
        def size(self):
            return (self.H, self.W)

    @property
    def is_transformer_based(self):
        return self.arch['name'].startswith('ast')

    @property
    def hidden_dim(self):
        if self.is_transformer_based:
            return self.arch['hidden_dim']
        else:
            raise ValueError("hidden_dim is undefined for non-Transformer-based models.")

    @staticmethod
    def get_1d_positional_embedding(embedding_dim: int, seq_len: int, n_extra_tokens=0, max_len=10000.0):
        """ Returns 1D positional embeddings (from the original Transformer paper). Extra tokens correspond to
         negative indices, such that sequences always have the same P.E. even if the number of extra tokens changes.

         :returns: 2D tensor (shape seq_len x embedding_dim) of positional encodings
         """
        embed = torch.unsqueeze(torch.arange(-n_extra_tokens, seq_len, dtype=torch.int).float(), dim=1)
        embed = embed.repeat(1, embedding_dim)  # Repeat copies the data (required here)
        for i in range(embedding_dim//2):
            omega_inverse = max_len ** (2.0 * i / embedding_dim)
            embed[:, 2 * i] = torch.sin(embed[:, 2 * i] / omega_inverse)
            embed[:, 2 * i + 1] = torch.cos(embed[:, 2 * i + 1] / omega_inverse)
        return embed

    def _compute_conv_next_H_W(self, input_H_W: Tuple[int, int], strided_conv_block: nn.Module):
        with torch.no_grad():
            x = torch.zeros((1, strided_conv_block.in_channels) + input_H_W)  # Tuple concat
            return tuple(strided_conv_block(x).shape[2:])

    def _get_conv_act(self):
        if self.arch['args']['swish']:
            return nn.SiLU()
        else:
            return nn.LeakyReLU(0.1)

    def _get_conv_norm(self):
        if self.arch['args']['wn']:
            return 'wn'
        else:  # Default (no arg): Batch Norm
            return 'bn'

