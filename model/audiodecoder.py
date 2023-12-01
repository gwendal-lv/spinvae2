import warnings
from typing import Tuple, Sequence, Optional

import numpy as np
import torch
import torch.nn as nn

import model.audiomodel
from model.convlayer import ConvBlock2D, UpsamplingResBlock, SelfAttentionConv2D, ResBlockBase, ResConvBlock2d

import utils.probability


class AudioDecoder(model.audiomodel.AudioBase):
    def __init__(self, dim_z: int, arch, n_cells: int, output_shape: Sequence[int], proba_distribution: str):
        """
        Multi-cell (multi-stage) audio decoder, based on a CNN or a ViT (spectrogram input).

        This class does not provide a forward(...) method.
        Its cells are to be used by the LadderDecoder module which owns an instance of this class.
        """
        super().__init__(dim_z, arch, output_shape)

        N = output_shape[0]  # expected batch size
        self.cells = list()
        self.cells_input_shapes = list()

        args = self.arch['args']
        n_blocks = self.arch['n_blocks']
        if args['depsep5x5'] and self.arch['n_layers_per_block'] == 1:
            raise AssertionError("Depth-separable convolutions require at least 2 layers per res-block.")

        # 1) Output probability distribution helper class (required to know the required num of output channels)
        if proba_distribution.lower() == "gaussian_unitvariance":
            self.proba_distribution = utils.probability.GaussianUnitVariance(reduction='none')
        else:
            raise ValueError("Unavailable audio probability distribution {}".format(proba_distribution))

        # 2) Build the audio decoder itself
        if self.arch['name'].startswith('specladder'):  # CNN decoder
            if args['adain']:
                raise NotImplementedError()
            if args['att'] and self.arch['n_layers_per_block'] < 2:
                raise ValueError("'_att' conv arg will add a residual self-attention layer and requires >= 2 layers")
            if args['att'] and args['depsep5x5'] and self.arch['n_layers_per_block'] < 3:
                raise ValueError("'_att' and '_depsep5x5' conv args (both provided) require >= 3 layers")

            if 1 <= n_cells <= (n_blocks - 2):
                cells_first_block = list(range(0, n_cells))
            else:
                raise NotImplementedError(f"Cannot build encoder with {n_cells} audio cells")

            for i_blk in range(n_blocks):
                residuals_path = nn.Sequential()
                blk_in_ch = 2 ** (10 - i_blk)  # number of input channels
                blk_out_ch = 2 ** (9 - i_blk)  # number of ch decreases after each strided Tconv (at block's end)
                if args['big']:
                    blk_in_ch, blk_out_ch = blk_in_ch * 2, blk_out_ch * 2
                min_ch = 1 if not args['bigger'] else 128
                max_ch = 512 if not args['bigger'] else 1024
                blk_in_ch, blk_out_ch = np.clip([blk_in_ch, blk_out_ch], min_ch, max_ch)
                blk_hid_ch = blk_out_ch  # base number of internal (hidden) block channels
                is_last_block = (i_blk == (n_blocks - 1))
                if is_last_block:
                    blk_out_ch = self.proba_distribution.num_parameters

                # First block of cell? Create new cell, store number of input channels
                if i_blk in cells_first_block:
                    self.cells.append(nn.Sequential())
                    self.cells_input_shapes.append([N, blk_in_ch, -1, -1])

                if not is_last_block:  # Hidden res blocks can contain multiple conv blocks
                    # Output padding (1, 1): with 4x4 kernels in hidden layers, this ensures same hidden feature maps
                    #  HxW in the encoder and this decoder. Output will be a bit too big but can be cropped.
                    conv = nn.ConvTranspose2d(blk_in_ch, blk_hid_ch, (4, 4), 2, 2, (1, 1))
                    residuals_path.add_module('stridedT', ConvBlock2D(
                        conv, self._get_conv_act(), self._get_conv_norm(), 'nac'))
                    # Usual convs are applied after upsampling (less parameters)
                    for j in range(self.arch['n_layers_per_block'] - 1):
                        if j == 0 and args['att'] and (2 <= i_blk <= 4):  # attention costs a lot of GPU RAM
                            self_att = SelfAttentionConv2D(blk_hid_ch, position_encoding=True)
                            residuals_path.add_module('self_att', ConvBlock2D(
                                self_att, None, self._get_conv_norm(), 'nc'))
                        elif args['depsep5x5']:  # Structure close to NVAE (NeurIPS 2020)
                            depth_sep_ch = blk_hid_ch * 2  # 3x channels expansion costs a lot of GPU RAM
                            sub_conv_block = nn.Sequential()
                            conv = nn.Conv2d(blk_hid_ch, depth_sep_ch, (1, 1))
                            sub_conv_block.add_module('more_ch_' + str(j), ConvBlock2D(
                                conv, None, self._get_conv_norm(), 'nc'))
                            conv = nn.Conv2d(depth_sep_ch, depth_sep_ch, (5, 5), 1, 2, groups=depth_sep_ch)
                            sub_conv_block.add_module('depthsep_' + str(j), ConvBlock2D(
                                conv, self._get_conv_act(), self._get_conv_norm(), 'nac'))
                            conv = nn.Conv2d(depth_sep_ch, blk_hid_ch, (1, 1))
                            sub_conv_block.add_module('less_ch_' + str(j), ConvBlock2D(
                                conv, self._get_conv_act(), self._get_conv_norm(), 'nac'))
                            # As this module contains 3 convs, we use it as residuals not to impair gradient backprop
                            residuals_path.add_module('depsep5x5_' + str(j), ResBlockBase(sub_conv_block))
                        else:
                            conv = nn.Conv2d(blk_hid_ch, blk_hid_ch, (3, 3), 1, 1)
                            conv_block = ConvBlock2D(conv, self._get_conv_act(), self._get_conv_norm(), 'nac')
                            residuals_path.add_module('res_conv', ResConvBlock2d(conv_block))
                else:  # last block: conv only, wider 5x5 kernel
                    residuals_path.add_module('stridedT', ConvBlock2D(
                        nn.ConvTranspose2d(blk_in_ch, blk_out_ch, (5, 5), 2, 2, 0), None, None, 'c'))

                if args['res'] and not is_last_block:
                    current_block = UpsamplingResBlock(residuals_path)
                else:
                    current_block = residuals_path  # No skip-connection
                self.cells[-1].add_module('blk{}'.format(i_blk), current_block)

        elif self.arch['name'].startswith('ast'):  # audio spectrogram transformer

            assert n_cells == 1  # TODO multiple cells
            self.cells.append(nn.Sequential())

            # TODO custom layers:

            # TODO several methods should be available

            # TODO transformer input: we can use z as memory while some learned positional embeddings are used as input
            #     or TODO use z to build input tokens (and use a Transformer encoder, no cross-attention anymore)
            for i_blk in range(n_blocks):
                # Last layer uses a final linear layer (final transformer output from a residual-add operation)
                #    embeddings could be bigger than the number of pixels (we would then need a reduction of E dim)
                audio_tfm_layer = AudioTransformerDecoderLayer(
                    self.dim_z,
                    self.patches.count,
                    is_first_layer=(i_blk == 0), is_last_layer=(i_blk == n_blocks - 1),
                    output_dim=(self.patches.H * self.patches.W),  # used by the last layer only
                    d_model=self.hidden_dim,
                    nhead=self.hidden_dim // 64,  # hidden_dim is a proper multiple; has been checked by the encoder
                    dim_feedforward=self.hidden_dim * 4,
                    batch_first=True,
                    dropout=0.0,  # TODO self._dropout_p,
                    activation="gelu",  # TODO get_transformer_act(self.arch_args),
                )
                self.cells[-1].add_module(f'dec{i_blk}', audio_tfm_layer)
        else:  # Spectrograms only are supported at the moment
            raise NotImplementedError(
                "Unimplemented '{}' architecture".format(self.arch['name']))

        # Finally, Python lists must be converted to nn.ModuleList to be properly recognized by PyTorch
        self.cells = nn.ModuleList(self.cells)

    def reshape_crop_pad_output(self, x):
        # known CNN output size: 257x257
        if self.arch['name'].startswith('specladder8x'):
            return x[:, :, :, 3:254]
        # ViT: "un-patch" the tokens, the crop and/or pad if needed
        elif self.is_transformer_based:
            # ----- 1) Reshape the sequence of embeddings into a proper 2D image -----
            # x is the output of a tranformer; currently has an N x L x E shape
            N, L = x.shape[0:2]
            # Currently: embedding size E has to be equal to patchH x patchW (= the number of pixel in a given patch)
            # We split each embedding vector to build a 2d patch
            list_of_2d_patches = x.view(N, L, self.patches.H, self.patches.W)
            # Then we turn the 1d list of patches into a 2d matrix of patches (not merged yet into an image)
            matrix_of_2d_patches = list_of_2d_patches.view(
                N, self.patches.n_rows, self.patches.n_cols, self.patches.H, self.patches.W)
            # Now we merge rows of pixels (along the columns axis), then columns of pixels
            x = matrix_of_2d_patches.transpose(2, 3)
            # (Can't view this one, we need to shape: "at least one dimension spans across two contiguous subspaces")
            x = x.reshape(N, self.patches.n_rows, self.patches.H, self.patches.n_cols * self.patches.W)
            x = x.view(N, self.patches.n_rows * self.patches.H, self.patches.n_cols * self.patches.W)
            # Finally add the singleton channel dimension (magnitude-only spectrogram)
            x = torch.unsqueeze(x, dim=1)

            # ----- 2) Crop and/or pad -----
            try:
                magnitude_padding_value = -1.0  # TODO padding value should be an arg
                if self.original_audio_shape[2:] == (257, 251):
                    x_padded = magnitude_padding_value * \
                               torch.ones(N, 1, *self.original_audio_shape[2:], device=x.device)
                    if self.patches.size == (16, 16):
                        x_padded[:, :, 0:256, :] = x[:, :, :, 0:251]  # also cropped here
                        return x_padded
                    elif self.patches.size == (256, 1):
                        x_padded[:, :, 0:256, :] = x[:, :, :, :]
                        return x_padded
                    else:
                        raise AssertionError()
                else:
                    raise AssertionError()
            except AssertionError:
                raise NotImplementedError(f"can't crop/pad decoder output for patch_size {self.patches.H, self.patches.W}"
                                          f" and audio shape {self.original_audio_shape[2:]} ")
        else:
            raise AssertionError(f"Cropping not implemented for the specific audio architecture '{self.arch['name']}'")




class AudioTransformerDecoderLayer(nn.Module):
    def __init__(self, dim_z: int, n_tokens: int, n_latent_memory_tokens=1,
                 is_first_layer=False, is_last_layer=False, output_dim: Optional[int]=None,
                 **kwargs):
        """
            Decoder layer which handles x, z inputs where x is a sequence of tokens and z is the latent code.

            TODO z can be used in different ways:
                - it can be pre-processed by the layer to be used as a memory
                -

        :param is_first_layer: If True, indicates that this layer is the first layer of cell #0 and specific
            inputs will be computed (e.g. positional encodings, learned embeddings, ...)
        :param kwargs: TransformerDecoderLayer ctor arguments
        """
        super().__init__()
        self.dim_z, self.n_tokens, self.n_latent_memory_tokens = dim_z, n_tokens, n_latent_memory_tokens
        self.is_first_layer, self.is_last_layer, self.output_dim = is_first_layer, is_last_layer, output_dim
        self.hidden_size = kwargs['d_model']

        # TODO allow skip-connection for the first dim_z values (which are going to be the raw z)
        #    and memory MLP (linear -> ELU -> linear), worked very well for the preset transformer decoder
        self.latent_memory_linear = nn.Linear(
            self.dim_z, self.n_latent_memory_tokens * self.hidden_size, bias=False)
        kwargs['norm_first'] = True  # seems to allow better performances
        self.transformer_layer = nn.TransformerDecoderLayer(**kwargs)

        # First layer adds positional embeddings
        self.positional_encodings, self.custom_in_tokens = None, None
        if self.is_first_layer:
            self.positional_encodings = model.audiomodel.AudioBase.get_1d_positional_embedding(
                self.hidden_size, self.n_tokens
            )
            # TODO should be optional (we may use z to build those)
            max_norm = np.sqrt(self.hidden_size)
            self.custom_in_tokens = nn.Embedding(self.n_tokens, self.hidden_size, max_norm=max_norm)

        # Last layer uses a final linear layer (final transformer output from a residual-add operation)
        if self.is_last_layer:
            assert self.output_dim is not None, "output_dim is required to compute the final linear projection"
            # Do use bias (spectrograms avg value is close to -1.0, median might be -1.0), no significant change anyway
            self.output_linear = nn.Linear(self.hidden_size, self.output_dim, bias=True)

    def forward(self, x_and_z_tuple):
        x, z = x_and_z_tuple
        N = x.shape[0]
        # x can be ignored for the first layer
        if self.is_first_layer:
            x = self.custom_in_tokens(torch.arange(0, self.n_tokens, device=x.device))
            x = x.expand(N, self.n_tokens, self.hidden_size).clone()  # clone required because expand does not copy
            x += self.positional_encodings.to(x.device)  # uses broadcasting

        # TODO other options to compute the memory tokens
        memory_tokens = self.latent_memory_linear(z).view(N, self.n_latent_memory_tokens, self.hidden_size)
        x = self.transformer_layer(x, memory=memory_tokens)

        if self.is_last_layer:
            x = self.output_linear(x)
        return x, z

