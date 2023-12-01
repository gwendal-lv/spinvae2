import warnings
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

import model.audiomodel
from model.convlayer import ConvBlock2D, DownsamplingResBlock, SelfAttentionConv2D, ResConvBlock2d


class AudioEncoder(model.audiomodel.AudioBase):
    def __init__(self, dim_z: int, arch, n_cells: int, input_tensor_size: Tuple[int, int, int, int]):
        """
        Multi-cell (multi-stage) audio encoder, based on a CNN or a ViT (spectrogram input).

        This class does not provide a forward(...) method.
        Its cells are to be used by the LadderEncoder module which owns an instance of this class.
        """
        super().__init__(dim_z, arch, input_tensor_size)
        assert self.original_audio_shape[1] == 1, "Multi-channel (multi-note) input spectrograms are deprecated"
        self._input_size_batch1 = (1, 1, input_tensor_size[2], input_tensor_size[3])  # 1-item batch

        self.cells = list()
        # transformer-based models only
        self.n_output_tokens = -1

        args = self.arch['args']
        n_blocks = self.arch['n_blocks']  # Each cell contains a few blocks (which can be 1 layer)
        if self.arch['name'].startswith('specladder'):  # Is a CNN
            assert n_blocks == 8  # 9 layers would allow to set dim_z exactly but is not implemented yet
            if args['adain']:
                raise NotImplementedError()
            if args['att'] and self.arch['n_layers_per_block'] < 2:
                raise ValueError("'_att' conv arg will add a residual self-attention layer and requires >= 2 layers")

            if 1 <= n_cells <= (n_blocks - 2):
                cells_last_block = list(range(n_blocks - n_cells, n_blocks))
            else:
                raise NotImplementedError("Cannot build encoder with {} latent levels".format(n_cells))
            self.cells.append(nn.Sequential())

            for i_blk in range(n_blocks):
                residuals_path = nn.Sequential()
                blk_in_ch = 2 ** (i_blk + 2)  # number of input channels
                blk_out_ch = 2 ** (i_blk + 3)  # number of ch increases after each strided conv (at the block's end)
                if args['big']:
                    blk_in_ch, blk_out_ch = blk_in_ch * 2, blk_out_ch * 2
                min_ch = 1 if not args['bigger'] else 128
                max_ch = 512 if not args['bigger'] else 1024
                blk_in_ch, blk_out_ch = np.clip([blk_in_ch, blk_out_ch], min_ch, max_ch)
                blk_hid_ch = blk_in_ch  # base number of internal (hidden) block channels

                if i_blk == 0:  # First block: single conv layer
                    strided_conv_block = ConvBlock2D(nn.Conv2d(1, blk_out_ch, (5, 5), 2, 2), None, None, 'c')
                    residuals_path.add_module('strided', strided_conv_block)
                else:  # Other block can contain multiple conv block
                    for j in range(self.arch['n_layers_per_block'] - 1):
                        # first layer can be self-attention w/ 2D positional encodings
                        if j == 0 and args['att'] and (3 <= i_blk <= 5):
                            self_att = SelfAttentionConv2D(blk_hid_ch, position_encoding=True)
                            residuals_path.add_module('self_att', ConvBlock2D(
                                self_att, None, self._get_audio_norm(), 'nc'))
                        else:
                            # No depth-separable conv in the encoder (see NVAE NeurIPS 2020) - only 3x3 conv
                            conv = nn.Conv2d(blk_hid_ch, blk_hid_ch, (3, 3), 1, 1)
                            conv_block = ConvBlock2D(conv, self._get_conv_act(), self._get_conv_norm(), 'nac')
                            residuals_path.add_module('res_conv' + str(j), ResConvBlock2d(conv_block))
                    strided_conv = nn.Conv2d(blk_hid_ch, blk_out_ch, (4, 4), 2, 2)
                    residuals_path.add_module('strided', ConvBlock2D(
                        strided_conv, self._get_conv_act(), self._get_conv_norm(), 'nac'))

                # Add a skip-connection if required, then add this new block to the current cell
                if args['res'] and i_blk > 0:
                    current_block = DownsamplingResBlock(residuals_path)
                else:
                    current_block = residuals_path  # No skip-connection
                self.cells[-1].add_module('blk{}'.format(i_blk), current_block)
                if i_blk in cells_last_block and i_blk < (n_blocks - 1):  # Start building the next cell
                    self.cells.append(nn.Sequential())

        elif self.arch['name'].startswith('ast'):  # audio spectrogram transformer
            self.cells.append(nn.Sequential())  # TODO multiple cells

            assert self.hidden_dim >= self.dim_z
            if self.hidden_dim < self.patches.size[0] * self.patches.size[1]:
                warnings.warn("Hidden token dimension is smaller than the patch it represents")
            self.n_output_tokens = 2  # TODO should be an arg
            self._patcher = LinearSpectrogramPatcher(
                input_tensor_size, self.patches.size,
                self.hidden_dim, self.n_output_tokens,
            )
            self.cells[-1].add_module("patch_and_PE", self._patcher)

            # TODO divide blocks into cells
            assert n_cells == 1, "Multiple cells are not implemented yet"
            assert self.hidden_dim % 64 == 0, "Each head's hidden size will be 64"
            n_head = self.hidden_dim // 64
            dropout_p = 0.0  # TODO should be an arg
            # We don't use a TransformerEncoder instance, which does not allow to use different layers
            #       TransformerEncode simply clones the given layer, and possibly applies an output norm
            for i_blk in range(self.arch['n_blocks']):
                # Each EncoderLayer contains only 1 (self-)attention layer
                #    -> 3 sub-layers (including the 2 Linear ones, norm excluded)
                tfm_layer = nn.TransformerEncoderLayer(
                    self.hidden_dim, n_head, dim_feedforward=self.hidden_dim*4,
                    batch_first=True, dropout=dropout_p, activation="gelu", # TODO get_transformer_act(self.arch_args)
                    norm_first=True,  # seems to allow better performance
                )
                self.cells[-1].add_module(f"enc{i_blk}", tfm_layer)
            # Will select the first tokens only, and crop the embeddings if needed (if larger than dim_z)
            use_linear, crop_output = False, True  # TODO maybe don't use if hidden_dim = z_dim?
            self.cells[-1].add_module("tokenselect", OutputTokensSelector(
                self.n_output_tokens, crop_output, use_linear, self.hidden_dim, self.dim_z
            ))

        else:  # Spectrograms only are supported at the moment
            raise NotImplementedError("Unimplemented '{}' architecture".format(self.arch['name']))

        # Finally, Python lists must be converted to nn.ModuleList to be properly recognized by PyTorch
        self.cells = nn.ModuleList(self.cells)

    @property
    def cells_output_shapes(self):
        x = torch.zeros(self._input_size_batch1)
        output_sizes = list()
        with torch.no_grad():
            for cell in self.cells:
                x = cell(x)
                output_sizes.append(x.shape)
        return output_sizes



class LinearSpectrogramPatcher(nn.Module):
    def __init__(self,
                 input_tensor_size: Tuple[int, int, int, int],
                 patch_size: Tuple[int, int],
                 embedding_dim: Optional[int] = None,
                 n_extra_tokens=0,
                 bias=True,
                 null_padding_value=-1.0):
        """
        Computes linear embeddings of patches from an input spectrogram
        TODO also handle PE, add PE args
        TODO extra tokens

        :param embedding_dim: Token dimension, which can be smaller or greater than the number of pixel of
                one patch. If None, embeddings dimension is patchH * patchW.
        :param null_padding_value: The value corresponding to -inf dB magnitude, used to pad spectrograms on their
                top and right size.
        """
        super().__init__()
        self.input_tensor_size, self.patch_size, self.pad_value = input_tensor_size, patch_size, null_padding_value
        assert self.input_tensor_size[1:] == (1, 257, 251)  # 1ch, 257 frequencies (FFT 512), 251 time frames
        self.embedding_dim = int(np.prod(patch_size)) if embedding_dim is None else embedding_dim
        self.n_extra_tokens = n_extra_tokens
        # We use a very large conv to build the patches
        self.patches_split_conv = nn.Conv2d(
            1, self.embedding_dim,
            patch_size, patch_size,  # Kernel defines the patch size, stride is also the patch size (no overlap)
            padding='valid',  # same as no padding - we'll pad manually to ensure the results
            bias=bias
        )
        # TODO Check sequence length, patch size, etc...
        # Extra token embeddings
        if self.n_extra_tokens > 0:
            self.extra_token_embeds = nn.Embedding(self.n_extra_tokens, self.embedding_dim)
        else:
            self.extra_token_embeds = None

    def forward(self, x: torch.Tensor):
        N = x.shape[0]
        # Crop the input - unimplemented cases are handled (exceptions) in the ctor
        if self.patch_size == (16, 16):
            # FFT 512, we can ditch the highest freq (supposed to be anti-aliasing filtered)
            # And we pad a few -1.0 pixels on the right
            if x.shape[2] == 257:  # Frequencies along the Height dimension
                n_W_patches = int(np.ceil(x.shape[3] / self.patch_W))  # Width is the time dimension
                x_cropped_padded = self.pad_value * torch.ones(
                    (N, x.shape[1], 256, n_W_patches * self.patch_W),
                    device=x.device, dtype=x.dtype
                )
                x_cropped_padded[:, :, :, 0:x.shape[3]] = x[:, :, 0:256, :]
            else:
                raise NotImplementedError(f"Can't crop {x} input for patch_size {self.patch_size}")
        elif self.patch_size == (256, 1):
            # We also forget the highest frequency; this code should work for any number of time frames, however
            if x.shape[2] == 257:
                x_cropped_padded = x[:, :, 0:256, :]
            else:
                raise NotImplementedError(f"Can't crop {x} input for patch_size {self.patch_size}")
        else:
            raise NotImplementedError(f"Crop not implemented for patch_size {self.patch_size}")
        # Compute patches -> output shape is N x E (embedding dim) x H x W
        x_patches = self.patches_split_conv(x_cropped_padded)
        # Each "pixel" is now a token with a large hidden dim (= number of conv output channels)
        #     -> we build a sequence
        # Shape after view(...): N x E x L where L is the sequence Length. Lines (last dim) remain contiguous.
        x_patches_seq = x_patches.view(N, self.embedding_dim, x_patches.shape[2] * x_patches.shape[3])
        x_patches_seq = x_patches_seq.transpose(1, 2)  # output shape: N x L x E (as expected by the Transformer)
        # TODO is contiguous required here??
        if self.n_extra_tokens > 0:
            extra_tokens = self.extra_token_embeds(torch.arange(0, 2, device=x.device))
            extra_tokens = extra_tokens.expand(N, *extra_tokens.shape)
            x_patches_seq = torch.cat((extra_tokens, x_patches_seq), dim=1)  # concat along sequence dim
        # append extra tokens, add positional encodings
        pe = model.audiomodel.AudioBase.get_1d_positional_embedding(
            self.embedding_dim, x_patches_seq.shape[1],
        ).to(x_patches_seq.device)
        x_patches_seq += pe  # pe doesn't have a batch dim; use broadcasting (trailing dimensions are the same)
        return x_patches_seq

    @property
    def patch_H(self):
        return self.patch_size[0]

    @property
    def patch_W(self):
        return self.patch_size[1]



class OutputTokensSelector(nn.Module):
    def __init__(self, n_tokens: int,
                 crop_output=False,
                 use_linear=False, hidden_dim: Optional[int] = None, dim_z: Optional[int] = None):
        """ Very simple module (can be appended to a nn.Sequential) to keep only the first few tokens.
            Expected input shape: N x T x E where T is the sequence length and E is the embedding size.

            :param crop_output: If True, output tokens' embedding dimension will be reduced to dim_z
            :param use_linear: If True, use a final linear layer. Requires hidden_dim and dim_z to be set, and
                    should be used if hidden_dim > dim_z
        """
        super().__init__()
        self.n_tokens = n_tokens

        self.dim_z, self.hidden_dim = dim_z, hidden_dim

        if use_linear:  # Linear on the "concatenated" tokens (we don't use the same linear for all tokens)
            assert hidden_dim is not None and dim_z is not None, "These args are required to build the linear layer."
            self.linear = nn.Linear(self.n_tokens * self.hidden_dim, self.n_tokens * self.dim_z)
        else:
            self.linear = None

        self.crop_output = crop_output
        if self.crop_output:
            assert self.dim_z is not None

        assert not (use_linear and self.crop_output), "Linear layer and cropping should not be used together"

    def forward(self, x):
        N = x.shape[0]
        x = x[:, 0:self.n_tokens, :]
        if self.linear is not None:
            x = x.view(N, self.n_tokens * self.hidden_dim)  # Sequence of n_tokens, to a 1D vector
            x = self.linear(x)
            x = x.view(N, self.n_tokens, self.dim_z)  # 1D vector to sequence of tokens
        if self.crop_output:
            x = x[:, :, 0:self.dim_z]
        return x

