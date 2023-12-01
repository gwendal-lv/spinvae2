import copy
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
import warnings

from model.audioencoder import AudioEncoder
import model.presetencoder
from model.ladderbase import LadderBase
from model.convlayer import ConvBlock2D, DownsamplingResBlock, SelfAttentionConv2D
from model.convlstm import ConvLSTM
from data.preset2d import Preset2d, Preset2dHelper


class LadderEncoder(LadderBase):

    def __init__(self, audio_arch, latent_arch,
                 n_cells: int,
                 audio_input_tensor_size: Tuple[int, int, int, int],
                 dim_z: int,
                 preset_architecture: Optional[str] = None,
                 preset_hidden_size: Optional[int] = None,
                 preset_encode_add: Optional[str] = None,
                 preset_helper: Optional[Preset2dHelper] = None,
                 preset_dropout_p=0.0):
        """
        Contains cells which define the hierarchy levels (output features of each cell can be used elsewhere)
            Each cell is made of blocks (skip connection may be added/concat/other at the end of block)
                Each block contains one or more conv, act and norm layers

        :param n_cells:
        :param audio_input_tensor_size:
        :param preset_encode_add: TODO doc, maybe rename this arg
        """

        super().__init__(audio_arch, latent_arch, dim_z)
        self.preset_encode_add = preset_encode_add

        # - - - - - 1) Build the audio encoder (input: single-channel spectrogram; 1 midi note) - - - - -
        self.audio_encoder = AudioEncoder(dim_z, audio_arch, n_cells, audio_input_tensor_size)

        # - - - - - 2) Latent inference network - - - - -  FIXME should be 3rd, to be able to merge preset data as well
        # TODO handle the non-conv case (e.g. Transformer-based latent module)

        # Conv alone: for a given level, feature maps from all input spectrograms are merged using a conv network only
        # and the output is latent distributions parameters mu and sigma^2. Should be a very simple and fast network
        if self.latent_arch['name'] == "conv" or self.latent_arch['name'] == 'lstm':
            # Latent space size - convolutional audio encoder
            last_audio_out_shape = self.audio_encoder.cells_output_shapes[-1]
            assert self.dim_z % (last_audio_out_shape[2] * last_audio_out_shape[3]) == 0, \
                f"dim_z must be a multiple of {last_audio_out_shape[2] * last_audio_out_shape[3]}"
            n_latent_ch = 2 * self.dim_z // (last_audio_out_shape[2] * last_audio_out_shape[3])  # Output mu and sigma
            self.z_CNN_shape = [s for s in last_audio_out_shape]  # Shape of a sampled z
            self.z_CNN_shape[1] = n_latent_ch // 2

            use_lstm = self.latent_arch['name'] == 'lstm'
            if self.latent_arch['args']['k1x1']:
                kernel_size, padding = (1, 1), (0, 0)
            elif self.latent_arch['args']['k3x3']:
                kernel_size, padding = (3, 3), (1, 1)
            else:
                raise NotImplementedError("Can't build latent cells: conv kernel arg ('_k1x1' or '_k3x3') not provided")
            # FIXME preset encodings can be added to those cells' inputs
            audio_seq_len = 1  # FIXME shouldn't be used anymore
            cell_args = (
                last_audio_out_shape[1], last_audio_out_shape[2:], audio_seq_len,
                n_latent_ch, self.latent_arch['n_layers'], kernel_size, padding, self.latent_arch['args']
            )
            self.latent_module = ConvLSTMLatentCell(*cell_args) if use_lstm else ConvLatentCell(*cell_args)
        elif self.latent_arch['name'] == 'none':
            self.latent_module = model.ladderbase.AudioPassthroughLatentModule()
            warnings.warn("No latent module - should be used for pre-training benchmarks only")
        else:
            raise NotImplementedError("Cannot build latent arch {}: name not implemented".format(latent_arch))

        # 3) Preset encoder (no hierarchical levels) - in its own Python module
        if preset_helper is not None:  # FIXME adapt to single-ch, single-latent level new model
            assert preset_architecture is not None and preset_hidden_size is not None
            # FIXME this reshaping should be done by a dedicated latent module (which handles the merging of modalities)
            # Shape does not include the batch dim
            if preset_encode_add.lower() == "before_latent_cell":  # FIXME move to latent module, be more general
                raise NotImplemented("Not compatible anymore with the new 1-lvl latent structure")
                self.u_hidden_add_before_latent_cell = True
                encoded_preset_fm_shape = list(self.audio_cells_output_shapes[0][1:])  # 2d feature maps
                encoded_preset_fm_shape[0] *= self._audio_seq_len
            elif preset_encode_add.lower() == "after_latent_cell":
                self.u_hidden_add_before_latent_cell = False
                encoded_preset_fm_shape = list(self.z_CNN_shape[1:])
                encoded_preset_fm_shape[0] *= 2  # mu and sigma (or var)
            else:
                raise ValueError("model_config.vae_preset_encode_add must be either 'before_latent_cell' or "
                                 "'after_latent_cell' (current: '{}')".format(preset_encode_add))
            self.preset_encoder = model.presetencoder.PresetEncoder(
                preset_architecture, preset_hidden_size, preset_helper,
                self.dim_z, encoded_preset_fm_shape, preset_dropout_p
            )
        else:
            self.preset_encoder = None

    def get_custom_group_module(self, group_name: str):
        """ Returns a module """
        if group_name == 'audio':
            return self.audio_encoder
        elif group_name == 'latent':
            return self.latent_module
        elif group_name == 'preset':
            return self.preset_encoder
        else:
            raise ValueError("Unavailable group_name '{}'".format(group_name))

    @property
    def audio_cells(self):
        return self.audio_encoder.cells

    def forward(self, x: Optional[torch.Tensor] = None, u: Optional[torch.Tensor] = None,
                midi_notes: Optional[torch.Tensor] = None):
        """
        This method should be able to use x alone, or u alone, or both. It will use what's available
        (its HierarchicalVAE owner will decided what is to be encoded, or not)

        :returns: (z_mu, z_var): lists of length latent_levels
        """
        audio_cells_outputs = []  # can be used for cross-modal training mechanisms
        # TODO preset_cells_outputs

        # 1) Apply single-channel CNN to all input channels
        if x is not None:  # usual case
            x_hidden = x
            for cell_index, cell in enumerate(self.audio_cells):
                x_hidden = cell(x_hidden)
                audio_cells_outputs.append(x_hidden)

        # 2) Optional: Compute hidden representation of the preset
        u_hidden = 0.0
        # preset might not be given at input (audio AE only, or AutoSynthProg mode)
        if self.preset_encoder is not None and u is not None:
            u_hidden = self.preset_encoder(u)

            # maybe add preset "hidden residuals" before FIXME this should be
            if self.u_hidden_add_before_latent_cell:
                raise NotImplementedError()
                u_hidden = u_hidden.view(u_hidden.shape[0], self._audio_seq_len, *self.audio_cells_output_shapes[0][1:])
                latent_cells_input_tensors[0] += u_hidden

        # 3) Compute latent vectors: mean and variance

        # if x is None, maybe don't even use the latent cells  FIXME latent cell should handle audio AND preset altogether
        if x is None and not self.u_hidden_add_before_latent_cell:  # Don't process zeroes through the latent NN
            z_out = 0.0
        elif x is None and self.u_hidden_add_before_latent_cell:
            raise NotImplementedError()
        else:  # Usual case: apply latent cell to extract features from audio
            z_out = self.latent_module(audio_cells_outputs[-1], midi_notes=midi_notes)

        # Maybe add preset "latent residuals" after  FIXME this should be done by the latent cell itself
        if self.preset_encoder is not None and not self.u_hidden_add_before_latent_cell:
            z_out = z_out + u_hidden  # One of these two can be a 0.0  scalar


        if self.audio_is_a_CNN:  # Reshape (flatten) z, if needed
            z_out = self.flatten_CNN_output(z_out)
            # Split into mu and variance
            z_mu = z_out[:, 0:self.dim_z]
            z_var = F.softplus(z_out[:, self.dim_z:])
        elif self.audio_is_a_transformer:
            z_mu = z_out[:, 0, :]
            z_var = F.softplus(z_out[:, 1, :])
        else:
            raise NotImplemented("Unknown z_out format")
        return z_mu, z_var

    def get_audio_only_summary(self):
        audio_encoder_without_latent = nn.Sequential()
        for i, cell in enumerate(self.audio_cells):
            audio_encoder_without_latent.add_module('audio_cell{}'.format(i), cell)
        with torch.no_grad():
            return torchinfo.summary(
                audio_encoder_without_latent, input_size=self.audio_encoder.original_audio_shape,
                depth=5, verbose=0, device=torch.device('cpu'),
                col_names=self._summary_col_names, row_settings=("depth", "var_names")
            )

    def get_latent_module_summary(self):  # FIXME should also include optional preset inputs
        latent_module_input_shape = list(self.audio_encoder.cells_output_shapes[-1])
        input_data = {'x_audio': torch.zeros(latent_module_input_shape),
                      'midi_notes': torch.zeros((latent_module_input_shape[0], 1, 2))}  # single MIDI note
        with torch.no_grad():
            return torchinfo.summary(
                self.latent_module, input_data=input_data,
                depth=5, verbose=0, device=torch.device('cpu'),
                col_names=self._summary_col_names, row_settings=("depth", "var_names")
            )



class ConvLatentCell(nn.Module):
    def __init__(self, num_audio_input_ch: int, audio_input_H_W, audio_sequence_len: int,
                 num_latent_ch: int, num_layers: int,
                 kernel_size: Tuple[int, int], padding: Tuple[int, int], arch_args: Dict[str, bool]):
        """
        A class purely based on convolutions to mix various image feature maps into a latent feature map with
        a different number of channels.

        :param num_audio_input_ch: Number of channels in feature map extracted from a single audio element
            (e.g. a single MIDI Note)
        :param audio_input_H_W: Height and Width of audio feature maps that will be provided to this module
        :param audio_sequence_len:
        :param num_latent_ch:
        :param num_layers:
        """
        super().__init__()
        # FIXME ALWAYS add channels to mix preset values (use zeros if not used)
        total_num_input_ch = audio_sequence_len * num_audio_input_ch
        self.conv = nn.Sequential()
        # Self-attention for "not-that-small" feature maps only, i.e., feature maps which are larger
        #    than the 4x4 conv kernels from the main convolutional path
        if arch_args['att']:
            if np.prod(audio_input_H_W) > 16:
                self.conv.add_module('resatt', SelfAttentionConv2D(
                    total_num_input_ch, position_encoding=arch_args['posenc'], input_H_W=audio_input_H_W))
        else:
            if arch_args['posenc']:
                warnings.warn("'posenc' arch arg works with 'att' only (which is False), thus will be ignored.")
        num_out_ch = num_latent_ch if not arch_args['gated'] else 2 * num_latent_ch
        if num_layers == 1:
            self.conv.add_module('c', nn.Conv2d(total_num_input_ch, num_out_ch, kernel_size, 1, padding))
        elif num_layers == 2:  # No batch-norm inside the latent conv arch
            n_intermediate_ch = int(round(np.sqrt(total_num_input_ch * num_out_ch)))  # Geometric mean
            self.conv.add_module('c1', nn.Conv2d(total_num_input_ch, n_intermediate_ch, kernel_size, 1, padding))
            self.conv.add_module('act', nn.ELU())
            self.conv.add_module('c2', nn.Conv2d(n_intermediate_ch, num_out_ch, kernel_size, 1, padding))
        else:
            raise ValueError("Convolutional arch. for latent vector computation must contain <= 2 layers.")
        if arch_args['gated']:
            self.conv.add_module('gated', nn.GLU(dim=1))

    def forward(self, x_audio: torch.tensor,
                u_preset: Optional[torch.Tensor] = None, midi_notes: Optional[torch.Tensor] = None):
        """

        :param x_audio: Tensor with shape N x T x C x W x H where T is the original audio sequence length
        :param u_preset:
        :param midi_notes: May be used (or not, depends on the exact configuration) to add some "positional
            encoding" information to each sequence item in x_audio
        :return:
        """
        if u_preset is not None:
            warnings.warn("Preset encoding not implemented in conv latent cells - preset input ignored")
        # x_audio = torch.flatten(x_audio, start_dim=1, end_dim=2)  # merge sequence and channels dimensions FIXME handle properly
        return self.conv(x_audio)


class ConvLSTMLatentCell(nn.Module):
    def __init__(self, num_audio_input_ch: int, audio_input_H_W, audio_sequence_len: int,
                 num_latent_ch: int, num_layers: int,
                 kernel_size: Tuple[int, int], padding: Tuple[int, int], arch_args: Dict[str, bool]):
        """
        Latent cell based on a Convolutional LSTM (keeps the spatial structure of data)

        See ConvLatentCell to get information about parameters.
        """
        super().__init__()
        if arch_args['att']:
            warnings.warn("Self-attention arch arg 'att' is not valid and will be ignored.")
        # Main LSTM
        n_intermediate_ch = int(round(np.sqrt(num_audio_input_ch * num_latent_ch)))  # Geometric mean
        self.lstm_net = ConvLSTM(num_audio_input_ch, n_intermediate_ch, kernel_size, num_layers, bidirectional=True)
        # Positional encodings are added as residuals to help the LSTM to know which MIDI note is being processed.
        if arch_args['posenc']:
            self.note_encoder = nn.Sequential(
                nn.Linear(2, 4), nn.ReLU(),
                nn.Linear(4, np.prod(audio_input_H_W) * num_audio_input_ch), nn.Tanh()
            )
        else:
            self.note_encoder = None
        # Final regular conv to extract latent values, because the LSTM hidden state is activated (and these
        #    final activations might be undesired in the VAE framework). Also required because the bidirectional
        #    LSTM increases 2x the number of hidden channels.
        self.final_conv = nn.Conv2d(n_intermediate_ch*2, num_latent_ch, kernel_size, 1, padding)

    def forward(self, x_audio: torch.tensor,
                u_preset: Optional[torch.Tensor] = None, midi_notes: Optional[torch.Tensor] = None):
        if u_preset is not None:  # FIXME also use preset feature maps (first hidden features?)
            warnings.warn("Encoder latent cells do not supports preset yet - preset input ignored")
        # Learned residual positional encoding (per-pixel bias on the full feature maps)
        if midi_notes is not None and self.note_encoder is not None:
            for seq_idx in range(midi_notes.shape[1]):
                pos_enc = self.note_encoder(-0.5 + midi_notes[:, seq_idx, :] / 64.0)
                pos_enc = pos_enc.view(-1, *x_audio.shape[2:])
                if len(x_audio.shape) == 4:  # 4d tensor: raw audio input
                    x_audio[:, seq_idx, :, :] += pos_enc
                elif len(x_audio.shape) == 5:  # 5d tensor: spectrogram input
                    x_audio[:, seq_idx, :, :, :] += pos_enc
                else:
                    raise AssertionError()
        # Warning: the first n_intermediate_ch contains information about the preset itself and the first
        #    spectrogram only. This provides an "almost direct" path from preset to latent values (which we
        #    might want to avoid...???)
        # OTHER WARNING: h is a product of a sigmoid-activated and a tanh-activated value
        #    this implies a form a regularization before computing latent distributions' parameters
        output_sequence, last_states_per_layer = self.lstm_net(x_audio)
        return self.final_conv(output_sequence[:, -1, :, :, :])  # Use the last hidden state


if __name__ == "__main__":
    # First tests for dev
    _audio_input_tensor_size = (11, 1, 257, 251)
    _audio_arch = {'name': 'ast3', 'n_blocks': 3, 'n_layers_per_block': -1, 'args': {'adain': False, 'big': False, 'bigger': False, 'res': False, 'att': False, 'depsep5x5': False, 'swish': False, 'wn': False}}
    _latent_arch = {'name': 'none', 'n_layers': 0, 'args': {'k1x1': False, 'k3x3': False, 'posenc': False, 'gated': False, 'att': False}}
    _n_cells = 1
    _dim_z = 256

    enc = LadderEncoder(_audio_arch, _latent_arch, _n_cells, _audio_input_tensor_size, _dim_z)
