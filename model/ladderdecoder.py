import warnings
from typing import Dict, Tuple, List, Sequence, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo

import utils.probability
from model.audiodecoder import AudioDecoder
from model.ladderbase import LadderBase, AudioPassthroughLatentModule
from model.presetmodel import PresetEmbedding
from model.presetdecoder import PresetDecoder
from model.convlayer import ConvBlock2D, UpsamplingResBlock, SelfAttentionConv2D, ResBlockBase
from data.preset2d import Preset2dHelper


class LadderDecoder(LadderBase):
    def __init__(self, audio_arch, latent_arch, n_cells: int, dim_z: int,
                 audio_output_shape: Sequence[int],
                 audio_proba_distribution: str,
                 z_CNN_shape: Optional[List[int]] = None,
                 preset_architecture: Optional[str] = None,
                 preset_hidden_size: Optional[int] = None,
                 preset_numerical_proba_distribution: Optional[str] = None,
                 preset_helper: Optional[Preset2dHelper] = None,
                 preset_embedding: Optional[PresetEmbedding] = None,
                 preset_internal_dropout_p=0.0, preset_cat_dropout_p=0.0,
                 preset_label_smoothing=0.0, preset_use_cross_entropy_weights=False,
                 preset_params_loss_exclude_useless=False):
        # TODO doc
        super().__init__(audio_arch, latent_arch, dim_z, z_CNN_shape)
        if self.audio_is_a_CNN:
            assert self.z_CNN_shape is not None, "z_CNN_shape is fixed by the CNN encoder, and must be provided"
            assert len(self.z_CNN_shape) == 4

        # - - - - - 1) Build the audio module (outputs a single reconstructed audio channel) - - - - -
        self.audio_decoder = AudioDecoder(dim_z, audio_arch, n_cells, audio_output_shape, audio_proba_distribution)

        # - - - - - 2) Build latent module - - - - - TODO also build a transformer-based latent module
        # To refine main conv feature maps using the different levels of latent values

        # The decoder ctor is different from the encoder, because the only latent architecture it provides
        # is the feedforward convolutions (in latent cells). So, those cells are built directly inside this
        # ctor instead of being built in a separate class. (might be refactored in the future)
        if self.latent_arch['name'] == "conv" or self.latent_arch['name'] == 'lstm':  # FIXME
            if self.latent_arch['name'] == 'lstm':
                warnings.warn("Decoder: LSTM latent structure not implemented (the usual conv will be used instead)")
            # Convolutional latent structure: increase number of channels of latent features maps. The new channels
            # will be used as residuals added to the main convolutional path.
            if self.latent_arch['args']['k1x1']:
                kernel_size, padding = (1, 1), (0, 0)
            elif self.latent_arch['args']['k3x3']:
                kernel_size, padding = (3, 3), (1, 1)
            else:
                raise NotImplementedError("Can't build latent cells: conv kernel arg ('_k1x1' or '_k3x3') not provided")

            # H, W input for audio cells is the H, W from the corresponding input latent
            #    (we keep the low-res spatial structure; actually useful for CNNs only, but we keep it anyway)
            n_latent_input_ch = self.z_CNN_shape[1]
            n_latent_output_ch = self.audio_decoder.cells_input_shapes[0][1]
            self.latent_module = nn.Sequential()
            if self.latent_arch['args']['att'] and n_latent_input_ch >= 8:  # Useless with a small num. of channels
                input_H_W = tuple(self.z_CNN_shape[2:])
                if np.prod(input_H_W) > 16:
                    self.latent_module.add_module('resatt', SelfAttentionConv2D(
                        n_latent_input_ch, n_latent_input_ch,
                        position_encoding=self.latent_arch['args']['posenc'], input_H_W=input_H_W)
                    )
            else:
                if self.latent_arch['args']['posenc']:
                    warnings.warn("'posenc' arch arg works with 'att' only (which is False), thus will be ignored.")
            if self.latent_arch['args']['gated']:
                n_latent_output_ch *= 2  # GLU will half the number of outputs
            if self.latent_arch['n_layers'] == 1:
                self.latent_module.add_module('c', nn.Conv2d(n_latent_input_ch, n_latent_output_ch, kernel_size, 1, padding))
            elif self.latent_arch['n_layers'] == 2:  # No batch-norm inside the latent conv arch
                n_intermediate_ch = int(round(np.sqrt(n_latent_input_ch * n_latent_output_ch)))
                self.latent_module.add_module('c1', nn.Conv2d(n_latent_input_ch, n_intermediate_ch, kernel_size, 1, padding))
                self.latent_module.add_module('act', nn.ELU())
                self.latent_module.add_module('c2', nn.Conv2d(n_intermediate_ch, n_latent_output_ch, kernel_size, 1, padding))
            else:
                raise ValueError("Convolutional arch. for latent vector computation must contain <= 2 layers.")
            if self.latent_arch['args']['gated']:
                self.latent_module.add_module('gated', nn.GLU(dim=1))

        elif self.latent_arch['name'] == 'none':
            self.latent_module = AudioPassthroughLatentModule()

        else:
            raise NotImplementedError("Cannot build latent arch {}: name not implemented".format(latent_arch))

        # 3) Preset decoder (separate class)
        if preset_architecture is not None:
            assert (preset_numerical_proba_distribution is not None and preset_helper is not None
                    and preset_hidden_size is not None)
            self.preset_decoder = PresetDecoder(
                preset_architecture,
                self.dim_z,
                preset_hidden_size,
                preset_numerical_proba_distribution,
                preset_helper,
                preset_embedding,
                internal_dropout_p=preset_internal_dropout_p, cat_dropout_p=preset_cat_dropout_p,
                label_smoothing=preset_label_smoothing, use_cross_entropy_weights=preset_use_cross_entropy_weights,
                params_loss_exclude_useless=preset_params_loss_exclude_useless
            )
        else:
            self.preset_decoder: Optional[PresetDecoder] = None

    def get_custom_group_module(self, group_name: str):
        """ Returns a module """
        if group_name == 'audio':
            return self.audio_decoder
        elif group_name == 'latent':
            return self.latent_module
        elif group_name == 'preset':
            return self.preset_decoder  # Can be None
        else:
            raise ValueError("Unavailable group_name '{}'".format(group_name))

    @property
    def audio_cells(self):
        return self.audio_decoder.cells

    def forward(self, z_sampled: torch.Tensor,
                u_target: Optional[torch.Tensor] = None,
                x_target: Optional[torch.Tensor] = None, compute_x_out=True):
        """ Returns the p(x|z) distributions, values sampled from them
            and the NLL of x_target (if it's not None)
            and preset_decoder_out (tuple of 5 values) if this instance has a preset decoder, and
                if u_target is not None.
        """
        assert len(z_sampled.shape) == 2, "z_sampled should be a batch of 1D vectors"

        # ---------- Latent module and Audio decoder ----------
        #    The latent module is only used to pre-process the audio decoder input (preset decoder uses its own preproc)
        if compute_x_out:  # We may use the forward(...) to compute a preset only
            if self.audio_is_a_CNN:
                z_sampled_reshaped = self.unflatten_CNN_input(z_sampled)
            else:
                z_sampled_reshaped = z_sampled

            x = self.latent_module(z_sampled_reshaped)
            # audio decoder - apply cell-by-cell
            for cell_index, cell in enumerate(self.audio_cells):
                if self.audio_is_a_transformer:
                    x, _ = cell((x, z_sampled_reshaped), )  # transformer decoder returns (x, z)
                else:
                    x = cell(x)
            # crop (and possibly pad a few pixels) the last output
            audio_prob_parameters = self.audio_decoder.reshape_crop_pad_output(x)
            # Apply activations
            # The ch dimension stores (single or multiple) parameter(s) of the probability distribution
            audio_prob_parameters = self.audio_decoder.proba_distribution.apply_activations(audio_prob_parameters)
            # Sample from the probability distribution should always be fast and easy (even for mixture models)
            audio_x_sampled = self.audio_decoder.proba_distribution.get_mode(audio_prob_parameters)
            # compute non-reduced target NLL (if target available)
            if x_target is not None:
                audio_NLL = self.audio_decoder.proba_distribution.NLL(audio_prob_parameters, x_target)
            else:
                audio_NLL = None
        else:
            audio_prob_parameters, audio_x_sampled, audio_NLL = None, None, None

        # ---------- Preset decoder ----------
        if self.preset_decoder is not None and u_target is not None:
            preset_decoder_out = self.preset_decoder(z_sampled, u_target)
        else:
            preset_decoder_out = (None, ) * 5

        return audio_prob_parameters, audio_x_sampled, audio_NLL, preset_decoder_out

    def get_latent_module_summary(self):
        if self.audio_is_a_CNN:
            input_size = self.z_CNN_shape
        elif self.audio_is_a_transformer:
            input_size = (1, self.dim_z)
        else:
            raise NotImplementedError()
        return torchinfo.summary(
                self.latent_module, input_size=input_size,
                depth=5, verbose=0, device=torch.device('cpu'),
                col_names=self._summary_col_names, row_settings=("depth", "var_names")
            )

    def get_audio_only_summary(self):
        audio_cells_without_latent = nn.Sequential()
        for i, cell in enumerate(self.audio_cells):
            audio_cells_without_latent.add_module(f'audiocell{i}', cell)
        if self.audio_is_a_CNN:
            input_shape = list(self.z_CNN_shape)
            input_shape[1] = self.audio_cells[0][0].in_channels  # FIXME quite dirty, not robust to any architectural change...
            input_data = torch.zeros(input_shape)
        elif self.audio_is_a_transformer:
            E = self.audio_decoder.patches.H * self.audio_decoder.patches.W
            input_data = ((torch.zeros(1, self.audio_decoder.patches.count, E), torch.zeros(1, self.dim_z)), )
        else:
            assert False
        with torch.no_grad():
            return torchinfo.summary(
                audio_cells_without_latent, input_data=input_data,
                depth=7, verbose=0, device=torch.device('cpu'),
                col_names=self._summary_col_names, row_settings=("depth", "var_names")
            )



