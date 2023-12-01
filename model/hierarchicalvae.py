from typing import Optional

import numpy as np
import torch
import torchinfo
from torch import nn
from torch.distributions.normal import Normal

import config
import model.base
import model.ladderbase
import model.ladderencoder
import model.ladderdecoder
import model.loss
import utils.probability
from data.preset2d import Preset2dHelper
from model.ladderbase import parse_latent_extract_architecture
from model.audiomodel import parse_audio_architecture
from utils.probability import MMD
import utils.exception


class HierarchicalVAEOutputs:
    def __init__(self, z_mu: torch.Tensor, z_var: torch.Tensor, z_sampled: torch.Tensor,
                 x_decoded_proba, x_sampled, x_target_NLL,
                 u_out, u_numerical_nll, u_categorical_nll, u_l1_error, u_accuracy
                 ):
        """
        Class to store outputs of a hierarchical VAE. Some constructor args are optional depending on the VAE
        architecture and training procedure
        """
        self.z_mu, self.z_var, self.z_sampled = z_mu, z_var, z_sampled
        self.x_decoded_proba, self.x_sampled, self.x_target_NLL = x_decoded_proba, x_sampled, x_target_NLL
        self.u_out, self.u_numerical_nll, self.u_categorical_nll, self.u_l1_error, self.u_accuracy = \
            u_out, u_numerical_nll, u_categorical_nll, u_l1_error, u_accuracy
        # Quick tests to try to ensure that the proper data was provided
        assert self.z_mu.shape == self.z_var.shape == self.z_sampled.shape
        assert len(self.z_mu.shape) == 2, "z should be a batch of vectors"
        if u_out is not None:
            assert len(u_out.shape) == 3
            assert len(u_numerical_nll.shape) == 1
            assert len(u_accuracy.shape) == 1
        # Can be assigned later
        self.z_loss = None


class HierarchicalVAE(model.base.TrainableMultiGroupModel):
    def __init__(self, model_config: config.ModelConfig, train_config: Optional[config.TrainConfig] = None,
                 preset_helper: Optional[Preset2dHelper] = None):
        """
        Builds a Hierarchical VAE which encodes and decodes multi-channel spectrograms or waveforms.
        A preset can be encoded/decoded as well (optional during pre-training).

        The hierarchical structures allows to infer latent vectors from different levels, not only from
        the top level (vanilla VAE).
        The encoder is based on a single-channel CNN build upon multiple cells. Each cell's output can be used to
        infer a partial latent vector.
        """
        trainable_param_group_names = ['audio', 'latent'] + ([] if train_config.pretrain_audio_only else ['preset'])
        for group_name in train_config.frozen_weight_groups:
            trainable_param_group_names.remove(group_name)
        super().__init__(train_config, ['audio', 'latent', 'preset'], trainable_param_group_names)

        # Save some important values from configurations
        self.dim_z = model_config.dim_z
        self._input_audio_tensor_size = model_config.input_audio_tensor_size
        self._latent_free_bits = train_config.latent_free_bits
        self._normalize_latent_loss = train_config.normalize_latent_loss

        # Pre-process general configuration
        self.audio_arch = parse_audio_architecture(model_config.vae_audio_architecture)
        self.preset_ae_method = model_config.preset_ae_method
        # Latent architecture and loss
        self.latent_arch = parse_latent_extract_architecture(model_config.vae_latent_extract_architecture)
        self._alignment_criterion = train_config.preset_alignment_criterion
        latent_loss_args = train_config.latent_loss.split('+')
        assert latent_loss_args[0].lower() == 'dkl'
        self.latent_regularization_gamma = train_config.latent_regularization_gamma
        self.latent_regularization_delta = train_config.latent_regularization_delta
        self._latent_attribute_based_regularization = None
        self._latent_label_based_regularization = None
        self._latent_extra_regularization_expand_to_z_chunks = False
        for latent_loss_arg in latent_loss_args[1:]:
            if latent_loss_arg.lower() == 'attrs2vae':
                assert self._latent_attribute_based_regularization is None, "Regularization can't be set twice."
                self._latent_attribute_based_regularization = "S2VAE"
            elif latent_loss_arg.lower() == 'attrarvae':
                assert self._latent_attribute_based_regularization is None, "Regularization can't be set twice."
                self._latent_attribute_based_regularization = "ARVAE"
            elif latent_loss_arg.lower() == "zchunks":
                self._latent_extra_regularization_expand_to_z_chunks = True
            else:
                raise ValueError(f"Unexpected latent loss arg '{latent_loss_arg}' in '{train_config.latent_loss}'")

        # Configuration checks
        if not train_config.pretrain_audio_only:
            if self.preset_ae_method not in \
                    ['no_encoding', 'combined_vae', 'no_audio', 'aligned_vaes', 'no_input_audio']:
                raise ValueError(f"preset_ae_method '{self.preset_ae_method}' not available")
            if self.preset_ae_method == 'aligned_vaes':
                assert model_config.vae_preset_encode_add == 'after_latent_cell', \
                    "Independent VAEs should not share the latent cells "
        else:
            assert self.preset_ae_method is None

        # Build encoder and decoder
        self._preset_helper = preset_helper
        if train_config.pretrain_audio_only:
            encoder_opt_args, dummy_u = (None, ) * 5, None
        else:
            assert preset_helper is not None
            encoder_opt_args = (model_config.vae_preset_architecture,
                                model_config.preset_hidden_size,
                                model_config.vae_preset_encode_add,
                                preset_helper,
                                train_config.preset_internal_dropout)
            dummy_u = preset_helper.get_null_learnable_preset(train_config.minibatch_size)
        self.encoder = model.ladderencoder.LadderEncoder(
            self.audio_arch, self.latent_arch, model_config.vae_n_cells, model_config.input_audio_tensor_size,
            self.dim_z,
            *encoder_opt_args
        )
        if train_config.pretrain_audio_only:
            decoder_opt_args = (None, ) * 7
        else:
            decoder_opt_args = (model_config.vae_preset_architecture,
                                model_config.preset_hidden_size,
                                model_config.preset_decoder_numerical_distribution,
                                preset_helper,
                                self.encoder.preset_encoder.embedding,  # Embedding net is built by the encoder
                                train_config.preset_internal_dropout, train_config.preset_cat_dropout,
                                train_config.preset_CE_label_smoothing,
                                train_config.preset_CE_use_weights,
                                train_config.params_loss_exclude_useless)
        self.decoder = model.ladderdecoder.LadderDecoder(
            self.audio_arch, self.latent_arch, model_config.vae_n_cells, self.dim_z,
            model_config.input_audio_tensor_size,
            model_config.audio_decoder_distribution,
            self.encoder.z_CNN_shape,
            *decoder_opt_args
        )

        # Build a ModuleList for each group of parameters (e.g. audio/latent/preset, ...)
        # Their only use is to aggregate parameters into a single nn.Module, in order to e.g. use
        #     a different optimizer for each one
        self._aggregated_modules_lists = {
            k : nn.ModuleList([m.get_custom_group_module(k) for m in [self.encoder, self.decoder]])
            for k in self.param_group_names
        }

        # Losses and metrics
        self.mmd = MMD()

        # Optimizers and schedulers (all sub-nets, param groups must have been created at this point)
        self._init_optimizers_and_schedulers()

    def get_custom_group_module(self, group_name: str) -> nn.Module:
        return self._aggregated_modules_lists[group_name]

    def forward(self, x_target, u_target=None, preset_uids=None, midi_notes=None, pass_index=0):
        if pass_index == 0:  # - - - - - 1st pass (might be the single one) - - - - -
            if self.pre_training_audio or self.preset_ae_method == 'no_encoding':
                x_input, u_input = x_target, None
            elif self.preset_ae_method == 'combined_vae':
                x_input, u_input = x_target, u_target
            elif self.preset_ae_method == 'aligned_vaes':
                x_input, u_input = None, u_target  # 1st default pass: preset (s.t. default always outputs a preset)
                x_target = None  # will disable computation of x outputs (audio distribution and samples)
            elif self.preset_ae_method == 'no_audio':
                x_input, u_input = None, u_target
                x_target = None  # will disable computation of x outputs (audio distribution and samples)
            elif self.preset_ae_method == 'no_input_audio':
                x_input, u_input = None, u_target
            else:
                raise NotImplementedError(self.preset_ae_method)
        elif pass_index == 1:  # - - - - -  2nd pass (optional) - - - - -
            if self.preset_ae_method == 'aligned_vaes':
                x_input, u_input = x_target, None  # 2nd pass: audio only
                u_target = None  # disable preset loss computation
            else:
                raise AssertionError("This model's training procedure is single-pass.")
        else:
            raise ValueError(pass_index)

        # Encode, sample, decode
        z_mu, z_var = self.encoder(x_input, u_input, midi_notes)
        z_sampled = self.sample_z(z_mu, z_var)
        # TODO maybe don't use x_target if auto-encoding preset only
        x_decoded_proba, x_sampled, x_target_NLL, preset_decoder_out = self.decoder(
            z_sampled, u_target=u_target, x_target=x_target, compute_x_out=(x_target is not None)
        )

        # Outputs: return all available values using Tensors only, for this method to remain usable
        #     with multi-GPU training (mini-batch split over GPUs, all output tensors will be concatenated).
        #     This tuple output can be parsed later into a proper HierarchicalVAEOutputs instance.
        #     All of these output tensors must retain the batch dimension
        return z_mu, z_var, z_sampled, x_decoded_proba, x_sampled, x_target_NLL, *preset_decoder_out

    def sample_z(self, z_mu, z_var):
        if self.training:  # Sampling in training mode only
            eps = Normal(torch.zeros_like(z_mu, device=z_mu.device),
                         torch.ones_like(z_mu, device=z_mu.device)).sample()
            return z_mu + torch.sqrt(z_var) * eps
        else:  # eval mode: no random sampling
            return z_mu

    def parse_outputs(self, forward_outputs):
        """ Converts a tuple output from a self.forward(...) call into a HierarchicalVAEOutputs instance. """
        i = 0
        z_mu = forward_outputs[i]
        z_var = forward_outputs[i + 1]
        z_sampled = forward_outputs[i + 2]
        i += 3
        x_decoded_proba, x_sampled, x_target_NLL = forward_outputs[i:i+3]
        i += 3
        # See preset_model.py
        u_out, u_numerical_nll, u_categorical_nll, num_l1_error, acc = forward_outputs[i:i+5]
        assert i+5 == len(forward_outputs)

        return HierarchicalVAEOutputs(
            z_mu, z_var, z_sampled,
            x_decoded_proba, x_sampled, x_target_NLL,
            u_out, u_numerical_nll, u_categorical_nll, num_l1_error, acc
        )

    def latent_loss(self, ae_out: HierarchicalVAEOutputs, beta,
                    target_labels: Optional[torch.Tensor] = None, target_attributes: Optional[torch.Tensor] = None):
        """ Returns the non-normalized latent loss, and the same value multiplied by beta (for backprop) """
        # FIXME for hybrid training (parallel preset/audio VAEs), use a different beta for different batch items
        #    (if possible? not batch-reduced yet?)

        # Only the vanilla-VAE Dkl loss (with free bits) is available at the moment
        #     Dkl for a batch item is the sum of per-coordinates Dkls; we keep the per-coord Dkl values
        #     (reduction is done just before return)
        z_loss = utils.probability.standard_gaussian_dkl(
            ae_out.z_mu, ae_out.z_var, batch_reduction='none', coordinates_reduction='none')
        if not np.isclose(self._latent_free_bits, 0.0):
            # "Free bits" constraint is given as a "per-coordinate" min loss; however, it was intended to work with
            # groups of latent variables (see IAF, NeurIPS16)
            min_dkl = torch.tensor(self._latent_free_bits, dtype=ae_out.z_mu.dtype, device=ae_out.z_mu.device)
            z_loss = torch.maximum(z_loss, min_dkl)  # Free-bits constraint

        # Reduce loss, then Store it in the ae_out structure
        z_loss = torch.sum(z_loss, dim=1) if not self._normalize_latent_loss else torch.mean(z_loss, dim=1)
        z_loss = z_loss.mean()  # Average over the batch dimension (the only remaining at this point)
        ae_out.z_loss = z_loss

        # TODO ALSO IMPLEMENT LABEL-BASED extra regularization loss
        extra_reg_loss = torch.zeros_like(z_loss)
        next_z_coordinate = 0  # index of the next coordinate available for extra regularization
        if self._latent_attribute_based_regularization is not None:
            assert target_attributes is not None
            n_attributes = target_attributes.shape[1]
            # apply to groups of latents, or not
            z_chunks_size = self.encoder.z_chunks_size if self._latent_extra_regularization_expand_to_z_chunks else 1
            mu = ae_out.z_mu[:, next_z_coordinate:next_z_coordinate + (n_attributes * z_chunks_size)]
            next_z_coordinate += mu.shape[1]
            if z_chunks_size > 1:
                target_attributes = \
                    torch.flatten(target_attributes.unsqueeze(dim=2).expand(-1, -1, z_chunks_size), start_dim=1)
            if self._latent_attribute_based_regularization == "S2VAE":
                extra_reg_loss += model.loss.s2_vae_latent_loss(mu, target_attributes, self._normalize_latent_loss)
            elif self._latent_attribute_based_regularization == "ARVAE":
                extra_reg_loss += model.loss.ar_vae_latent_loss(
                    mu, target_attributes, self._normalize_latent_loss, self.latent_regularization_delta)
            else:
                raise NotImplementedError(self._latent_attribute_based_regularization)
        # TODO if label reg: maybe don't start from z coord 0

        return z_loss, z_loss * beta, extra_reg_loss * self.latent_regularization_gamma

    def latent_alignment_loss(self, ae_out_audio: HierarchicalVAEOutputs, ae_out_preset: HierarchicalVAEOutputs, beta):
        """ Returns a loss that should be minimized to better align two parallel VAEs (one preset-VAE, one audio-VAE),
            or returns 0.0 if self.preset_ae_method != 'aligned_vaes' """
        if self.preset_ae_method != 'aligned_vaes':
            assert self._alignment_criterion is None
            return 0.0
        else:  # FIXME don't flatten anything anymore - will be done already
            mu_audio = self.flatten_latent_values(ae_out_audio.z_mu)
            var_audio = self.flatten_latent_values(ae_out_audio.z_var)
            mu_preset = self.flatten_latent_values(ae_out_preset.z_mu)
            var_preset = self.flatten_latent_values(ae_out_preset.z_var)
            if self._alignment_criterion == 'kld':  # KLD ( q(z|preset) || q(z|audio) )
                loss = utils.probability.gaussian_dkl(mu_preset, var_preset, mu_audio, var_audio, reduction='mean')
            elif self._alignment_criterion == 'symmetric_kld':
                loss = utils.probability.symmetric_gaussian_dkl(
                    mu_preset, var_preset, mu_audio, var_audio, reduction='mean')
            # TODO implement 2-Wasserstein distance (and MMD?)
            else:
                raise NotImplementedError(self._alignment_criterion)
        return loss * beta

    def audio_vae_loss(self, audio_log_prob_loss, x_shape, ae_out: HierarchicalVAEOutputs):
        """
        Returns a total loss that corresponds to the ELBO if this VAE is a vanilla VAE with Dkl.

        :param audio_log_prob_loss: Mean-reduced log prob loss (averaged over all dimensions)
        :param x_shape: Shape of audio input tensors
        :param ae_out:
        """
        # Factorized distributions - we suppose that the independant log-probs were added
        # We don't consider the  beta factor for the latent loss (but z_loss must be average over the batch dim)
        x_data_dims = np.prod(np.asarray(x_shape[1:]))  # C spectrograms of size H x W
        return audio_log_prob_loss * x_data_dims + ae_out.z_loss

    def set_preset_decoder_scheduled_sampling_p(self, p: float):
        if self.decoder.preset_decoder is not None:  # Does not send a warning if preset decoder does not exist
            self.decoder.preset_decoder.child_decoder.scheduled_sampling_p = p

    def get_detailed_summary(self):
        sep_str = '************************************************************************************************\n'
        summary = sep_str + '********** ENCODER audio **********\n' + sep_str
        summary += str(self.encoder.get_audio_only_summary()) + '\n\n'
        if self.encoder.preset_encoder is not None:
            summary += sep_str + '********** ENCODER preset **********\n' + sep_str
            summary += str(self.encoder.preset_encoder.get_summary(self._input_audio_tensor_size[0])) + '\n\n'
        summary += sep_str + '********** ENCODER latent module **********\n' + sep_str
        summary += str(self.encoder.get_latent_module_summary()) + '\n\n'
        summary += sep_str + '********** DECODER latent module **********\n' + sep_str
        summary += str(self.decoder.get_latent_module_summary()) + '\n\n'
        summary += sep_str + '********** DECODER audio cells **********\n' + sep_str
        summary += str(self.decoder.get_audio_only_summary()) + '\n\n'
        if self.decoder.preset_decoder is not None:
            summary += sep_str + '********** DECODER preset **********\n' + sep_str
            summary += str(self.decoder.preset_decoder.get_summary()) + '\n\n'
        summary += sep_str + '********** FULL VAE SUMMARY **********\n' + sep_str
        if self._preset_helper is None:
            input_data = (torch.rand(self._input_audio_tensor_size) - 0.5, )
        else:
            input_u = self._preset_helper.get_null_learnable_preset(self._input_audio_tensor_size[0])
            input_data = (torch.rand(self._input_audio_tensor_size) - 0.5, input_u)
        summary += str(torchinfo.summary(
            self, input_data=input_data, depth=5, device=torch.device('cpu'), verbose=0,
            col_names=("input_size", "output_size", "num_params", "mult_adds"),
            row_settings=("depth", "var_names")
        ))
        # TODO also retrieve total number of mult-adds/item and parameters
        return summary


class AudioDecoder:
    def __init__(self, hierachical_vae: HierarchicalVAE):
        """ A simple wrapper class for a HierarchicalVAE instance to be used by an
         evaluation.interp.LatentInterpolation instance. """
        self._hierarchical_vae = hierachical_vae

    @property
    def dim_z(self):
        return self._hierarchical_vae.dim_z

    def generate_from_latent_vector(self, z: torch.Tensor):
        assert len(z.shape) == 2, "input z is expected to be a batch of vectors"
        # No preset: will return lots of None (or zeroes, maybe...)
        decoder_out = self._hierarchical_vae.decoder(z, x_target=None, u_target=None)
        return decoder_out[1]  # Return x_sampled only

