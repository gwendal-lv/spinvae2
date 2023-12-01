from typing import List, Sequence, Optional

import numpy as np
import torch
import torchinfo
import warnings
from torch import nn as nn
from torch.nn import functional as F

from data.preset2d import Preset2dHelper
from model.presetmodel import parse_preset_model_architecture, get_act, PresetEmbedding, get_transformer_act
from synth.dexedbase import DexedCharacteristics
from utils.probability import GaussianUnitVariance, DiscretizedLogisticMixture, SoftmaxNumerical


class PresetDecoder(nn.Module):
    def __init__(self, architecture: str,
                 dim_z: int,
                 hidden_size: int,
                 numerical_proba_distribution: str,
                 preset_helper: Preset2dHelper,
                 embedding: PresetEmbedding,
                 internal_dropout_p=0.0, cat_dropout_p=0.0,
                 label_smoothing=0.0, use_cross_entropy_weights=False,
                 params_loss_exclude_useless=True):
        """
        TODO DOC
        """
        super().__init__()
        self.arch = parse_preset_model_architecture(architecture)
        arch_args = self.arch['args']
        self.dim_z, self.hidden_size = dim_z, hidden_size
        self.preset_helper = preset_helper
        self.embedding = embedding
        self.seq_len = self.preset_helper.n_learnable_params
        # FIXME this should be a property and should depend on auto-encoding presets, or not
        self.params_loss_exclude_useless = params_loss_exclude_useless

        # Warning: PyTorch broadcasting:
        #   - tensor broadcasting for (basic?) tensor operations requires the *trailing* dimensions to be equal
        #   - tensor default "broadcasting" for masks uses the mask on the 1st dimensions of the tensor to be masked
        # Masks don't need to be on the same device as the tensor they are masking
        self.seq_numerical_items_bool_mask = self.preset_helper.matrix_numerical_bool_mask.clone()
        self.seq_categorical_items_bool_mask = self.preset_helper.matrix_categorical_bool_mask.clone()

        # B) Categorical/Numerical final modules and probability distributions (required for A))
        # In hidden layers, the transformer does: "position-wise FC feed-forward network", equivalent to
        #     two 1-kernel convolution with ReLU in-between.
        # Output transformations are linear (probably position-wise...) then softmax to get the next-token probs;
        #     equivalent to conv1d, kernel 1, without bias
        self.categorical_module = MultiCardinalCategories(
            self.hidden_size, self.preset_helper,
            dropout_p=cat_dropout_p, label_smoothing=label_smoothing,
            use_cross_entropy_weights=use_cross_entropy_weights
        )
        if numerical_proba_distribution == "gaussian_unitvariance":
            self.numerical_distrib = GaussianUnitVariance(
                mu_activation=nn.Hardtanh(0.0, 1.0), reduction='none'
            )
        elif numerical_proba_distribution.startswith("logistic_mixt"):
            numerical_proba_distribution = numerical_proba_distribution.replace("logistic_mixt", "")
            n_mix_components = int(numerical_proba_distribution[0])
            prob_mass_leakage = numerical_proba_distribution.endswith('_leak')
            self.numerical_distrib = DiscretizedLogisticMixture(
                n_mix_components, reduction='none', prob_mass_leakage=prob_mass_leakage
            )
        elif numerical_proba_distribution == "softmax":
            self.numerical_distrib = SoftmaxNumerical(
                self.preset_helper.matrix_numerical_rows_card, torch.float, reduction='none'
            )
        else:
            raise NotImplementedError("Unknown distribution '{}'".format(numerical_proba_distribution))
        # Bias seems appropriate to get means in [0, 1], also to get small scales for mixt of discretized logistics
        #   (will be useless, however, for mixture weights which are to be softmaxed)
        # TODO use a bigger linear, and mask outputs? (to have one linear / token?)
        self.numerical_distrib_linear = nn.Linear(self.hidden_size, self.numerical_distrib.num_parameters, bias=True)

        # A) Build the main network (uses some self. attributes)
        if self.arch['name'] == 'mlp':
            # Feed-forward decoder
            # MLP is mostly used as a baseline/debugging model - so it uses its own quite big 2048 hidden dim
            self.child_decoder = MlpDecoder(self, mlp_hidden_features=2048, dropout_p=internal_dropout_p)
        elif self.arch['name'] in ['lstm', 'gru']:
            self.child_decoder = RnnDecoder(self, cell_type=self.arch['name'])  # TODO dropout ctor arg
        elif self.arch['name'] == 'tfm':  # Transformer
            self.child_decoder = TransformerDecoder(self, dropout_p=internal_dropout_p)  # TODO args: n_head, ...
        else:
            self.child_decoder: Optional[ChildDecoderBase] = None
            raise NotImplementedError("Preset architecture {} not implemented".format(self.arch['name']))

    def _move_masks_to(self, device):
        if self.seq_numerical_items_bool_mask.device != device:
            self.seq_numerical_items_bool_mask = self.seq_numerical_items_bool_mask.to(device)
            self.seq_categorical_items_bool_mask = self.seq_categorical_items_bool_mask.to(device)

    def forward(self, z_sampled: torch.Tensor, u_target: Optional[torch.Tensor] = None):
        """

        :param z_sampled: 1d sampled vectors
        :param u_target: Input preset, sequence-like (expected shape: N x n_synth_presets x 3)
        :return:
        """
        assert len(z_sampled.shape) == 2, "z must be a minibatch of 1d vectors"
        self._move_masks_to(z_sampled.device)

        # ----- A) Apply the "main" feed-forward or recurrent / sequential network -----
        # TODO handle u_target=None
        u_out, u_categorical_nll, u_categorical_samples, u_numerical_nll, u_numerical_samples \
            = self.child_decoder(z_sampled, u_target)

        # ----- B) Compute metrics (quite easy, we can do it now) -----
        if u_target is not None:  # might be None if we only want to sample a preset from z
            # we don't return separate metric for teacher-forcing
            #    -> because teacher-forcing will be activated or not from the owner of this module
            with torch.no_grad():
                num_l1_error = torch.mean(
                    torch.abs(u_target[:, self.seq_numerical_items_bool_mask, 1] - u_numerical_samples), dim=1
                )
                acc = torch.eq(u_target[:, self.seq_categorical_items_bool_mask, 0].long(), u_categorical_samples)
                acc = acc.count_nonzero(dim=1) / acc.shape[1]

            # TODO set the NLL of "useless params" (useless as in the target preset) to zero
            # FIXME don't do this when auto-encoding a preset (only during auto synth prog from audio)
            if self.params_loss_exclude_useless:
                u_categorical_nll, u_numerical_nll = self._remove_useless_loss(u_target, u_categorical_nll, u_numerical_nll)

            # sum NLLs and divide by total sequence length (num of params) - keep batch dimension (if multi-GPU)
            u_numerical_nll = u_numerical_nll.sum(dim=1) / self.seq_len
            u_categorical_nll = u_categorical_nll.sum(dim=1) / self.seq_len
        else:
            num_l1_error, acc = None, None

        return u_out, u_numerical_nll, u_categorical_nll, num_l1_error, acc

    def _remove_useless_loss(self, u_target, u_categorical_nll, u_numerical_nll):
        """ If a DX7 operator has a null volume, the loss corresponding to parameters of this operator is set to zero.
        This aims at reducing the amount of noise that is back-propagated into the network.
        """
        if self.preset_helper.synth_name.lower() == "dexed":
            with torch.no_grad():
                # get index of volume parameters
                vst_output_level_indices = DexedCharacteristics.get_op_output_level_indices()
                # retrieve the matrix of numerical values, get a bool matrix of zero-volume operators
                target_output_level_rows = [self.preset_helper._vst_idx_to_matrix_row[idx]
                                            for idx in vst_output_level_indices]
                target_output_levels = u_target[:, target_output_level_rows, 1].detach().cpu()  # Faster on CPU?
                is_zero_volume_op = torch.isclose(target_output_levels, torch.zeros_like(target_output_levels))
                # only check items (batch elements) which have at least 1 null volume
                zero_volume_batch_indices = torch.nonzero(torch.any(is_zero_volume_op, dim=1)).squeeze(dim=1)
            for batch_i in zero_volume_batch_indices:
                zero_volume_ops = torch.nonzero(is_zero_volume_op[batch_i]).squeeze(dim=1)
                for op_i in zero_volume_ops:  # Apply mask pre-computed by the preset helper instance
                    u_categorical_nll[batch_i, self.preset_helper.dexed_operators_categorical_bool_masks[op_i]] = 0.0
                    u_numerical_nll[batch_i, self.preset_helper.dexed_operators_numerical_bool_masks[op_i]] = 0.0
        else:
            raise NotImplementedError("'Useless params' can be identified for Dexed presets only.")
        return u_categorical_nll, u_numerical_nll

    def get_summary(self, device='cpu'):
        N, device = 1, torch.device(device)
        input_z = torch.rand((N, self.dim_z)).to(device) - 0.5
        input_u = self.preset_helper.get_null_learnable_preset(N).to(device)
        was_training = self.training
        if self.arch['name'] == 'tfm':
            # Recursive inference makes the transformer summary absolutely impossible to read - force use training mode
            self.train()
            summary_mode = "train"
        else:
            summary_mode = "eval"
        with torch.no_grad():
            summary = torchinfo.summary(
                self, input_data=(input_z, input_u), mode=summary_mode,
                depth=6, verbose=0, device=device,
                col_names=("input_size", "output_size", "num_params", "mult_adds"),
                row_settings=("depth", "var_names")
            )
        self.train(mode=was_training)
        return summary



class ChildDecoderBase(nn.Module):
    def __init__(self, parent_dec: PresetDecoder, dropout_p=0.0):
        """
        Base class for any "sub-decoder" child (e.g. Mlp, RNN, Transformer) of a PresetDecoder instance.
        This allows to easily share useful instances and information with all children.
        """
        super().__init__()
        self.seq_len = parent_dec.seq_len
        self.dim_z = parent_dec.dim_z
        self.hidden_size = parent_dec.hidden_size
        self.preset_helper = parent_dec.preset_helper
        self.embedding = parent_dec.embedding
        self.arch_args = parent_dec.arch['args']
        self.n_layers = parent_dec.arch['n_layers']
        self._dropout_p = dropout_p

        self.categorical_module = parent_dec.categorical_module
        self.seq_categorical_items_bool_mask = parent_dec.seq_categorical_items_bool_mask

        self.numerical_distrib_linear = parent_dec.numerical_distrib_linear
        self.numerical_distrib = parent_dec.numerical_distrib
        self.seq_numerical_items_bool_mask = parent_dec.seq_numerical_items_bool_mask
        # Discretized logistics require to know the cardinal of the set of values for each token
        self._numerical_tokens_card = torch.tensor(self.preset_helper.matrix_numerical_rows_card, dtype=torch.long)
        self.is_type_numerical = self.preset_helper.is_type_numerical

        self.scheduled_sampling_p = 0.0  # Probability to use own output (corresponds to 1.0 - teacher_forcing_p)

    def get_numerical_tokens_card(self, device):
        if self._numerical_tokens_card.device != device:
            self._numerical_tokens_card = self._numerical_tokens_card.to(device)
        return self._numerical_tokens_card

    def get_expanded_numerical_tokens_card(self, batch_size: int, device):
        numerical_tokens_card = self.get_numerical_tokens_card(device)
        return numerical_tokens_card.view(1, 1, numerical_tokens_card.shape[0]).expand(batch_size, 1, -1)

    def compute_full_sequence_samples_and_losses(self, u_hidden,
                                                 u_target: Optional[torch.Tensor] = None, sample_only=False):
        """
        Compute parameters of probability distributions, compute NLLs (all at once)

        :param u_hidden: the hidden sequence representation of the preset - shape N x L x Hembed
        :param sample_only: If True, the NLLs won't be computed
        """
        u_out = self.preset_helper.get_null_learnable_preset(u_hidden.shape[0]).to(u_hidden.device)
        # --- Categorical distribution(s) ---
        u_categorical_logits, u_categorical_nll, u_categorical_samples = self.categorical_module.forward_full_sequence(
            u_hidden[:, self.seq_categorical_items_bool_mask, :].transpose(2, 1),  # FIXME seq dim should NOT be last
            u_target[:, self.seq_categorical_items_bool_mask, 0] if u_target is not None else None,
            sample_only=sample_only
        )
        # FIXME this masked assignation breaks when using deterministic algorithms (pytorch 1.10)
        u_out[:, self.seq_categorical_items_bool_mask, 0] = u_categorical_samples.float()  # TODO reproduce minimal bug?
        # --- Numerical distribution (s) ---
        # FIXME different distributions for parameters w/ a different cardinal??
        #    Gaussian does not care, Discretized Logistics will handle the cardinal
        numerical_distrib_params = self.numerical_distrib_linear(u_hidden[:, self.seq_numerical_items_bool_mask, :])
        # Set sequence dim last for the probability distribution (channels: distrib params)
        numerical_distrib_params = self.numerical_distrib.apply_activations(numerical_distrib_params.transpose(2, 1))
        if not sample_only:
            u_numerical_nll = self.numerical_distrib.NLL(
                numerical_distrib_params,
                u_target[:, self.seq_numerical_items_bool_mask, 1:2].transpose(2, 1),
                self.get_expanded_numerical_tokens_card(u_target.shape[0], u_target.device)  # for discrete logistics
            )
            # These are NLLs and samples for a "single-channel" distribution -> squeeze for consistency vs. categorical
            u_numerical_nll = torch.squeeze(u_numerical_nll, dim=1)  # don't squeeze batch dim
        else:
            u_numerical_nll: Optional[torch.Tensor] = None
        with torch.no_grad():
            u_numerical_samples = self.numerical_distrib.get_mode(
                numerical_distrib_params, self.get_numerical_tokens_card(u_hidden.device))
            # Squeeze singleton "channel" dimension
            u_numerical_samples = torch.squeeze(u_numerical_samples, dim=1)
        u_out[:, self.seq_numerical_items_bool_mask, 1] = u_numerical_samples

        return u_out, u_categorical_nll, u_categorical_samples, u_numerical_nll, u_numerical_samples

    def get_init_u_out_and_nlls(self, N, device):
        u_out = self.preset_helper.get_null_learnable_preset(N).to(device)
        u_categorical_nll = torch.zeros((N, self.preset_helper.n_learnable_categorical_params), device=device)
        u_numerical_nll = torch.zeros((N, self.preset_helper.n_learnable_numerical_params), device=device)
        return u_out, u_categorical_nll, u_numerical_nll

    def compute_single_token_sample_and_loss(
            self,
            token_hidden, target_token,
            u_out, u_categorical_nll, u_numerical_nll,
            categorical_idx, numerical_idx,
            sample_only=False
    ):
        """ Computes the loss about a single token (either categorial or numerical), samples from it, and
        stores values into the proper structures (u_out, indexes, and NLL numerical or categorical).

        :param token_hidden: (N x 1 x hidden_size) output from the last hidden layer
        :param target_token: (N x 1 x 3) target (can be None if sample_only is True)
        """
        N, device = token_hidden.shape[0], token_hidden.device
        t = categorical_idx + numerical_idx
        # TODO implement sample_only w/ null target_token
        if target_token is None or sample_only:
            raise NotImplementedError()
        type_class = int(target_token[0, 0, 2].item())
        # compute NLLs and sample (and embed the sample)  TODO no_grad, no NLL
        #    we could compute NLLs only once at the end, but this section should be run with no_grad() context
        #    (optimization left for future works)
        if self.is_type_numerical[type_class]:
            # All distributions requires seq dim to be last (each token corresponds to "a pixel")
            #    FIXME different distribs for different discrete cards
            numerical_distrib_params = self.numerical_distrib_linear(token_hidden).transpose(1, 2)
            numerical_distrib_params = self.numerical_distrib.apply_activations(numerical_distrib_params)
            with torch.no_grad():
                samples = self.numerical_distrib.get_mode(
                    numerical_distrib_params,
                    self.get_numerical_tokens_card(device)[numerical_idx:numerical_idx+1]  # unsqueezed tensor
                ).view((N,))
            u_out[:, t, 1] = samples
            u_numerical_nll[:, numerical_idx] = self.numerical_distrib.NLL(
                numerical_distrib_params,
                target_token[:, :, 1:2].transpose(2, 1),
                self.get_expanded_numerical_tokens_card(N, device)[:, :, numerical_idx:numerical_idx+1]
            ).view((N,))
            numerical_idx += 1
        else:
            logits, ce_nll, samples = self.categorical_module.forward_single_token(
                token_hidden, target_token[:, 0, 0].long(), type_class
            )
            u_out[:, t, 0] = samples.float()
            u_categorical_nll[:, categorical_idx] = ce_nll
            categorical_idx += 1
        return u_out, u_categorical_nll, u_numerical_nll, categorical_idx, numerical_idx


class MlpDecoder(ChildDecoderBase):
    def __init__(self, parent_dec: PresetDecoder, mlp_hidden_features=2048, sequence_dim_last=True, dropout_p=0.0):
        """
        Simple module that reshapes the output from an MLP network into a 2D feature map with seq_len as
        its first dimension (number of channels will be inferred automatically), and applies a 1x1 conv
        to increase the number of channels to seq_hidden_dim.

        :param sequence_dim_last: If True, the sequence dimension will be the last dimension (use this when
            the output of this module is used by a CNN). If False, the features (channels) dimension will be last
            (for use as RNN input).
        """
        super().__init__(parent_dec, dropout_p)
        # self.hidden_size is NOT the number of hidden neurons in this MLP (it's the feature vector size, for
        #  recurrent networks only - not applicable to this MLP).
        self.seq_hidden_dim = self.hidden_size
        if self.arch_args['ff']:
            warnings.warn("Useless '_ff' arch arg: MLP decoder is always feed-forward.")
        if self.arch_args['posenc']:
            warnings.warn("Cannot use positional embeddings with an MLP network - ignored")
        if self.arch_args['memmlp']:
            warnings.warn("'_memmlp' arch arg can be used with a Transformer decoder only - ignored")

        self.mlp = nn.Sequential()
        n_pre_out_features = self.seq_len * (mlp_hidden_features // self.seq_len)
        for l in range(0, self.n_layers):
            if l > 0:
                if l < self.n_layers - 1:  # No norm before the last layer
                    if self.arch_args['bn']:
                        self.mlp.add_module('bn{}'.format(l), nn.BatchNorm1d(mlp_hidden_features))
                    if self.arch_args['ln']:
                        raise NotImplementedError()
                self.mlp.add_module('act{}'.format(l), get_act(self.arch_args))
                if self._dropout_p > 0.0:
                    self.mlp.add_module('drop{}'.format(l), nn.Dropout(self._dropout_p))
            n_in_features = mlp_hidden_features if (l > 0) else self.dim_z
            n_out_features = mlp_hidden_features if (l < self.n_layers - 1) else n_pre_out_features
            self.mlp.add_module('fc{}'.format(l), nn.Linear(n_in_features, n_out_features))

        # Last layer should output a 1d tensor that can be reshaped as a "sequence-like" 2D tensor.
        #    This would represent an overly huge FC layer.... so it's done in 2 steps
        self.in_channels = n_pre_out_features // self.seq_len
        self.conv = nn.Sequential(
            get_act(self.arch_args),
            nn.Conv1d(self.in_channels, self.seq_hidden_dim, 1)
        )
        self.sequence_dim_last = sequence_dim_last

    def forward(self, z: torch.Tensor, u_target: Optional[torch.Tensor] = None):
        # Apply the feed-forward MLP
        u_hidden = self.mlp(z).view(-1, self.in_channels, self.seq_len)
        u_hidden = self.conv(u_hidden)  # After this conv, sequence dim ("time" or "step" dimension) is last
        # Full-sequence loss (shared with the transformer in training mode) - embed dim last
        return self.compute_full_sequence_samples_and_losses(
            u_hidden.transpose(1, 2), u_target, sample_only=(u_target is None))


class RnnDecoder(ChildDecoderBase):
    def __init__(self, parent_dec: PresetDecoder, cell_type='lstm'):
        """
        Decoder based on a single-way RNN (LSTM)
        """
        super().__init__(parent_dec)
        if cell_type != 'lstm':
            raise NotImplementedError()
        self.autoregressive = not self.arch_args['ff']
        self.bidirectional = False  # TODO use arch arg instead of (not self.autoregressive). Disabled, worse results
        # Force use custom input tokens for non-AR (LSTMs are very bad at using tfm-like positional encodings)
        if not self.autoregressive:
            max_norm = np.sqrt(self.hidden_size) if self.arch_args['embednorm'] else None
            self.custom_in_tokens = nn.Embedding(self.seq_len, self.hidden_size, max_norm=max_norm)
        else:
            self.custom_in_tokens = None
        if self.arch_args['memmlp']:
            warnings.warn("'_memmlp' arch arg can be used with a Transformer decoder only - ignored")
        # Network to use the flattened z as hidden state
        # We use z sampled values as much as possible (in c0 then in h0) not to add another gradient to the
        # computational path. Remaining 'empty spaces' in h0 are filled using an MLP.
        if 2 * self.hidden_size <= self.dim_z:
            raise NotImplementedError()
        self.latent_expand_fc = nn.Linear(self.dim_z, self.n_layers * (2 * self.hidden_size - self.dim_z))
        self.lstm = nn.LSTM(
            self.hidden_size, self.hidden_size, self.n_layers,  # TODO only if not AR
            batch_first=True, dropout=self._dropout_p,
            bidirectional=self.bidirectional, proj_size=((self.hidden_size//2) if self.bidirectional else 0)
        )

    def forward(self, z: torch.Tensor, u_target: Optional[torch.Tensor] = None):
        N, device = z.shape[0], z.device
        # Compute h0 and c0 hidden states for each layer - expected shapes (num_layers, N, hidden_size)
        z_expansion_split = torch.chunk(self.latent_expand_fc(z), self.n_layers, dim=1)
        # We fill this "merged" tensor, then we'll split it into c0 (first) and h0
        c0_h0 = torch.empty((self.n_layers, N, 2 * self.hidden_size), device=device)
        c0_h0[:, :, 0:z.shape[1]] = z  # use broadcasting
        for l in range(self.n_layers):
            c0_h0[l, :, z.shape[1]:] = z_expansion_split[l]
        c0, h0 = torch.chunk(c0_h0, 2, dim=2)
        # Bidirectional: we have to reshape... without mixing batch and layer/hidden dimensions (view is risky)
        if self.bidirectional:
            h0 = torch.cat(torch.chunk(h0, 2, dim=2), dim=0)
            c0 = torch.cat([c0, c0], dim=0)  # Same long-term memory provided for both directions
        c0, h0 = c0.contiguous(), h0.contiguous()
        # apply tanh to h0; not to c0? c_t values are not bounded in -1, +1 by LSTM cells
        h0 = torch.tanh(h0)

        # Single-call computation
        if self.training or not self.autoregressive:
            # Compute input embeddings
            if not self.autoregressive:
                input_embed = self.custom_in_tokens(torch.arange(0, self.seq_len, dtype=torch.long).to(device))
                input_embed = input_embed.unsqueeze(dim=0).expand(N, -1, -1)
            else:
                assert u_target is not None, "AR RNN training requires a target to be trained on"
                if self.scheduled_sampling_p > 0.0:
                    raise NotImplementedError("Scheduled sampling not implemented with RNNs")
                else:  # No scheduled sampling - don't use the RNN's own outputs, use inputs shifted right
                    input_embed = self.embedding(u_target, start_token=True)[:, 0:self.seq_len, :]  # Discard last
            # Then apply the RNN in a single call
            lstm_output, (h_n, c_n) = self.lstm(input_embed, (h0, c0))
            # Compute output params, and losses if a target was given
            return self.compute_full_sequence_samples_and_losses(lstm_output, u_target, sample_only=(u_target is None))

        # Item-per-item computation (token-by-token auto-regressive computation)
        else:
            raise AssertionError()  # FIXME deprecated auto-regressive mode - reimplement properly
            # Prepare data structures to store all results (will be filled token-by-token)
            u_out, u_categorical_nll, u_numerical_nll = self.get_init_u_out_and_nlls(N, device)
            numerical_idx, categorical_idx = 0, 0
            # Input FIXME don't always use this (u_target can be None)
            embed_target = self.embedding(u_target, start_token=True)  # Inputs are shifted right
            # apply LSTM, token-by-token
            input_embed = embed_target[:, 0:1, :]  # Initial: Start token with zeros
            h_t, c_t = h0, c0
            for t in range(u_target.shape[1]):
                output, (h_t, c_t) = self.lstm(input_embed, (h_t, c_t))
                u_out, u_categorical_nll, u_numerical_nll, categorical_idx, numerical_idx \
                    = self.compute_single_token_sample_and_loss(
                        output, u_target[:, t:t+1, :],
                        u_out, u_categorical_nll, u_numerical_nll, categorical_idx, numerical_idx
                    )
                # Compute next embedding
                if self.training and True:  # TODO teacher forcing proba
                    warnings.warn("Scheduled sampling not implemented")
                    input_embed = self.embedding.forward_single_token(u_target[:, t:t+1, :])  # FIXME if u_target is None ???
                else:  # Eval or no teacher forcing: use the net's own output
                    input_embed = self.embedding.forward_single_token(u_out[:, t:t+1, :])  # expected in shape: N x 1 x 3
            # After the whole sequence has been processed: retrieve numerical and categorical samples from u_out
            u_categorical_samples = u_out[:, self.seq_categorical_items_bool_mask, 0].long()
            u_numerical_samples = u_out[:, self.seq_numerical_items_bool_mask, 1]
            return u_out, u_categorical_nll, u_categorical_samples, u_numerical_nll, u_numerical_samples


class TransformerDecoder(ChildDecoderBase):
    def __init__(self, parent_dec: PresetDecoder, n_head=None, dropout_p=0.0):
        """
        TODO doc
        """
        super().__init__(parent_dec, dropout_p)
        if n_head is None:
            assert self.hidden_size % 64 == 0
            self.n_head = self.hidden_size // 64
        else:
            self.n_head = n_head
        self.autoregressive = not self.arch_args['ff']

        # Maybe use an MLP to get 1 more memory token? (and fill some incomplete token, if dim_z < hidden_size)
        if self.arch_args['memmlp']:
            self.n_raw_memory_tokens = int(np.ceil(self.dim_z / self.hidden_size))  # "partially raw"
            # We always add at least one full MLP-computed extra token
            self.n_extra_tokens_dims = (self.n_raw_memory_tokens + 1) * self.hidden_size - self.dim_z
            self.input_memory_mlp = nn.Sequential(
                nn.Linear(self.dim_z, self.n_extra_tokens_dims),
                nn.ELU(),
                nn.Linear(self.n_extra_tokens_dims, self.n_extra_tokens_dims)
            )
        # Otherwise, z is directly used as single-token memory
        #     Not using an MLP requires to have a sufficiently large dim_z (to build at least a single full token)
        else:
            # 2nd condition is redundant but improves clarity
            assert self.dim_z % self.hidden_size == 0 and self.dim_z >= self.hidden_size
            self.n_raw_memory_tokens = self.dim_z // self.hidden_size
            self.input_memory_mlp = None

        # Maybe use custom input tokens (added to pos encodings) for non-AR
        if self.arch_args['fftoken']:
            assert not self.autoregressive, "_fftoken arg must be used with a non-AR transformer decoder"
            max_norm = np.sqrt(self.hidden_size) if self.arch_args['embednorm'] else None
            self.custom_in_tokens = nn.Embedding(self.seq_len, self.hidden_size, max_norm=max_norm)
        else:
            self.custom_in_tokens = None

        # Transformer decoder
        # Final (3rd) dropout could impair regression, but this does not seem to happen in practice
        # TODO maybe last layer should have no output norm
        tfm_layer = nn.TransformerDecoderLayer(
            self.hidden_size, self.n_head,  # each head's embed dim will be: self.hidden_size // num_heads
            dim_feedforward=self.hidden_size * 4, batch_first=True, dropout=self._dropout_p,
            activation=get_transformer_act(self.arch_args),
            norm_first=self.arch_args['normfirst'],
        )
        self.tfm = nn.TransformerDecoder(tfm_layer, self.n_layers)  # opt norm: between blocks? (default: None)
        self.subsequent_mask = nn.Transformer.generate_square_subsequent_mask(self.seq_len)

    def forward(self, z: torch.Tensor, u_target: Optional[torch.Tensor] = None):
        N, device = z.shape[0], z.device  # minibatch size, current device

        # Build memory from z
        # Simplest memory: 1-token sequence (use latent space of size hidden_dim not to use this linear)
        # Should we compute some extra memory token(s)?
        #     - ICCV21 "3D human motion transformer VAE" seems to use the raw latent vector as a single memory token
        #          https://github.com/Mathux/ACTOR
        #     - or: input memory mlp (small general improvement...)
        #        - will append another memory token (we'll always use the raw latent token(s))
        if self.input_memory_mlp is None:  # hidden size here is always <= dim_z
            memory_in = z.view(N, self.n_raw_memory_tokens, self.hidden_size)
        else:
            extra_memory = self.input_memory_mlp(z)
            memory_in = torch.cat((z, extra_memory), dim=1)
            # unflatten at the very end only - we always have at least 1 full MLP-computed extra token
            memory_in = memory_in.view(N, self.n_raw_memory_tokens + 1, self.hidden_size)

        ar_forward_mask = self.subsequent_mask.to(device)
        # Different training and eval procedures: parallel training, sequential evaluation
        #   (however if non-AR: we always use the "training" parallel procedure, even for validation)
        if self.training or not self.autoregressive:  # Parallel computation
            # Usual AR transformer: Get embeddings (shifted right) w/ start token, discard the last token
            if self.autoregressive:
                u_input_embeds = self.embedding(u_target, start_token=True)[:, 0:self.seq_len, :]
            # non-AR transformer: pos encoding input only
            else:
                u_input_embeds = torch.unsqueeze(self.embedding.pos_embed_L.to(device), dim=0)  # Add batch dimension
                u_input_embeds = u_input_embeds.expand(N, -1, -1).clone()
                if self.custom_in_tokens is not None:
                    custom_tokens = self.custom_in_tokens(torch.arange(0, self.seq_len, dtype=torch.long).to(device))
                    u_input_embeds += custom_tokens.unsqueeze(dim=0).expand(N, -1, -1)
                if self.scheduled_sampling_p > 0.0:
                    warnings.warn("Scheduled sampling probability > 0.0 can't be applied w/ this non-AR decoder.")

            # Some details about PyTorch's standard Transformer decoding layers and modules:
            #     (see torch/nn/modules/transformer.py, line 460 for pytorch 1.10)
            # Target data corresponds to the decoder's input embeddings (i.e. the target in teacher-forcing training)
            #    -> the first mha layer computes Q, K, V from x, x, x   (attention mask: tgt_mask)
            #       where x is the target for the very first layer, and is the target+residuals for hidden layers
            # Memory data corresponds to hidden states from the encoder (or extracted from the latent space....)
            #    -> memory data does not contain some K and V, but can be turned into Keys and Values using
            #       the decoder's own Wv and Wk matrices.
            #    -> the second mha layer computes Q, K, V from x, memory, memory
            #       (attention mask: memory_mask, don't use any mask to use all information from the input seq)

            # Training might require 2 (or more) passes if we follow a scheduled sampling procedure.
            # Don't backprop the first pass
            #    NeurIPS15, for RNNs: (Sequential) Scheduled Sampling https://arxiv.org/abs/1506.03099?context=cs
            #    arxiv19 (ICLR20 rejected) Parallel Scheduled Sampling   https://arxiv.org/abs/1906.04331
            #    ACL student workshop Sched Sampling for Transformers https://arxiv.org/abs/1906.07651
            if self.scheduled_sampling_p > 0.0 and self.autoregressive:
                with torch.no_grad():  # 1st pass without gradient
                    tfm_out_hidden = self.tfm(u_input_embeds, memory_in, tgt_mask=ar_forward_mask)
                    # we need to sample from tfm_out (which contains hidden values only), but don't need NLLs
                    out_tokens_1st_pass = self.compute_full_sequence_samples_and_losses(
                        tfm_out_hidden, u_target, sample_only=True)[0]  # keep u_out only
                    # Build new random partially-AR input
                    token_feedback_mask = torch.empty(out_tokens_1st_pass.shape[0:2], dtype=torch.bool, device=device)
                    token_feedback_mask.bernoulli_(self.scheduled_sampling_p)
                    sched_sampling_tokens = u_target.clone()
                    # Mask is 'broadcast' to the first dimensions
                    sched_sampling_tokens[token_feedback_mask] = out_tokens_1st_pass[token_feedback_mask]
                    in_embeds_2nd_pass = self.embedding(sched_sampling_tokens, start_token=True)[:, 0:self.seq_len, :]
                # 2nd pass with gradient
                tfm_out_hidden = self.tfm(in_embeds_2nd_pass, memory_in, tgt_mask=ar_forward_mask)
            # No scheduled sampling (or non-AR): Single pass w/ gradients
            else:
                # Don't use mask for feed-forward (non-AR) transformer (because input is pos encodings only)
                tfm_out_hidden = self.tfm(u_input_embeds, memory_in,
                                          tgt_mask=(ar_forward_mask if self.autoregressive else None))

            # Compute logits and losses - all at once (shared with the MLP decoder)
            return self.compute_full_sequence_samples_and_losses(
                tfm_out_hidden, u_target, sample_only=(u_target is None))

        else:  # Eval mode - AR forward inference (feed-forward, non-AR case is handled in the previous 'if' block)
            sample_only = (u_target is None)
            # Prepare data structures to store all results (will be filled token-by-token)
            u_out, u_categorical_nll, u_numerical_nll = self.get_init_u_out_and_nlls(N, device)
            if sample_only:
                u_numerical_nll: Optional[torch.Tensor] = None
                u_categorical_nll: Optional[torch.Tensor] = None
            numerical_idx, categorical_idx = 0, 0
            # Default null embeddings (we don't need to pre-embed null values): will contain positional embeddings only
            u_input_feedback_embeds = torch.zeros((N, self.seq_len, self.hidden_size), device=device)
            u_input_feedback_embeds[:, 0:1, :] = self.embedding.get_start_token(device, batch_dim=False)
            u_input_feedback_embeds += self.embedding.pos_embed_L

            for t in range(self.seq_len):
                # use the TFM on a reduced set of input embeds
                #  Sub-optimal (default pytorch) implementation: we'll recompute the same Q, K, V multiple times....
                tfm_out_hidden = self.tfm(
                    u_input_feedback_embeds[:, 0:t+1, :],
                    memory_in[:, 0:t+1, :],
                    tgt_mask=ar_forward_mask[0:t+1, 0:t+1]
                )
                u_out, u_categorical_nll, u_numerical_nll, categorical_idx, numerical_idx \
                    = self.compute_single_token_sample_and_loss(
                        tfm_out_hidden[:, t:t+1, :], (u_target[:, t:t + 1, :] if u_target is not None else None),
                        u_out, u_categorical_nll, u_numerical_nll, categorical_idx, numerical_idx
                )
                # add the next embedding to its (previously computed) positional embedding
                next_embed = self.embedding.forward_single_token(u_out[:, t:t + 1, :])  # expected in shape: N x 1 x 3
                u_input_feedback_embeds[:, t+1:t+2, :] += next_embed

            # Retrieve numerical and categorical samples from u_out - and return everything
            u_categorical_samples = u_out[:, self.seq_categorical_items_bool_mask, 0].long()
            u_numerical_samples = u_out[:, self.seq_numerical_items_bool_mask, 1]
            return u_out, u_categorical_nll, u_categorical_samples, u_numerical_nll, u_numerical_samples


class MultiCardinalCategories(nn.Module):
    def __init__(self, hidden_size: int, preset_helper: Preset2dHelper,
                 dropout_p=0.0, label_smoothing=0.0, use_cross_entropy_weights=False):
        """
        This class is able to compute the categorical logits using several groups, and to compute the CE loss
        for each synth param of each group.
        Each group contains categorical synth params with the same number of classes (cardinal of the set of values).
        Each group has a different number of logits (using only 1 group with masking could lead to a biased
         preference towards the first logits).
        """
        super().__init__()
        self.preset_helper = preset_helper
        self.label_smoothing = label_smoothing
        self.use_ce_weights = use_cross_entropy_weights
        # 1 conv1d for each categorical group - 4 groups instead of 1 (the overhead should be acceptable...)
        self.categorical_distribs_conv1d = dict()
        self.cross_entropy_weights = dict()
        for card, mask in self.preset_helper.categorical_groups_submatrix_bool_masks.items():
            self.categorical_distribs_conv1d[card] = nn.Sequential(
                nn.Dropout(dropout_p),
                nn.Conv1d(hidden_size, card, 1, bias=False)
            )
            if self.use_ce_weights:
                # Compute some "merged class counts" or "weights". We don't instantiate a CE loss because we don't know
                # the device yet (and Pytorch's CE class is a basic wrapper that calls F.cross_entropy)
                class_counts = self.preset_helper.categorical_groups_class_samples_counts[card] + 10  # 10 is an "epsilon"
                weights = 1.0 - (class_counts / class_counts.sum())
                # average weight will be 1.0
                self.cross_entropy_weights[card] = torch.tensor(weights / weights.mean(), dtype=torch.float)

        # This dict allows nn parameters to be registered by PyTorch
        self._mods_dict = nn.ModuleDict(
            {'card{}'.format(card): m for card, m in self.categorical_distribs_conv1d.items()}
        )
        # Constant sizes
        self.Lc = self.preset_helper.n_learnable_categorical_params

    def get_ce_weights(self, card: int, device):
        if self.use_ce_weights and self.training:
            return self.cross_entropy_weights[card].to(device)
        else:
            return None

    def forward(self):
        raise NotImplementedError("Use forward_full_sequence or forward_item.")

    def forward_full_sequence(self, u_categorical_last_hidden,
                              u_target: Optional[torch.Tensor] = None, sample_only=False):
        """

        :param u_categorical_last_hidden: Tensor of shape N x hidden_size x Lc
            where Lc is the total number of categorical output variables (synth params)
        :param u_target: target tensor, shape N x Lc (can be None if sample_only)
        :param sample_only: If True, don't compute the cross-entropy (NLL)
        :return:    out_logits (dict, different shapes),
                    out_ce (shape N x Lc) (is a NLL),
                    samples_categories (shape N x Lc)
        """
        if u_target is not None:
            u_target = u_target.long()
        out_logits = dict()
        out_ce = torch.empty((u_categorical_last_hidden.shape[0], self.Lc), device=u_categorical_last_hidden.device)
        sampled_categories = torch.empty(
            (u_categorical_last_hidden.shape[0], self.Lc), dtype=torch.long, device=u_categorical_last_hidden.device)
        for card, module in self.categorical_distribs_conv1d.items():
            mask = self.preset_helper.categorical_groups_submatrix_bool_masks[card]
            logits = module(u_categorical_last_hidden[:, :, mask])
            # logits is the output of a conv-like module, so the sequence dim is dim=2, and
            # class probabilities are dim=1 (which is expected when calling F.cross_entropy)
            sampled_categories[:, mask] = torch.argmax(logits, dim=1, keepdim=False)
            if not sample_only:
                # Don't turn off label smoothing during validation (we don't want validation over-confidence either)
                # However, during validation, we should use the training weights only
                weights = self.get_ce_weights(card, u_target.device)
                out_ce[:, mask] = F.cross_entropy(
                    logits, u_target[:, mask],
                    reduction='none', label_smoothing=self.label_smoothing, weight=weights
                )
            out_logits[card] = logits
        return out_logits, out_ce if not sample_only else None, sampled_categories

    def forward_single_token(self, u_token_hidden, u_target_classes, type_class: int):
        """
        :param u_categorical_last_hidden: Tensor of shape N x 1 x hidden_size
        :returns logits (shape N), out_ce (shape N), samples (shape N)
        """
        # TODO allow None target (sampling only)
        card_group = self.preset_helper.param_type_to_cardinality[type_class]
        logits = self.categorical_distribs_conv1d[card_group](u_token_hidden.transpose(1, 2))
        logits = torch.squeeze(logits, 2)
        out_ce = F.cross_entropy(
            logits, u_target_classes,
            reduction='none', label_smoothing=self.label_smoothing,
            weight=self.get_ce_weights(card_group, u_token_hidden.device)
        )
        return logits, out_ce, torch.argmax(logits, dim=1, keepdim=False)

