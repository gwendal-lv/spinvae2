from typing import Dict, Any

import torch
from torch import nn
from torch.nn import functional as F

import utils.exception
from model.hierarchicalvae import HierarchicalVAE


def process_minibatch(
        ae_model: HierarchicalVAE, ae_model_parallel: nn.DataParallel, main_device,
        x_in, v_in, uid, notes, labels, attributes,
        epoch: int, scalars: Dict[str, Any], super_metrics: Dict[str, Any]
):
    training = ae_model.training
    suffix = "/Train" if training else "/Valid"

    if training:
        ae_model.optimizers_zero_grad()
    # 1- or 2-pass forward
    ae_out_0 = ae_model_parallel(x_in, v_in, uid, notes, pass_index=0)
    ae_out_0 = ae_model.parse_outputs(ae_out_0)
    if ae_model.preset_ae_method == 'aligned_vaes':
        ae_out_1 = ae_model_parallel(x_in, v_in, uid, notes, pass_index=1)
        ae_out_1 = ae_model.parse_outputs(ae_out_1)
    else:
        ae_out_1 = None
    if ae_model.pre_training_audio:
        ae_out_audio, ae_out_preset = ae_out_0, None
    else:
        # assign the output(s) to preset- and/or audio-outputs (not to log Nones, and to log the proper values)
        if ae_model.preset_ae_method in ['no_encoding', 'combined_vae', 'no_input_audio']:  # Actual preset AE methods
            ae_out_audio, ae_out_preset = ae_out_0, ae_out_0
        elif ae_model.preset_ae_method == 'no_audio':
            ae_out_audio, ae_out_preset = None, ae_out_0
        elif ae_model.preset_ae_method == 'aligned_vaes':
            ae_out_audio, ae_out_preset = ae_out_1, ae_out_0
        else:
            raise NotImplementedError(f"Unknown preset_ae_method {ae_model.preset_ae_method}")

    # The single-pass output which is the most interesting to us, in order to know how the latent space behaves:
    # Always log "preset-related" latent metrics (in the end, we'll be interested in presets only).
    #    Anyway, they are often the same as "audio-related" latent metrics
    ae_out_latent = ae_out_preset if not ae_model.pre_training_audio else ae_out_audio

    super_metrics['LatentMetric' + suffix].append(
        ae_out_latent.z_mu, ae_out_latent.z_var, ae_out_latent.z_sampled, labels=labels)

    # Losses (some of them are computed on 1 GPU using the non-parallel original model instance)
    if ae_out_audio is not None:
        audio_log_prob_loss = ae_out_audio.x_target_NLL.mean()
        scalars['Audio/LogProbLoss' + suffix].append(audio_log_prob_loss)
    else:
        audio_log_prob_loss = torch.zeros((1,), device=main_device)
    # TODO which AE_OUT to use for 2-pass models???
    #    compute both, then average?
    #    use a different beta
    beta = scalars['Sched/VAE/beta'].get(epoch)
    latent_loss_args = (beta, labels, attributes)
    lat_loss, lat_backprop_loss, lat_extra_loss = ae_model.latent_loss(ae_out_0, *latent_loss_args)
    if ae_out_1 is not None:
        lat_loss_1, lat_backprop_loss_1, lat_extra_loss_1 = ae_model.latent_loss(ae_out_1, *latent_loss_args)
        lat_loss += lat_loss_1  # don't average - would be equivalent to reducing beta for each VAE
        lat_backprop_loss += lat_backprop_loss_1
        lat_extra_loss += lat_extra_loss_1
    scalars['Latent/Loss' + suffix].append(lat_loss)
    scalars['Latent/BackpropLoss' + suffix].append(lat_backprop_loss)  # Includes beta
    scalars['Latent/ExtraBpLoss' + suffix].append(lat_extra_loss)  # TODO doc
    if not ae_model.pre_training_audio:
        align_loss = ae_model.latent_alignment_loss(ae_out_audio, ae_out_preset, beta)
        scalars['Latent/AlignLoss' + suffix].append(align_loss)

        u_categorical_nll, u_numerical_nll = ae_out_preset.u_categorical_nll.mean(), ae_out_preset.u_numerical_nll.mean()
        scalars['Preset/NLL/Numerical' + suffix].append(u_numerical_nll)
        scalars['Preset/NLL/CatCE' + suffix].append(u_categorical_nll)
        scalars['Preset/NLL/Total' + suffix].append(u_numerical_nll + u_categorical_nll)
        preset_loss = ae_model.params_categorical_loss_factor * u_categorical_nll + \
            ae_model.params_numerical_loss_factor * u_numerical_nll
    else:
        align_loss, preset_loss = torch.zeros((1,), device=main_device), torch.zeros((1,), device=main_device)
    # FIXME training or not
    preset_reg_loss = torch.zeros((1,), device=main_device)  # No regularization yet....

    with torch.no_grad():  # Monitoring-only losses
        scalars['Latent/MMD' + suffix].append(ae_model.mmd(ae_out_latent.z_sampled))
        if not ae_model.pre_training_audio:
            scalars['Preset/Accuracy' + suffix].append(ae_out_preset.u_accuracy.mean())
            scalars['Preset/L1error' + suffix].append(ae_out_preset.u_l1_error.mean())
        if ae_out_audio is not None:
            scalars['Audio/MSE' + suffix].append(F.mse_loss(ae_out_audio.x_sampled, x_in))
            scalars['VAELoss/Total' + suffix].append(ae_model.audio_vae_loss(
                audio_log_prob_loss, x_in.shape, ae_out_audio))
            scalars['VAELoss/Backprop' + suffix].append(audio_log_prob_loss + lat_backprop_loss)

    if training:
        utils.exception.check_nan_values(
            epoch, audio_log_prob_loss, lat_backprop_loss, align_loss, preset_loss, preset_reg_loss)
        # Backprop and optimizers' step (before schedulers' step)
        (audio_log_prob_loss + lat_backprop_loss + lat_extra_loss + align_loss + preset_loss + preset_reg_loss).backward()
        ae_model.optimizers_step()

    return ae_out_audio, ae_out_preset
