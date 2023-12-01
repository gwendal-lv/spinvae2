from typing import Optional

import numpy as np
import torch
from comet_ml import Experiment  # Needs to be imported first

import matplotlib.pyplot as plt

import config
from data.abstractbasedataset import AudioDataset
from logs.metrics import LatentMetric


class CometWriter:
    def __init__(self, model_config: config.ModelConfig, train_config: config.TrainConfig):
        # We'll create a new experiment, or load an existing one if restarting from a checkpoint
        if train_config.start_epoch == 0:
            self.experiment = Experiment(
                api_key=model_config.comet_api_key, workspace=model_config.comet_workspace,
                project_name=model_config.comet_project_name,
                log_git_patch=False,  # Quite large, and useless during remote dev (different commit on remote machine)
                log_env_cpu=False,  # annoying with numerous CPU cores
            )
            self.experiment.set_name(model_config.name + '/' + model_config.run_name)
            if len(model_config.comet_tags) > 0:
                self.experiment.add_tags(model_config.comet_tags)
            # TODO refactor - MODEL CONFIG IS MODIFIED HERE - dirty but quick....
            model_config.comet_experiment_key = self.experiment.get_key()
        else:
            raise NotImplementedError()  # TODO implement ExistingExperiment

    def log_config_hparams(self, model_config: config.ModelConfig, train_config: config.TrainConfig):
        # Ordering is the same as in config.py (easier to find added/removed params)
        self.experiment.log_parameter("vae_audio_arch", model_config.vae_audio_architecture)
        self.experiment.log_parameter("vae_latent_arch", model_config.vae_latent_extract_architecture)
        self.experiment.log_parameter("vae_n_cells", model_config.vae_n_cells)
        self.experiment.log_parameter("z_dim", model_config.dim_z)
        self.experiment.log_parameter("mel_bins", model_config.mel_bins)
        self.experiment.log_parameter("n_midi_notes", len(model_config.midi_notes))
        self.experiment.log_parameter("batch_size", train_config.minibatch_size)
        self.experiment.log_parameter("n_extra_random_presets", train_config.n_extra_random_presets)
        self.experiment.log_parameter("z_loss", train_config.latent_loss)
        self.experiment.log_parameter("z_beta", train_config.beta)
        self.experiment.log_parameter("z_beta_cycle", str(train_config.beta_cycle))
        self.experiment.log_parameter("z_beta_cycle_duration", str(train_config.beta_cycle_duration))
        self.experiment.log_parameter("z_free_bits", train_config.latent_free_bits)
        self.experiment.log_parameter("z_gamma", train_config.latent_regularization_gamma)
        self.experiment.log_parameter("z_delta", train_config.latent_regularization_delta)
        self.experiment.log_parameter("params_cat_loss_factor", train_config.params_categorical_loss_factor)
        self.experiment.log_parameter("params_num_loss_factor", train_config.params_numerical_loss_factor)
        self.experiment.log_parameter("params_loss_exclude", train_config.params_loss_exclude_useless)
        self.experiment.log_parameter("params_loss_permut", train_config.params_loss_with_permutations)
        self.experiment.log_parameter("preset_lbl_smooth", train_config.preset_CE_label_smoothing)
        self.experiment.log_parameter("preset_CE_weights", train_config.preset_CE_use_weights)
        self.experiment.log_parameter("preset_arch", model_config.vae_preset_architecture)
        self.experiment.log_parameter("preset_hid_size", model_config.preset_hidden_size)
        self.experiment.log_parameter("preset_num_distribution", model_config.preset_decoder_numerical_distribution)
        self.experiment.log_parameter("preset_encode_add", model_config.vae_preset_encode_add)
        self.experiment.log_parameter("preset_ae_method", model_config.preset_ae_method)
        self.experiment.log_parameter("optimizer", train_config.optimizer)
        if train_config.optimizer.lower() == 'adam':
            self.experiment.log_parameter("opt_adam_beta1", train_config.adam_betas[0])
            self.experiment.log_parameter("opt_adam_beta2", train_config.adam_betas[1])
        for k, lr in train_config.initial_learning_rate.items():
            self.experiment.log_parameter("LR_" + k + "_initial", lr)
        self.experiment.log_parameter("LR_sched", train_config.scheduler_name)
        self.experiment.log_parameter("LR_sched_period", train_config.scheduler_period)
        # TODO log ALL scheduler options (not only StepLR period...)
        self.experiment.log_parameter("weight_decay", train_config.weight_decay)
        self.experiment.log_parameter("AE_FC_dropout", train_config.ae_fc_dropout)
        self.experiment.log_parameter("preset_cat_drop", train_config.preset_cat_dropout)
        self.experiment.log_parameter("preset_intern_drop", train_config.preset_internal_dropout)

    def log_metric_with_context(self, name: str, value):
        """ Logs a scalar metric, and tries to set a comet context before doing it (e.g. "Loss/Train" will be
            logged as "Loss" in a train experiment context). """
        if self._has_train_suffix(name):
            with self.experiment.train():
                self.experiment.log_metric(self._remove_train_valid_suffix(name), value)
        elif self._has_validation_suffix(name):
            with self.experiment.validate():
                self.experiment.log_metric(self._remove_train_valid_suffix(name), value)
        else:  # Log without context (e.g. scheduler, ...)
            self.experiment.log_metric(name, value)

    def log_figure_with_context(self, name: str, fig, step: Optional[int] = None):
        """ Logs a MPL figure, and tries to set a comet context before doing it (see log_metric_with_context).
            Step can be forced if this method is called asynchronously (threaded plots). """
        if self._has_train_suffix(name):
            with self.experiment.train():
                self.experiment.log_figure(self._remove_train_valid_suffix(name), fig, step=step)
        elif self._has_validation_suffix(name):
            with self.experiment.validate():
                self.experiment.log_figure(self._remove_train_valid_suffix(name), fig, step=step)
        else:  # Log without context (e.g. scheduled hyper-parameters (LR, ...), ...)
            self.experiment.log_figure(name, fig, step=step)
        plt.close(fig)

    @staticmethod
    def _assert_is_train_or_valid_dataset_type(ds_type: str):
        """ Returns True if this is a training dataset, False if it is a validation dataset, raises an exception
            otherwise. """
        if "train" in ds_type.lower():
            return True
        elif "valid" in ds_type.lower():
            return False
        else:
            raise ValueError("Dataset type '{}' cannot be identified as a training or a validation dataset.")


    @staticmethod
    def _has_train_suffix(name: str):
        return name.endswith("/Train") or name.endswith("/train")

    @staticmethod
    def _has_validation_suffix(name: str):
        return name.endswith("/Valid") or name.endswith("/valid") \
               or name.endswith("/Validation") or name.endswith("/validation")

    @staticmethod
    def _remove_train_valid_suffix(s: str, suffix_separator='/'):
        s_split = s.split(suffix_separator)
        s_split.pop()
        return suffix_separator.join(s_split)

    def log_latent_histograms(self, latent_metric: LatentMetric, dataset_type: str,
                              epoch: Optional[int] = None, step: Optional[int] = None):
        """
        Adds histograms related to z0 and zK samples to Tensorboard.

        :param dataset_type: 'Train' or 'Valid'
        :param epoch: Optional epoch, should be used with async (threaded) plot rendering.
        :param step: Optional step, should be used with async (threaded) plot rendering.
        """
        is_training = self._assert_is_train_or_valid_dataset_type(dataset_type)
        with self.experiment.train() if is_training else self.experiment.validate():
            z0 = latent_metric.get_z('z0').flatten()
            zK = latent_metric.get_z('zK').flatten()
            self.experiment.log_histogram_3d(z0, name='z0', step=step, epoch=epoch)
            self.experiment.log_histogram_3d(zK, name='zK', step=step, epoch=epoch)

    def log_latent_embedding(self, latent_metric: LatentMetric, dataset_type: str, epoch: int, dataset: AudioDataset):
        z_K = latent_metric.get_z('zK')
        embeddings = list()
        # labels converted to strings
        rng = np.random.default_rng(seed=epoch)  # TODO refactor this, move to utils/label.py
        labels_uint8 = latent_metric.get_z('label')  # One sample can have 0, 1 or multiple labels
        labels = list()
        for i in range(labels_uint8.shape[0]):  # np.nonzero does not have a proper row-by-row option
            label_indices = np.flatnonzero(labels_uint8[i, :])
            if len(label_indices) == 0:  # We remove latent samples which do not contain any label
                # All samples might be logged in the future, if samples (presets) names are also provided
                pass  # labels.append("-")
            else:  # Only 1 label will be displayed (randomly chosen if multiple labels)
                # TODO try add multiple labels ? or use metadata to build a secondary label ?
                rng.shuffle(label_indices)  # in-place
                labels.append(dataset.available_labels_names[label_indices[0]])
                embeddings.append(z_K[i, :])
        is_training = self._assert_is_train_or_valid_dataset_type(dataset_type)
        with self.experiment.train() if is_training else self.experiment.validate():
            self.experiment.log_embedding(embeddings, labels, title="zK")

