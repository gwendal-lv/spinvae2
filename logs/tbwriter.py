import warnings

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

from .metrics import BufferedMetric, LatentMetric, VectorMetric

import utils.stat


# TODO check if this mod remains necessary with PyTorch 1.10
class CorrectedSummaryWriter(SummaryWriter):
    """ SummaryWriter corrected to prevent extra runs to be created
    in Tensorboard when adding hparams.

    Original code in torch/utils/tensorboard.writer.py,
    modification by method overloading inspired by https://github.com/pytorch/pytorch/issues/32651 """

    def add_hparams(self, hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None):
        assert run_name is None  # Disabled feature. Run name init by summary writer ctor

        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict, hparam_domain_discrete)

        # run_name argument is discarded and the writer itself is used (no extra writer instantiation)
        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            self.add_scalar(k, v)


class TensorboardSummaryWriter(CorrectedSummaryWriter):
    """ Tensorboard SummaryWriter with corrected add_hparams method
     and extra functionalities. """

    def __init__(self, log_dir=None, comment='', purge_step=None, max_queue=10,
                 flush_secs=120, filename_suffix='',
                 model_config=None, train_config=None  # Added (actually mandatory) arguments
                 ):
        super().__init__(log_dir, comment, purge_step, max_queue, flush_secs, filename_suffix)
        # Full-Config is required. Default constructor values allow to keep the same first constructor args
        self.model_config = model_config
        self.train_config = train_config
        self.resume_from_checkpoint = (train_config.start_epoch > 0)
        self.hyper_params = dict()
        self.hparams_domain_discrete = dict()  # TODO hparam domain discrete
        # General and dataset hparams
        self.hyper_params['batchsz'] = self.train_config.minibatch_size
        self.hyper_params['kfold'] = self.train_config.current_k_fold
        self.hparams_domain_discrete['kfold'] = list(range(self.train_config.k_folds))
        self.hyper_params['wdecay'] = self.train_config.weight_decay
        self.hyper_params['synth'] = self.model_config.synth
        self.hyper_params['syntargs'] = self.model_config.synth_args_str
        self.hyper_params['nmidi'] = '{}{}'.format(len(self.model_config.midi_notes),
                                                   ("stack" if model_config.stack_spectrograms else "inde"))
        self.hyper_params['cont_max_reso'] = self.model_config.continuous_params_max_resolution
        self.hyper_params['normalizeloss'] = self.train_config.normalize_losses
        # Latent space hparams
        self.hyper_params['z_dim'] = self.model_config.dim_z
        self.hyper_params['latloss'] = self.train_config.latent_loss
        self.hyper_params['latbeta'] = self.train_config.beta
        self.hyper_params['stylearch'] = self.model_config.style_architecture
        # Synth controls regression
        self.hyper_params['ncontrols'] = self.model_config.synth_params_count
        # self.hyper_params['contloss'] = self.model_config.controls_losses
        self.hyper_params['regr_arch'] = self.model_config.params_regression_architecture
        self.hyper_params['regr_FCdrop'] = self.train_config.reg_fc_dropout
        self.hyper_params['regr_outhtanh'] = self.model_config.params_reg_hardtanh_out
        self.hyper_params['regr_outsoftm'] = self.model_config.params_reg_softmax
        self.hyper_params['regr_catloss'] = 'BinCE' if self.train_config.params_cat_bceloss else 'CatCE'
        #self.hyper_params['regloss_factor'] = self.train_config.params_loss_compensation_factor
        self.hyper_params['regloss_excl_useless'] = self.train_config.params_loss_exclude_useless
        self.hyper_params['regloss_permutations'] = self.train_config.params_loss_with_permutations
        self.hyper_params['regloss_target_noise'] = self.train_config.params_target_noise
        self.hyper_params['regloss_label_smooth'] = self.train_config.params_cat_CE_label_smoothing
        self.hyper_params['regloss_cat_weights'] = self.train_config.params_cat_CE_use_weights
        # Auto-Encoder hparams
        self.hyper_params['VAE_FCdrop'] = self.train_config.ae_fc_dropout
        self.hyper_params['main_conv_arch'] = self.model_config.vae_audio_architecture
        # self.hyper_params['recloss'] = self.train_config.ae_reconstruction_loss
        self.hyper_params['specmindB'] = self.model_config.spectrogram_min_dB
        self.hyper_params['mel_nbins'] = self.model_config.mel_bins
        self.hyper_params['mel_fmin'] = self.model_config.mel_f_limits[0]
        self.hyper_params['mel_fmax'] = self.model_config.mel_f_limits[1]
        # Easily improved tensorboards hparams logging: convert bools to strings
        for k, v in self.hyper_params.items():
            if isinstance(v, bool):
                self.hyper_params[k] = str(v)
                self.hparams_domain_discrete[k] = ['True', 'False']

    def init_hparams_and_metrics(self, metrics):
        """ Hparams and Metric initialization. Will pass if training resumes from saved checkpoint.
        Hparams will be definitely set but metrics can be updated during training.

        :param metrics: Dict of BufferedMetric
        """
        if not self.resume_from_checkpoint:  # tensorboard init at epoch 0 only
            # Some processing on hparams can be done here... none at the moment
            self.update_metrics(metrics)

    def update_metrics(self, metrics):
        """ Updates Tensorboard metrics

        :param metrics: Dict of values and/or BufferedMetric instances
        :return: None
        """
        metrics_dict = dict()
        for k, metric in metrics.items():
            if isinstance(metrics[k], BufferedMetric):
                try:
                    metrics_dict[k] = metric.mean
                except ValueError:
                    metrics_dict[k] = 0  # TODO appropriate default metric value?
            else:
                metrics_dict[k] = metric
        self.add_hparams(self.hyper_params, metrics_dict, hparam_domain_discrete=self.hparams_domain_discrete)

    def add_vector_histograms(self, metric: VectorMetric, metric_name: str, global_step: int, bins='fd'):
        """ Plots flattened data as Tensorboard native histograms, with and without outliers.
        Name must contain a '/Train' or '/Valid' suffix """
        data = metric.get().flatten()
        self.add_histogram(metric_name, data, global_step=global_step, bins=bins)
        if '/' in metric_name:
            no_outlier_name = metric_name.replace('/', '_no_outlier/')
            self.add_histogram(no_outlier_name, utils.stat.remove_outliers(data), global_step=global_step, bins=bins)
        else:
            warnings.warn("Metric name '{}' does not contain a dataset type (e.g. /Train or /Validation)"
                          .format(metric_name))

    def add_latent_histograms(self, latent_metric: LatentMetric, dataset_type: str, global_step: int, bins='fd'):
        """
        Adds histograms related to z0 and zK samples to Tensorboard.

        :param dataset_type: 'Train', 'Valid', ...
        """
        z0 = latent_metric.get_z('z0').flatten()
        zK = latent_metric.get_z('zK').flatten()
        self.add_histogram("z0/{}".format(dataset_type), z0, global_step=global_step, bins=bins)
        self.add_histogram("zK/{}".format(dataset_type), zK, global_step=global_step, bins=bins)
        # also add no-outlier histograms (the other ones are actually unreadable...)
        self.add_histogram("z0/{}_no_outlier".format(dataset_type), utils.stat.remove_outliers(z0),
                           global_step=global_step, bins=bins)
        self.add_histogram("zK/{}_no_outlier".format(dataset_type), utils.stat.remove_outliers(zK),
                           global_step=global_step, bins=bins)

    def add_latent_embedding(self, latent_metric: LatentMetric, dataset_type: str, global_step: int):
        """ FIXME nothing shows up in tensorboard - this is one of the reasons for moving to comet """
        rng = np.random.default_rng(seed=global_step)
        labels_uint8 = latent_metric.get_z('label')  # One sample can have 0, 1 or multiple labels
        # labels converted to strings
        metadata = list()
        for i in range(labels_uint8.shape[0]):  # np.nonzero does not have a proper row-by-row option
            label_indices = np.nonzero(labels_uint8[i, :])
            if len(label_indices) == 0:
                metadata.append("No label")
            else:  # Only 1 label will be displayed (randomly chosen if multiple labels)
                rng.shuffle(label_indices)  # in-place
                metadata.append(str(label_indices[0]))  # FIXME use string label instead of int index
        # TODO add z0 and zK embeddings
        # tensorboard issue: https://github.com/pytorch/pytorch/issues/30966
        # => need to remove tensorflow from the conda venv => runs, but metadata still doesn't show
        self.add_embedding(latent_metric.get_z('z0'), metadata=metadata,   # TODO use metadata_header?
                           tag="z0/{}".format(dataset_type), global_step=global_step)
