"""
Allows easy modification of all configuration parameters required to define,
train or evaluate a model.
This script is not intended to be run, it only describes parameters (see classes constructors).
After building the config instances, the update_dynamic_config_params(...) method
must be called to update some "dynamic" hyper-parameters which depend on some others.

This configuration is used when running train.py as main.
When running train_queue.py, configuration changes are relative to this config.py file.

When a run starts, this file is stored as a config.json file. To ensure easy restoration of
parameters, please only use simple types such as string, ints, floats, tuples (no lists) and dicts.
"""


import datetime
import pathlib

# The config_confidential.py file must be created by the user in the ./utils folder.
# It must contain the following fields:
#     data_root_path, logs_root_dir,
#     comet_api_key, comet_project_name, comet_workspace
from utils import config_confidential


# ===================================================================================================================
# ================================================= Model configuration =============================================
# ===================================================================================================================
class ModelConfig:
    def __init__(self):
        # ----------------------------------------------- Data ---------------------------------------------------
        self.data_root_path = config_confidential.data_root_path
        self.logs_root_dir = config_confidential.logs_root_dir
        self.name = "dev"  # experiment base name
        # experiment run: different hyperparams, optimizer, etc... for a given exp
        self.run_name = 'dummy'
        self.pretrained_VAE_checkpoint = self.logs_root_dir + \
            "/pAE_zAttr/AttrARVAE+zChunks_z256_gamma0.00100/checkpoint.tar"  # Full model
        #    "/audioAE_zAttr/AttrS2VAE+zChunks_z256/checkpoint.tar"  # Audio only - specladder8x1_res_swish
        # self.pretrained_VAE_checkpoint = None  # Uncomment this to train a full model from scratch
        self.allow_erase_run = True  # If True, a previous run with identical name will be erased before training
        # Comet.ml logger (replaces Tensorboard)
        self.comet_api_key = config_confidential.comet_api_key
        self.comet_project_name = config_confidential.comet_project_name
        self.comet_workspace = config_confidential.comet_workspace
        self.comet_experiment_key = 'xxxxxxxx'  # Will be set by cometwriter.py after experiment has been created
        self.comet_tags = ['latent_attributes']

        # ---------------------------------------- General Architecture --------------------------------------------
        # See model/ladderencoder.py to view available architecture(s).
        # Decoder architecture will be as symmetric as possible.
        #    'specladder': ladder CNNs (cells, outputs from different levels) for spectrogram reconstruction
        #                  also contains num of blocks and num of conv layers per block (e.g. specladder8x1_res_swish)
        #    'ast': Audio Spectrogram Transformer (inspired by, but not exactly AST Interspeech21), TODO doc
        #             also specifies the number of transformer blocks (e.g. 'ast6')
        # General arch args:
        #    '_big' (small improvements but +50% GPU RAM usage),   '_bigger'
        # CNN-specific arch args:
        #    '_adain' some BN layers are replaced by AdaIN (fed with a style vector w, dim_w < dim_z)
        #    TODO '_film' FiLM (AAAI 17) layers added after some BN layers, in order to e.g. condition on the MIDI note
        #    '_att' self-attention in deep conv layers (requires at least 8x2)
        #    '_res' residual connections after each hidden strided conv layer (up/down sampling layers)
        #    '_depsep5x5' uses 5x5 depth-separable convolutional layers in each res block (requires at least 8x2)
        #    '_ln' uses LayerNorm instead of BatchNorm, '_wn' uses Weight Normalization attached to conv weights
        #    '_swish' uses Swish activations (SiLU) instead of LeakyReLU (negligible overhead)
        self.vae_audio_architecture = 'specladder8x1_res_swish'  # 'ast6h256_p256x1'
        # TODO doc Number of encoder and decoder "cells"; cells' outputs from audio and preset encoder/decoders can be mixed
        self.vae_n_cells = 1
        # Sets the family of decoder output probability distribution p_theta(x|z), e.g. :
        #    - 'gaussian_unitvariance' corresponds to the usual MSE reconstruction loss (up to a constant and factor)
        self.audio_decoder_distribution = 'gaussian_unitvariance'
        # Preset encoder/decoder architecture. Base archs are 'mlp', 'lstm', 'tfm' (transformer). Options:
        #   '_ff': feed-forward, non-AR decoding - applicable to sequential models: RNN, Transformer (pos enc only)
        #   '_memmlp': doubles the number of Transformer decoder memory tokens using a "Res-MLP" on the latent vector
        #              -> seems to improve perfs a bit (lower latent loss, quite similar auto synth prog losses)
        #   '_fftoken': learned positional embeddings (input tokens) for a non-autoregressive transformer decoder
        #   '_embednorm': input embeddings and learned positional embeddings have a max_norm (norm-2)
        #   TODO DOC '_normfirst': ???
        self.vae_preset_architecture = 'tfm_6l_ff_memmlp_fftoken_embednorm_normfirst'  # tfm_6l_memmlp_ff
        # Size of the hidden representation of 1 synth parameter
        #     Transformer num heads auto increases (head size: 64)
        self.preset_hidden_size = 256
        # Distribution for modeling (discrete-)numerical synth param values; available options:
        #   - 'logistic_mixtX' where X is the number of mixture components (mu, s and pi learned for each component)
        #   - 'softmax' will learn values as categories - custom softmax (cardinality) for each param
        # (categorical variables always use a softmaxed categorical distribution)
        self.preset_decoder_numerical_distribution = 'logistic_mixt3'

        # --------------------------------------------- Latent space -----------------------------------------------
        self.dim_z = 256
        # TODO update doc....
        # Network plugged after sequential conv blocks (encoder) or before sequential conv blocks (decoder)
        # E.g.: 'conv_1l_1x1' means regular convolution, 1 layer, 1x1 conv kernels
        #       'lstm_2l_3x3' mean ConvLSTM, 2 layers, 3x3 conv kernels
        # other args: '_gated' (for regular convolutions) seems very effective to improve the overall ELBO
        #             '_att' adds self-attention conv residuals at the beginning of shallow latent cells
        #             '_posenc' adds input positional information to LSTMs and to self-attention conv layers
        self.vae_latent_extract_architecture = 'conv_1l_k1x1_gated'  # 'none'
        # Preset encoding method: TODO update, deprecated
        # "before_latent_cell" (encoded preset will be the same size as encoded audio, added then processed together)
        # or "after_latent_cell"" (encoded preset size will be 2*dim_z, and will be added to z_mu and z_sigma)
        self.vae_preset_encode_add = "after_latent_cell"
        # If True, encoder output is reduced by 2 for 1 MIDI pitch and 1 velocity to be concat to the latent vector
        self.concat_midi_to_z = None  # See update_dynamic_config_params() - FIXME deprecated, remove
        # Describes how (if) the presets should be auto-encoded:
        #    - "no_encoding": presets are inferred from audio but are not encoded (not provided at
        #               encoder input). Corresponds to a Sound Matching (or automatic synth programming) task.
        #    - "combined_vae": preset is encoded with audio, their hidden representations are then summed or mixed
        #           together
        #    - TODO "asp+vae": hybrid method/training: TODO DOC
        #    - "aligned_vaes": the preset VAE and audio VAE are trained as independent models, but a loss
        #           (e.g. contrastive, Dkl, ... TODO TBD) is computed using the two latent representations
        #    - "no_audio": the preset alone is auto-encoded, audio is discarded
        #    - "no_input_audio": input spectrogram is not encoded - forces the tfm-encoder and cnn-decoder
        #                        to behave as a neural synth proxy
        self.preset_ae_method = "combined_vae"#"no_input_audio"

        # ------------------------------------------------ Audio -------------------------------------------------
        # Spectrogram size cannot easily be modified - all CNN decoders should be re-written
        self.note_duration = (3.0, 1.0)
        self.sampling_rate = 16000  # 16000 for NSynth dataset compatibility
        # # fft size and hop size, and number of Mel bins (-1 disables Mel-scale spectrogram, try: 257, 513, ...)
        # self.stft_args, self.mel_bins = (512, 256), -1
        self.stft_args, self.mel_bins = (1024, 256), 257
        # Spectrogram sizes @ 22.05 kHz:
        #   (513, 433): audio 5.0s, fft size 1024, fft hop 256
        #   (257, 347): audio 4.0s, fft size 512 (or fft 1024 w/ mel_bins 257), fft hop 256
        #   (513, 347): audio 4.0s, fft size 1024 (no mel), fft hop 256
        # Sizes @ 16 kHz:
        #   (257, 251): audio 4.0s, fft size 512 (or fft 1024 w/ mel_bins 257), fft hop 256
        self.spectrogram_size = (257, 251)  # H x W. see data/dataset.py to retrieve this from audio/stft params
        self.mel_f_limits = (0, 8000)  # min/max Mel-spectrogram frequencies (librosa default 0:Fs/2)
        # All notes that must be available for each instrument (even if we currently use only a subset of those notes)
        # self.required_dataset_midi_notes = ((41, 75), (48, 75), (56, 75), (63, 75), (56, 25), (56, 127))
        self.required_dataset_midi_notes = ((56, 75), )
        # Tuple of (pitch, velocity) tuples. Using only 1 midi note is fine.
        # self.midi_notes = ((56, 75), )  # Reference note: G#3 , intensity 75/127
        # self.midi_notes = ((41, 75), (48, 75), (56, 25), (56, 75), (56, 127), (63, 75))  # 6 notes
        # self.midi_notes = ((41, 75), (56, 25), (56, 75), (56, 127), (63, 75))  # 5 notes
        # self.midi_notes = ((41, 75), (56, 75), (56, 127))  # 3 notes (faster training, or much bigger conv models)
        self.midi_notes = ((56, 75), )
        self.main_midi_note_index = len(self.midi_notes) // 2  # 56, 75
        self.stack_spectrograms = True  # If True, dataset will feed multi-channel spectrograms to the encoder
        # If True, each preset is presented several times per epoch (nb of train epochs must be reduced) such that the
        # dataset size is increased (6x bigger with 6 MIDI notes) -> warmup and patience epochs must be scaled
        self.increased_dataset_size = None  # See update_dynamic_config_params()
        self.spectrogram_min_dB = -120.0
        self.input_audio_tensor_size = None  # see update_dynamic_config_params()

        # ---------------------------------- Synth (not used during pre-training) ----------------------------------
        self.synth = 'dexed'
        self.synth_params_count = -1  # Will be set automatically - see data.build.get_full_and_split_datasets
        # flags/values to describe the dataset to be used
        self.dataset_labels = None  # tuple of labels (e.g. ('harmonic', 'percussive')), or None to use all labels

        # Timbre attributes (features, from TimbreToolbox and/or AudioCommons Timbral Models) used for
        # semi-supervised latent regularization. This can contain individual attributes names,
        # e.g. 'ac_warmth' or 'tt_SpecCent_med' (see utils/timbrefeatures.py), 'ac_*', 'tt_*',
        # '__no_high_corr__' to exclude attributes with a high correlation.
        # This could be in TrainConfig instead, but datasets' ctors currently use ModelConfig only
        self.latent_timbre_attributes = ['ac_*', 'tt_*', '__no_high_corr__']


# ===================================================================================================================
# ======================================= Training procedure configuration ==========================================
# ===================================================================================================================
class TrainConfig:
    def __init__(self):
        self.pretrain_audio_only = False  # Should we pre-train the audio+latent parts of the auto-encoder model only?
        self.frozen_weight_groups = ()  # Can contain 'audio', 'latent' or 'preset'
        # Interpolation will be evaluated vs. a reference model, which must have been evaluated first
        self.evaluate_interpolation_after_training = True  # Parallel eval (long, CPU-intensive)
        self.start_datetime = datetime.datetime.now().isoformat()

        # 256 is okay for smaller conv structures - reduce to 64 to fit '_bigger' models into 24GB GPU RAM
        self.minibatch_size = 128  # reduce for big models - also smaller N seems to improve VAE pretraining perfs...
        self.main_cuda_device_idx = 0  # CUDA device for nonparallel operations (losses, ...)
        self.start_epoch = 0  # 0 means a restart (previous data erased). If > 0: will load the last saved checkpoint
        # Total number of epochs (including previous training epochs).  275 for StepLR regression model training
        self.n_epochs = 220  # See update_dynamic_config_params()


        self.test_holdout_proportion = 0.1  # FIXME DEPRECATED This can be reduced without mixing the train and test subsets
        self.k_folds = 9  # FIXME DEPRECATED 10% for validation set, 80% for training
        self.current_k_fold = 0  # FIXME DEPRECATED  k-folds are not used anymore, but we'll keep the training/validation/test splits
        # How many randomly-generated presets should be added to the training dataset split
        self.n_extra_random_presets = 0  # Dexed max: 23614
        # The max ratio between the number of items from each synth/instrument used for each training epoch (e.g. Dexed
        # has more than 30x more instruments than NSynth). All available data will always be used for validation.
        self.pretrain_synths_max_imbalance_ratio = 10.0  # Set to -1 to disable the weighted sampler.

        # ------------------------------------------------ Losses -------------------------------------------------
        # - - - Latent loss - - -
        # Latent regularization loss: only 'Dkl' is currently available. Use '+' to include extra regularizations:
        #     - 'AttrS2VAE': https://arxiv.org/abs/1905.01258 ICLR20: originally used with "ordinal GT factors of
        #            variations", but https://arxiv.org/abs/2108.01450 (ISMIR21) used it with continuous attributes.
        #            Based on a BCE loss (even if we don't have 0/1 binary target classes).
        #     - 'AttrARVAE': enforces a monotonic relationship between attribute values and latent codes
        #             (but also somewhat enforces values of latent codes differences)
        #             https://arxiv.org/abs/2004.05485 and https://arxiv.org/abs/2108.01450
        #     TODO and/or 'LabelS2VAE' ???
        #    - 'zChunks': apply the regularization vs. 1 attr or label to a chunk of latent coordinates (e.g.
        #            a 2D latent feature map) instead of a single latent coordinate.
        self.latent_loss = 'Dkl+AttrARVAE+zChunks'  # AR seems best for fine-tuning, but S2 better for pre-training
        # Losses normalization allow to get losses in the same order of magnitude, but does not optimize the true ELBO.
        # When un-normalized, the reconstruction loss (log-probability of a multivariate gaussian) is orders of
        # magnitude bigger than other losses. Must remain True to ease convergence (too big recons loss)
        self.normalize_losses = True  # Normalize audio and preset losses over their dimension
        # To compare different latent sizes, Dkl or MMD losses are not normalized such that each latent
        # coordinate always 'has' the same amount of regularization
        self.normalize_latent_loss = False
        # Here, beta = beta_vae / Dx in the beta-VAE formulation (ICLR 2017)
        # where Dx is the input dimensionality (257 * 251 = 64 507 for 1 spectrogram)
        # E.g. here: beta = 1 corresponds to beta_VAE = 6.5 e+4
        #            ELBO loss is obtained by using beta = 1.55 e-5 (for 1 spectrogram)
        self.beta = 5.0e-5  # TODO 5e-6 for pre-training?
        self.beta_start_value = 0.0  # 1.6e-8  # minimal beta value
        # Beta Cyclical "annealing", to mitigate posterior collapse (also called: KL vanishing, etc...)
        #    https://arxiv.org/pdf/1903.10145.pdf    https://arxiv.org/abs/2004.04092
        #    'BETA_MIN' and 'BETA_MAX' will be replaced (see update method) by beta_start_value and beta
        self.beta_cycle = [[0.0, 'BETA_MIN'], [0.25, 'BETA_MAX'], [0.75, 'BETA_MAX'], [1.0, 'BETA_MAX']]
        self.beta_cycle_duration = -1  # if -1: will be auto-set to number of epochs
        # Free-bits from the IAF paper (NeurIPS 16) https://arxiv.org/abs/1606.04934
        # TODO doc: should we increase or reduce this to try to prevent extreme posterior collapse (during pre-training)
        self.latent_free_bits = 0.125  # this is a *single-coordinate* min KLD value TODO reduce this?

        # - - - Additional latent loss - - -
        # Gamma: factor applied to the extra regularization (e.g. attribute-based, ...)
        self.latent_regularization_gamma = 1e-3  # TODO doc and study this!!    1e-5 for pre-training ?
        # AR-VAE hparam: a larger delta allows the latent space to be closer to 0 (larger delta does not impose
        # large differences between values of a z-coordinate corresponding to different attribute values)
        self.latent_regularization_delta = 5.0  # TODO study this!!

        # - - - Synth parameters losses - - -
        # - General options
        # applied to the preset loss FIXME because MSE loss of the VAE is much lower (approx. 1e-2)
        self.params_categorical_loss_factor = 0.5
        self.params_numerical_loss_factor = 0.5
        self.params_loss_exclude_useless = False  # if True, sets to 0.0 the loss related to 0-volume oscillators
        self.params_loss_with_permutations = False  # Backprop loss only; monitoring losses always use True
        # - Cross-Entropy loss
        self.preset_CE_label_smoothing = 0.0  # torch.nn.CrossEntropyLoss: label smoothing since PyTorch 1.10
        self.preset_CE_use_weights = False
        # Probability to use the model's outputs during training (AR decoder)
        self.preset_sched_sampling_max_p = 0.0  # Set to zero for FF decoder
        # self.preset_sched_sampling_start_epoch = 40  # TODO IMPLEMENT Required for the embeddings to train properly?
        self.preset_sched_sampling_warmup_epochs = 100
        # Alignment loss (is actually a latent loss):
        #   - 'kld': q(z|preset) regularized towards q(z|audio) through KLD
        #   - 'symmetric_kld'
        #   TODO optional arg:
        #       - implement "__sgaudio" (stop audio-VAE gradient)
        self.preset_alignment_criterion = 'symmetric_kld'

        # ------------------------------------------- Optimizer + scheduler -------------------------------------------
        # Different optimizer parameters can be used for the pre-trained AE and the regression networks
        # (see below: 'ae' or 'reg' prefixes or dict keys)
        self.optimizer = 'Adam'
        self.adam_betas = (0.9, 0.999)  # default (0.9, 0.999)
        # Maximal learning rate (reached after warmup, then reduced on plateaus)
        #     use 4e-4 for an audio CNN 8x1 or 8x2 (batch size 128)
        #     bigger CNNs (more conv kernels, e.g. 8x1_big) : use TODO which value? 2e-4
        self.initial_learning_rate = {'audio': 4e-4, 'latent': 4e-4, 'preset': 4e-4}
        self.initial_audio_latent_lr_factor_after_pretrain = 1.0  # audio-related LR reduced after pre-train
        # Learning rate warmup (see https://arxiv.org/abs/1706.02677). Same warmup period for all schedulers.
        self.lr_warmup_epochs = 20  # Will be decreased /2 during pre-training (stable CNN structure)
        self.lr_warmup_start_factor = 0.05  # Will be increased 2x during pre-training
        # 'ReduceLROnPlateau', 'StepLR' or 'ExponentialLR'
        self.scheduler_name = 'ExponentialLR'
        # - - - Exponential scheduler options - - -
        self.scheduler_gamma = 0.982  # transformer are slower to train: +0.008 during fine-tuning TODO maybe increase even more
        # - - - StepLR scheduler options - - -
        self.scheduler_lr_factor = 0.4
        self.scheduler_period = 100  # resnets train quite fast - 75 for transformers?  TODO retry step LR with 75 period
        # - - - ReduceLROnPlateau scheduler options - - -
        # Set a longer patience with smaller datasets and quite unstable trains
        # See update_dynamic_config_params(). 16k samples dataset:  set to 10
        self.scheduler_patience = 20
        self.scheduler_cooldown = 20
        self.scheduler_threshold = 1e-4
        # Training considered "dead" when dynamic LR reaches this ratio of the initial LR
        # Early stop is currently used for the regression loss only, for the 'ReduceLROnPlateau' scheduler only.
        self.early_stop_lr_ratio = 1e-4
        self.early_stop_lr_threshold = None  # See update_dynamic_config_params()

        # -------------------------------------------- Regularization -----------------------------------------------
        # WD definitely helps for regularization but significantly impairs results. 1e-4 seems to be a good compromise
        # for both Basic and MMD VAEs (without regression net). 3e-6 allows for the lowest reconstruction error.
        self.weight_decay = 1e-5
        self.ae_fc_dropout = 0.0
        # FIXME use this dropout?
        self.preset_cat_dropout = 0.0  # Applied only to subnets which do not handle value regression tasks
        # This dropout (e.g. in hidden tfm layers) seems to have a quite bad effect on interpolation smoothness
        self.preset_internal_dropout = 0.0

        # -------------------------------------------- Logs, figures, ... ---------------------------------------------
        self.validate_period = 5  # Period between validations (very long w/ autoregressive transformers)
        self.plot_period = 20   # Period (in epochs) for plotting graphs into Tensorboard (quite CPU and SSD expensive)
        self.large_plots_min_period = 100  # Min num of epochs between plots (e.g. embeddings, approx. 80MB .tsv files)
        self.plot_epoch_0 = False
        self.verbosity = 1  # 0: no console output --> 3: fully-detailed per-batch console output
        self.init_security_pause = 0.0  # Short pause before erasing an existing run
        # Number of logged audio and spectrograms for a given epoch
        self.logged_samples_count = 4  # See update_dynamic_config_params()

        # -------------------------------------- Performance and Profiling ------------------------------------------
        self.dataloader_pin_memory = False
        self.dataloader_persistent_workers = True
        self.profiler_enabled = False
        self.profiler_epoch_to_record = 0  # The profiler will record a few minibatches of this given epoch
        self.profiler_kwargs = {'record_shapes': True, 'with_stack': True}
        self.profiler_schedule_kwargs = {'skip_first': 5, 'wait': 1, 'warmup': 1, 'active': 3, 'repeat': 2}





def update_dynamic_config_params(model_config: ModelConfig, train_config: TrainConfig):
    """ This function must be called before using any train attribute """

    # TODO perform config coherence checks in this function

    # Beta cycles
    for i in range(len(train_config.beta_cycle)):
        if train_config.beta_cycle[i][1] == 'BETA_MIN':
            train_config.beta_cycle[i][1] = train_config.beta_start_value
        elif train_config.beta_cycle[i][1] == 'BETA_MAX':
            train_config.beta_cycle[i][1] = train_config.beta
    if train_config.beta_cycle_duration < 0:
        train_config.beta_cycle_duration = train_config.n_epochs


    if train_config.pretrain_audio_only:
        model_config.comet_tags.append('pretrain')
        model_config.params_regression_architecture = 'None'
        model_config.preset_ae_method = None
        model_config.pretrained_VAE_checkpoint = None

        train_config.evaluate_interpolation_after_training = False
        train_config.lr_warmup_epochs = train_config.lr_warmup_epochs // 2
        train_config.lr_warmup_start_factor *= 2

    else:  # Full-training (using the pre-trained audio CNN)
        train_config.initial_learning_rate['audio'] *= train_config.initial_audio_latent_lr_factor_after_pretrain
        train_config.initial_learning_rate['latent'] *= train_config.initial_audio_latent_lr_factor_after_pretrain
        train_config.scheduler_gamma += 0.008  # Slower decrease (higher LR)

    # FIXME deprecated, remove: stack_spectrograms must be False for 1-note datasets - security check
    model_config.stack_spectrograms = model_config.stack_spectrograms and (len(model_config.midi_notes) > 1)
    model_config.concat_midi_to_z = (len(model_config.midi_notes) > 1) and not model_config.stack_spectrograms
    # Mini-batch size can be smaller for the last mini-batches and/or during evaluation
    model_config.input_audio_tensor_size = \
        (train_config.minibatch_size, 1 if not model_config.stack_spectrograms else len(model_config.midi_notes),
         model_config.spectrogram_size[0], model_config.spectrogram_size[1])

    # Dynamic train hyper-params
    train_config.early_stop_lr_threshold = {k: lr * train_config.early_stop_lr_ratio
                                            for k, lr in train_config.initial_learning_rate.items()}
    train_config.logged_samples_count = max(train_config.logged_samples_count, len(model_config.midi_notes))


    # Hparams that may be useless, depending on some other hparams
    if train_config.pretrain_audio_only or model_config.preset_ae_method != 'aligned_vaes':
        train_config.preset_alignment_criterion = None
    if train_config.scheduler_name == 'ExponentialLR':
        train_config.scheduler_cooldown, train_config.scheduler_patience, train_config.scheduler_threshold, \
            train_config.scheduler_period, train_config.scheduler_lr_factor = (None, ) * 5
    elif train_config.scheduler_name == 'StepLR':
        train_config.scheduler_cooldown, train_config.scheduler_patience, train_config.scheduler_threshold, \
            train_config.scheduler_gamma = (None, ) * 4
    elif train_config.scheduler_name == 'ReduceLROnPlateau':
        train_config.scheduler_period, train_config.scheduler_lr_factor, train_config.scheduler_gamma = (None, ) * 3
    else:
        raise ValueError(f"Unexpected scheduler name {train_config.scheduler_name}")


    assert model_config.synth == "dexed"

