"""
Contains methods for (re)generating datasets of synthesizer sounds.
This script must be run as main to generate some data (see __name__ == "__main__" section at the end of the file)

The current configuration from config.py will be used (automatically imported from inside the functions).
"""

import sys
import pathlib
import importlib
import warnings
from datetime import datetime

import config

from data import dataset
from data.dataset import SurgeDataset, NsynthDataset, DexedDataset
from data.abstractbasedataset import AudioDataset

import synth.surge  # To re-generate the list of patches (included in the synth itself)

import utils.label



def gen_dexed_dataset(regen_wav: bool, regen_spectrograms: bool, regen_learnable_presets: bool,
                      regen_labels: bool, regen_timbre_features: bool, try_read_dataset: bool):
    """
    Learnable preset regeneration:
        30k presets with 4x data augmentation (presets variations):
            2.6 minutes. , 1.3 ms / file.  (on 1 CPU),    500 MB
        54k presets, 4x data augmentation:
            3.8 minutes, 850 MB

    Approx audio rendering time:
        30293 patches, 6 notes and 1 variations / patch (48-core CPU),
            10.8 minutes (3.6ms/file
            Total 44 GB
            --> w/ data augmentation (4 presets variations): 44 min (3.6ms/file), 727k files, 175 GB
            --> 1 MIDI note, 4x data augmentation: 30 GB
        54k patches, 215k audio files:
            14min, 4ms/file, 52GB

    Approx spectrograms computation time: (2 variations)
        Compute and store:    Mel: 26.6 min (4.4 ms / spectrogram)   ;     STFT only: 8.3 min (1.4 ms/spec)
            54k presets, 430k spectrograms: Mel ??? min    STFT 11.5 min
        Normalize and store:
            54k presets, 430k spectrograms:  26min, 3.6ms / spec
        ---> 1 MIDI note, 8x data augmentation: 60 GB / 30k presets

    Timbre features:
    54k preset, 8x data augmentation: 430k audio files to be processed
        VERY LONG even on 24 CPUs (2 threads/CPU...) : approx. 24h

    If both params are set to False, the entire dataset will be read on 1 CPU (testing procedure)
        (4.4 ms / __getitem__ call (6 notes / item ; 1 CPU) - live learnable representations computation)
        3.1 ms / __getitem__ call (6 notes / item ; 1 CPU) <- pre-computed learnable representations
        (1.0ms without preset learnable representations calculations <- )

    """
    model_config, train_config = config.ModelConfig(), config.TrainConfig()
    config.update_dynamic_config_params(model_config, train_config)

    # No label restriction, no normalization, etc...
    # TODO check if using the randomly gnerated presets, or not
    dexed_dataset = DexedDataset(
        ** dataset.model_config_to_dataset_kwargs(model_config),
        restrict_to_labels=None,
        check_constrains_consistency=(not regen_wav) and (not regen_spectrograms) and (not regen_learnable_presets),
        # n_extra_random_presets=23614,
    )
    if regen_learnable_presets:
        dexed_dataset.compute_and_store_learnable_presets()
    _gen_dataset(dexed_dataset, regen_wav, regen_spectrograms, try_read_dataset)
    if regen_labels:  # Instruments labels only
        labeler = utils.label.NameBasedLabeler(dexed_dataset)
        labeler.extract_labels(verbose=True)
        print(labeler)
        dexed_dataset.save_labels(labeler.instrument_labels, labeler.labels_per_UID)
    if regen_timbre_features:
        dexed_dataset.compute_and_store_timbre_features()



def gen_surge_dataset(regen_patches_list: bool, regen_wav: bool, regen_spectrograms: bool,
                      regen_labels: bool, regen_timbre_features: bool, try_read_dataset: bool):
    """
    Approx audio rendering time:
        35 minutes (7ms/patch) for 2300 patches, 6 notes and 18 variations / patch (48-core CPU),
        Total 30 Go
    Approx spectrograms computation time:
        Compute and store:    Mel: 17min (4.1ms / spectrogram)   ;     STFT only: 5.5min (1.4ms/spec)
        Normalize and store: 1.6ms / spectrogram
        Total 60 Go (nfft 1024 hop 0256 mels 0257 : spectrograms are twice the size of wav files)

    If both params are set to False, the entire dataset will be read on 1 CPU (testing procedure)
        1.4ms / __getitem__ call (6 notes / item ; 1 CPU)
    """
    model_config, train_config = config.ModelConfig(), config.TrainConfig()
    config.update_dynamic_config_params(model_config, train_config)

    if regen_patches_list:
        synth.surge.Surge.update_patches_list()

    # No label restriction, etc...
    surge_dataset = SurgeDataset(** dataset.model_config_to_dataset_kwargs(model_config),
                                 data_augmentation=True,
                                 check_consistency=(not regen_wav) and (not regen_spectrograms))
    if regen_labels:  # Instruments labels only
        labeler = utils.label.SurgeReLabeler(surge_dataset)
        labeler.extract_labels(verbose=True)
        print(labeler)
        surge_dataset.save_labels(labeler.instrument_labels, labeler.labels_per_UID)

    _gen_dataset(surge_dataset, regen_wav, regen_spectrograms, try_read_dataset)

    if regen_timbre_features:
        surge_dataset.compute_and_store_timbre_features()


def gen_nsynth_dataset(regen_json: bool, regen_spectrograms: bool,
                       regen_labels: bool, regen_timbre_features: bool, try_read_dataset: bool):
    """
    Approx downloaded audio size: 30 GB?
        --> 39 GB with re-sorted JSON files and added symlinks (< 30s to compute and write all of them)

    Approx spectrograms computation time:
        Compute and store:    Mel: ????min (????ms / spectrogram)   ;     STFT only: ????min (????ms/spec)
        Normalize and store: ????ms / spectrogram
        Total 1.1 GB (nfft 1024 hop 0256 mels 0257 : spectrograms are twice the size of wav files)
            for each spectrogram configuration

    If both params are set to False, the entire dataset will be read on 1 CPU (testing procedure)
        1.0ms / __getitem__ call (6 notes / item ; 1 CPU)
    """
    model_config, train_config = config.ModelConfig(), config.TrainConfig()
    config.update_dynamic_config_params(model_config, train_config)
    # No label restriction, etc... FIXME also regenerate labels
    nsynth_dataset = NsynthDataset(** dataset.model_config_to_dataset_kwargs(model_config),
                                   data_augmentation=True,
                                   dataset_type='full',
                                   exclude_instruments_with_missing_notes=True,
                                   exclude_sonic_qualities=None,#['reverb'],
                                   force_include_all_acoustic=True,
                                   required_midi_notes=model_config.required_dataset_midi_notes
                                   )
    if regen_json:
        nsynth_dataset.regenerate_json_and_symlinks()
    _gen_dataset(nsynth_dataset, False, regen_spectrograms, try_read_dataset)
    if regen_labels:  # Instruments labels only
        labeler = utils.label.NSynthReLabeler(nsynth_dataset)
        labeler.extract_labels(verbose=True)
        print(labeler)
        nsynth_dataset.save_labels(labeler.instrument_labels, labeler.labels_per_UID)
    if regen_timbre_features:
        nsynth_dataset.compute_and_store_timbre_features()


def _gen_dataset(_dataset: AudioDataset, regenerate_wav: bool, regenerate_spectrograms: bool, try_read_dataset: bool):
    print(_dataset)
    # When computing stats, please make sure that *all* midi notes are available
    # --------------- WRITE ALL WAV FILES ---------------
    if regenerate_wav:
        _dataset.generate_wav_files()
    # ----- whole-dataset spectrograms and stats (for proper normalization) -----
    if regenerate_spectrograms:
        _dataset.compute_and_store_spectrograms_and_stats()

    if try_read_dataset:
        print("Test: reading the entire dataset once...")
        t_start = datetime.now()
        for i in range(len(_dataset)):
            _item = _dataset[i]  # try get an item - for debug purposes
        delta_t = (datetime.now() - t_start).total_seconds()
        print("{} __getitem__ calls: {:.1f}s total, {:.1f}ms/call"
              .format(len(_dataset), delta_t, 1000.0 * delta_t / len(_dataset)))



if __name__ == "__main__":
    # Datasets must be generated step-by-step :
    #   1) If applicable: Presets only  **must be run TWICE to ensure that presets are properly updated**
    #   2) Audio and spectrograms
    #   3) Additional data: labels, timbre features

    #gen_dexed_dataset(False, False, False,
    #                  False, False, True)
    gen_surge_dataset(False, False, False,
                      False, False, True)
    gen_nsynth_dataset(False, False,
                       False, False, True)

