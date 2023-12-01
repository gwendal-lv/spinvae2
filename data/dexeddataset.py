"""
Implementation of the DexedDataset, based on the PresetBased abstract class.

Wav files, spectrograms statistics, etc., can be re-generated by running this script as main.
See end of file.
"""
import os
import shutil
import sys
import warnings
from typing import Optional, Iterable, List, Dict, Tuple, Sequence
import multiprocessing
from datetime import datetime

import torch
import torch.utils
import soundfile
import numpy as np

from synth import dexed
from data import abstractbasedataset  # 'from .' raises ImportError when run from PyCharm as __main__
# from data.preset import DexedPresetsParams, PresetIndexesHelper  # Deprecated now
from data.preset2d import Preset2dHelper, Preset2d



class DexedDataset(abstractbasedataset.PresetDataset):
    def __init__(self, note_duration, n_fft, fft_hop, Fs,
                 midi_notes=((60, 100),), multichannel_stacked_spectrograms=False,
                 n_mel_bins=-1, mel_fmin=0, mel_fmax=8000,
                 normalize_audio=False, spectrogram_min_dB=-120.0,
                 spectrogram_normalization: Optional[str] = 'min_max',
                 data_storage_root_path: Optional[str] = None,
                 random_seed=0, data_augmentation=True,
                 algos: Optional[List[int]] = None,
                 operators: Optional[List[int]] = None,
                 restrict_to_labels=None, constant_filter_and_tune_params=True,
                 constant_middle_C=True,
                 prevent_SH_LFO=False,  # TODO re-implement
                 learn_mod_wheel_params=True,  # see dexed.py / get_mod_wheel_related_param_indexes()
                 check_constrains_consistency=True,
                 timbre_attributes: Optional[Sequence[str]] = tuple(),
                 n_extra_random_presets=0
                 ):
        """
        Allows access to Dexed preset values and names, and generates spectrograms and corresponding
        parameters values. Can manage a reduced number of synth parameters (using default values for non-
        learnable params). Only Dexed-specific ctor args are described - see base PresetDataset class.

        It uses both the SQLite DB (through dexed.PresetDatabase) and the pre-written files extracted from
        the DB (see dexed.PresetDatabase.write_all_presets_to_files(...)).

        :param algos: List. Can be used to limit the DX7 algorithms included in this dataset. Set to None
            to use all available algorithms
        :param operators: List of ints, or None. Enables the specified operators only, or all of them if None.
        :param restrict_to_labels: List of strings. If not None, presets of this dataset will be selected such
            that they are tagged with at least one of the given labels.
        :param constant_filter_and_tune_params: if True, the main filter and the detune settings are set to default.
        :param constant_middle_C: If True, the 'Middle C' transpose control will be set to its center position.
        :param prevent_SH_LFO: if True, replaces the SH random LFO by a square-wave deterministic LFO
        :param check_constrains_consistency: Set to False when this dataset instance is used to pre-render
            audio files
        :param n_extra_random_presets: see get_UID_splits(n_extra_random_UIDs) from synth/dexed.py
        """
        super().__init__(note_duration, n_fft, fft_hop, Fs,
                         midi_notes, multichannel_stacked_spectrograms, n_mel_bins, mel_fmin, mel_fmax,
                         normalize_audio, spectrogram_min_dB, spectrogram_normalization, data_storage_root_path,
                         random_seed, data_augmentation, learn_mod_wheel_params, timbre_attributes)
        assert learn_mod_wheel_params  # Must be learned, because LFO modulation also depends on these params
        self.prevent_SH_LFO = prevent_SH_LFO
        if prevent_SH_LFO:
            raise NotImplementedError()  # TODO re-implement S&H enable/disable
        self.constant_filter_and_tune_params = constant_filter_and_tune_params
        self.constant_middle_C = constant_middle_C
        self.algos = algos if algos is not None else []
        self._operators = operators if operators is not None else [1, 2, 3, 4, 5, 6]
        self.restrict_to_labels = restrict_to_labels
        # Pandas Dataframe db, always kept in memory (very fast loading and access)
        self._dexed_db = dexed.PresetDfDatabase(n_extra_random_presets > 0)
        # - - - Constraints on parameters, learnable VST parameters - - -
        self.learnable_params_idx = list(range(0, self.total_nb_vst_params))
        if self.constant_filter_and_tune_params:  # (see dexed_db_explore.ipynb)
            for vst_idx in [0, 1, 2, 3]:
                self.learnable_params_idx.remove(vst_idx)
        if self.constant_middle_C:
            self.learnable_params_idx.remove(13)
        for i_op in range(6):  # Search for disabled operators
            if not (i_op+1) in self._operators:  # If disabled: we remove all corresponding learnable params
                for vst_idx in range(21):  # Don't remove the 22nd param (OP on/off selector) yet
                    self.learnable_params_idx.remove(23 + 22*i_op + vst_idx)  # idx 23 is the first param of op 1
        # Oscillators can be enabled or disabled, but OP SWITCHES are never learnable parameters
        for col in [44, 66, 88, 110, 132, 154]:
            self.learnable_params_idx.remove(col)
        # Mod-wheel related params?
        if not self.learn_mod_wheel_params:
            for vst_param_idx in dexed.Dexed.get_mod_wheel_related_param_indexes():
                # Some might have been removed already (deactivated operators)
                if vst_param_idx in self.learnable_params_idx:
                    self.learnable_params_idx.remove(vst_param_idx)
        # - - - Valid presets - UIDs of presets, and not their database row index - - -
        # Select valid presets by algorithm
        if len(self.algos) == 0:  # All presets are initially considered valid (some might be excluded later...)
            self.valid_preset_UIDs = self._dexed_db.all_preset_UIDs
        else:
            raise AssertionError("All DX7 algorithms must be used - obsolete functionality")
            if len(self.algos) == 1:
                self.learnable_params_idx.remove(4)  # Algo parameter column idx
            valid_presets_row_indexes = dexed_db.get_preset_indexes_for_algorithms(self.algos)
            self.valid_preset_UIDs = dexed_db.all_presets_df\
                .iloc[valid_presets_row_indexes]['index_preset'].values
        # Select valid presets by label. We build a list of list-indexes to remove
        if self.restrict_to_labels is not None:
            raise AssertionError("All labels must be used")
            self.valid_preset_UIDs = [uid for uid in self.valid_preset_UIDs
                                      if any([self.is_label_included(l) for l in self.get_labels_name(uid)])]
        self.n_extra_random_presets = n_extra_random_presets
        # - - - Parameters constraints, cardinality, indexes management, ... - - -
        # Param cardinalities are stored - Dexed cardinality involves a short search which can be avoided
        self._params_cardinality = np.asarray([dexed.Dexed.get_param_cardinality(idx)
                                               for idx in range(self.total_nb_vst_params)])
        self._params_default_values = dict()
        # Algo cardinality is manually set. We consider an algo-limited DX7 to be a new synth
        if len(self.algos) > 0:  # len 0 means all algorithms are used
            self._params_cardinality[4] = len(self.algos)
        if len(self.algos) == 1:  # 1 algo: constrained constant param
            self._params_default_values[4] = (self.algos[0] - 1) / 31.0
        # cardinality 1 for constrained parameters (operators are always constrained)
        self._params_cardinality[[44, 66, 88, 110, 132, 154]] = np.ones((6,), dtype=int)
        for op_i, op_switch_idx in enumerate([44, 66, 88, 110, 132, 154]):
            self._params_default_values[op_switch_idx] = 1.0 if ((op_i+1) in self._operators) else 0.0
        if self.constant_filter_and_tune_params:
            self._params_cardinality[[0, 1, 2, 3]] = np.ones((4,), dtype=int)
            self._params_default_values[0] = 1.0  # 'cutoff'
            self._params_default_values[1] = 0.0  # 'resonance'
            self._params_default_values[2] = 1.0  # 'output' (volume)
            self._params_default_values[3] = 0.5  # 'master tune adj' (small detune)
        if self.constant_middle_C:
            self._params_cardinality[13] = 1
            self._params_default_values[13] = 0.5  # 'transpose'
        if not self.learn_mod_wheel_params:
            mod_vst_params_indexes = dexed.Dexed.get_mod_wheel_related_param_indexes()
            self._params_cardinality[mod_vst_params_indexes] = np.ones((len(mod_vst_params_indexes),), dtype=int)
            for vst_param_idx in mod_vst_params_indexes:
                self._params_default_values[vst_param_idx] = 0.0  # Default: no modulation when MIDI mod wheel changes
        # - - - None / Numerical / Categorical learnable status array - - -
        self._vst_param_learnable_model = list()
        for vst_idx in range(self.total_nb_vst_params):  # We go through all VST params indexes
            if vst_idx not in self.learnable_params_idx:
                self._vst_param_learnable_model.append(None)
            else:
                if vst_idx in dexed.Dexed.get_numerical_params_indexes():
                    self._vst_param_learnable_model.append('num')
                else:
                    self._vst_param_learnable_model.append('cat')
        # - - - Final initializations - - -
        self._preset_idx_helper = Preset2dHelper(self)
        # Don't need to load spectrograms stats here anymore: will be loaded on demand
        if check_constrains_consistency:  # check consistency of pre-rendered audio files
            self.check_audio_render_constraints_file()

    @property
    def synth_name(self):
        return "Dexed"

    def __str__(self):
        return "{}\nRestricted to labels: {}. Enabled algorithms: {}. Enabled operators: {}"\
            .format(super().__str__(), self.restrict_to_labels,
                    ('all' if len(self.algos) == 0 else self.algos), self._operators_config_description)

    @property
    def total_nb_presets(self):
        return self._dexed_db.nb_presets

    def get_split_UIDs(self, split_name: str, exclude_zero_volume_UIDs=True):
        """ Returns a list of UIDs corresponding to a given dataset split (train, validation or test), using
         fixed splits. """
        split_UIDs = dexed.get_UID_splits(self.n_extra_random_presets)[split_name]
        if exclude_zero_volume_UIDs:
            for zero_volume_UID in self.zero_volume_preset_UIDs():
                if zero_volume_UID in split_UIDs:
                    split_UIDs.remove(zero_volume_UID)
        # quick error/consistency check: all of these UIDs must be actual valid UIDs for this dataset
        for UID in split_UIDs:
            assert UID in self.valid_preset_UIDs
        return split_UIDs

    def get_name_from_preset_UID(self, preset_UID: int, long_name=False) -> str:
        """ Returns the preset's name.
        If long_name is True, the original DX7 cartridge name is appended at the end. """
        return self._dexed_db.get_preset_name(preset_UID, long_name)

    def get_cartridge_name_from_preset_UID(self, preset_UID: int) -> str:
        return self._dexed_db.get_cartridge_name_from_preset_UID(preset_UID)

    def get_original_instrument_family(self, preset_UID: int) -> str:
        return "Cartridge: '{}'".format(self.get_cartridge_name_from_preset_UID(preset_UID))

    # ============================== Presets and parameters (PresetDataset only) =============================

    @property
    def vst_param_learnable_model(self):
        return self._vst_param_learnable_model

    @property
    def numerical_vst_params(self):
        return dexed.Dexed.get_numerical_params_indexes()

    @property
    def categorical_vst_params(self):
        return dexed.Dexed.get_categorical_params_indexes()

    @property
    def params_default_values(self):
        return self._params_default_values

    @property
    def total_nb_vst_params(self):
        return self._dexed_db.nb_params_per_preset

    @property
    def preset_indexes_helper(self):
        return self._preset_idx_helper

    @property
    def preset_param_names(self):
        return self._dexed_db.param_names

    @property
    def preset_param_types(self) -> List[str]:
        return dexed.Dexed.get_param_types(operator_index=False)  # All operators will have the same param types

    def get_preset_param_cardinality(self, idx, learnable_representation=True):
        if idx == 4 and learnable_representation is False:
            return 32  # Algorithm is always an annoying special case... could be improved
        return self._params_cardinality[idx]

    def _pre_process_presets_database(self, verbose=True):
        """ re-generate the dataframe (from the slow-to-read SQLite original database) """
        self._dexed_db.save_sqlite_to_df(verbose=verbose)

    def get_full_preset_params(self, preset_UID, preset_variation=0):
        raw_full_preset = self._dexed_db.get_preset_params_values(preset_UID)
        if preset_variation > 0:
            # send VST indexes which are learned (data augmentation on these ones only)
            raw_full_preset = dexed.Dexed.get_similar_preset(
                raw_full_preset, preset_variation, self.learnable_params_idx, random_seed=preset_UID)
        return Preset2d(self, raw_full_preset)

    # ================================== Constraints (on presets' parameters) =================================

    @property
    def _operators_config_description(self) -> str:
        """ Returns a description (to be used in file and directory names) that describes enabled DX7 operators,
        as configured in this dataset's constructor. """
        if self._operators != [1, 2, 3, 4, 5, 6]:
            ops_description = 'operators_' + ''.join(['{}'.format(op) for op in self._operators])
        else:
            ops_description = 'operators_all'
        return ops_description

    @property
    def audio_constraints(self):
        constraints = super().audio_constraints
        constraints['operators'] = self._operators_config_description
        constraints['constant_filter_and_tune_params'] = self.constant_filter_and_tune_params
        constraints['prevent_SH_LFO'] = self.prevent_SH_LFO
        constraints['constant_middle_C'] = self.constant_middle_C
        return constraints

    # ================================== Labels =================================

    def is_label_included(self, label):
        """ Returns True if the label belongs to the restricted labels list. """
        warnings.warn("Deprecated", DeprecationWarning)
        if self.restrict_to_labels is None:
            return True
        else:
            return any([label == l_ for l_ in self.restrict_to_labels])

    def get_labels_tensor(self, preset_UID):
        return torch.tensor(self._dexed_db.get_labels_array_from_UID(preset_UID), dtype=torch.int8)

    def get_labels_name(self, preset_UID):
        return self._dexed_db.get_labels_str_from_UID(preset_UID)

    def save_labels(self, labels_names: List[str], labels_per_UID: Dict[int, List[str]]):
        super().save_labels(labels_names, labels_per_UID)
        dexed.PresetDfDatabase.update_labels_in_pickled_df(labels_names, labels_per_UID)


    # ================================== Audio files =================================

    @property
    def _nb_preset_variations_per_note(self):
        return 4

    @property
    def _nb_audio_delay_variations_per_note(self):
        return 2

    def get_audio_file_stem(self, preset_UID, midi_note, midi_velocity, variation=0):
        return "{:06d}_pitch{:03d}vel{:03d}_var{:03d}".format(preset_UID, midi_note, midi_velocity, variation)

    def _get_wav_file_path(self, patch_UID, midi_pitch, midi_vel, variation):
        return self.audio_storage_path.joinpath("{}.wav".format(self.get_audio_file_stem(patch_UID, midi_pitch,
                                                                                          midi_vel, variation)))

    def get_wav_file(self, preset_UID, midi_note, midi_velocity, variation=0):
        # The 'audio' variation is computed when rendering spectrograms only (they use the same audio file, but
        # this method rolls the audio np array).
        # The 'preset' variation, however, changes the sound (we need to render/load a different wav file).
        # So:
        #    - load the same for different delays
        #    - apply a delay if necessary
        preset_variation, audio_delay = self._get_variation_args(variation)
        new_variation = self._get_variation_index_from_args(preset_variation, audio_delay=0)
        file_path = self._get_wav_file_path(preset_UID, midi_note, midi_velocity, new_variation)
        try:
            audio, Fs = soundfile.read(file_path)
            if audio_delay > 0:
                audio = self.pseudo_random_audio_delay(audio, random_seed=(self._random_seed + preset_UID + audio_delay))
            return audio, Fs
        except RuntimeError:
            raise RuntimeError("[data/dexeddataset.py] Can't open file {}. Please pre-render audio files for this "
                               "dataset configuration.".format(file_path))

    def generate_wav_files(self):
        """ Reads all presets (names, param values, and labels) from .pickle and .txt files
         (see dexed.PresetDatabase.write_all_presets_to_files(...)) and renders them
         using attributes and constraints of this class (midi note, normalization, etc...)

         Floating-point .wav files will be stored in dexed presets' folder (see synth/dexed.py)

         Also writes a audio_render_constraints.json file that should be checked when loading data.
         """
        print("Dexed audio files rendering...")
        t_start = datetime.now()
        # Handle previously written files and folders
        if os.path.exists(self.audio_storage_path):
            shutil.rmtree(self.audio_storage_path)
        self.audio_storage_path.mkdir(parents=True, exist_ok=False)
        self.write_audio_render_constraints_file()
        self._delete_all_spectrogram_data()
        # multi-processed audio rendering
        if sys.gettrace() is None:  # if No PyCharm debugger
            num_workers = os.cpu_count()
            split_preset_UIDs = np.array_split(self.valid_preset_UIDs, num_workers)
            with multiprocessing.Pool(num_workers) as p:  # automatically closes and joins all workers
                p.map(self._generate_wav_files_batch, split_preset_UIDs)
        else:  # Debugging
            self._generate_wav_files_batch(self.valid_preset_UIDs)
        # final display
        delta_t = (datetime.now() - t_start).total_seconds()
        num_wav_written = len(self.valid_preset_UIDs) * len(self.midi_notes) * self._nb_preset_variations_per_note
        print("Finished writing {} .wav files ({:.1f} min total, {:.1f} ms / file)"
              .format(num_wav_written, delta_t/60.0, 1000.0*delta_t/num_wav_written))

    def _generate_wav_files_batch(self, preset_UIDs):
        """ Generates all audio files for a given list of preset UIDs. """
        # don't generate files for >0 audio delays
        audio_delay = 0
        for UID in preset_UIDs:
            for note in self.midi_notes:
                for preset_variation in range(self._nb_preset_variations_per_note):
                    variation = self._get_variation_index_from_args(preset_variation, audio_delay)
                    self._generate_single_wav_file(UID, note[0], note[1], variation)

    def _generate_single_wav_file(self, preset_UID, midi_pitch, midi_velocity, variation):
        preset_variation, audio_delay = self._get_variation_args(variation)
        if audio_delay > 0:
            # We do not need to render all variations (do not render delayed audio to save some SSD storage)
            raise ValueError("Audio files should all be rendered with a 0 note-on delay.")
        # Constrained params (1-element batch)
        preset_params = self.get_full_preset_params(preset_UID, preset_variation)
        x_wav, Fs = self._render_audio(preset_params.to_raw(), midi_pitch, midi_velocity)  # Slow: re-Loads the VST
        soundfile.write(self._get_wav_file_path(preset_UID, midi_pitch, midi_velocity, variation),
                        x_wav, Fs, subtype='FLOAT')

    def _render_audio(self, preset_params: Iterable, midi_note, midi_velocity,
                      custom_note_duration: Tuple[int, int] = None):
        """ Does not require a 'variation' (preset_params must have been modified accordingly, before calling
        this method) """
        note_duration = custom_note_duration if custom_note_duration is not None else self.note_duration
        # We always have to reload the VST to prevent hanging notes/sounds
        dexed_renderer = dexed.Dexed(output_Fs=self.Fs,
                                     midi_note_duration_s=note_duration[0],
                                     render_duration_s=note_duration[0] + note_duration[1])
        dexed_renderer.assign_preset(dexed.PresetDatabase.get_params_in_plugin_format(preset_params))
        x_wav, Fs = dexed_renderer.render_note(midi_note, midi_velocity, normalize=self.normalize_audio)
        return x_wav, Fs

