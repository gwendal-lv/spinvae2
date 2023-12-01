"""
Dexed VSTi audio renderer and presets database reader classes.

More information about the original DX7 paramaters:
https://www.chipple.net/dx7/english/edit.mode.html
https://djjondent.blogspot.com/2019/10/yamaha-dx7-algorithms.html
"""
import contextlib
import json
import socket
import sys
import os
import pickle
import multiprocessing
from multiprocessing.pool import ThreadPool
import time
from typing import Iterable, List, Dict
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
import pathlib

import librosa
import numpy as np
import sqlite3
import io
import pandas as pd

import synth.dexedbase
import librenderman as rm  # A symbolic link to the actual librenderman.so must be found in the current folder

import utils.text


# Pickled numpy arrays storage in sqlite3 DB
def adapt_array(arr):
    """ http://stackoverflow.com/a/31312102/190597 (SoulNibbler) """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)
# Converts TEXT to np.array when selecting
sqlite3.register_converter("NPARRAY", convert_array)


def get_partial_presets_df(db_row_index_limits):
    """ Returns a partial dataframe of presets from the DB, limited a tuple of row indexes
    (first and last included).

    Useful for fast DB reading, because it involves a lot of unpickling which can be parallelized. """
    conn = PresetDatabaseABC._get_db_connection()
    nb_rows = db_row_index_limits[1] - db_row_index_limits[0] + 1
    presets_df = pd.read_sql_query("SELECT * FROM preset LIMIT {} OFFSET {}"
                                   .format(nb_rows, db_row_index_limits[0]), conn)
    conn.close()
    return presets_df


def get_UID_splits(n_extra_random_UIDs=0):
    """
    Returns training/validation/test splits, by UID, which can be loaded from a local JSON file.

    :param n_extra_random_UIDs: Number of randomly-generated UIDs to be appended to the training split.
    """
    with open(pathlib.Path(__file__).parent.joinpath('dexed_UID_splits.json'), 'r') as f:
        original_splits = json.load(f)
    if n_extra_random_UIDs > 0:
        with open(pathlib.Path(__file__).parent.joinpath('dexed_UID_splits__extra_random.json'), 'r') as f:
            extra_UIDs = json.load(f)
        assert list(extra_UIDs.keys()) == ['train'], "Extra random presets should be assigned to the train split only"
        assert n_extra_random_UIDs <= len(extra_UIDs['train']), f"max extra presets = {len(extra_UIDs['train'])}"
        original_splits['train'] += extra_UIDs['train'][0:n_extra_random_UIDs]
    return original_splits



class PresetDatabaseABC(ABC):
    def __init__(self):
        # We also pre-load the names in order to close the sqlite DB
        conn = self._get_db_connection()
        names_df = pd.read_sql_query("SELECT * FROM param ORDER BY index_param", conn)
        conn.close()
        self._param_names = names_df['name'].to_list()

    @staticmethod
    def _get_db_path():
        return pathlib.Path(__file__).parent.joinpath('dexed_presets.sqlite')  # pkgutil would be better

    @staticmethod
    def _get_db_connection():
        db_path = PresetDatabaseABC._get_db_path()
        return sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)

    @property
    @abstractmethod
    def nb_presets(self) -> int:
        pass

    @abstractmethod
    def get_preset_name(self, preset_UID: int) -> str:
        pass

    @property
    @abstractmethod
    def nb_params_per_preset(self) -> int:
        pass

    @property
    def param_names(self):
        return self._param_names

    @staticmethod
    def get_params_in_plugin_format(params: Iterable):
        """ Converts a 1D array of param values into an list of (idx, param_value) tuples """
        preset_values = np.asarray(params, dtype=np.double)  # np.float32 is not valid for RenderMan  TODO use built-in float instead of np.double? (deprecated?)
        # Dexed parameters are nicely ordered from 0 to 154
        return [(i, preset_values[i]) for i in range(preset_values.shape[0])]



class PresetDatabase(PresetDatabaseABC):
    def __init__(self, num_workers=None):
        """ DEPRECATED - Opens the SQLite DB and copies all presets internally. This uses a lot of memory
        but allows easy multithreaded usage from multiple parallel dataloaders (1 db per dataloader). """
        warnings.warn("PresetDatabase class uses the original SQLite database, which is very slow and "
                      "is not up to date. Please use PresetDfDatabase instead", DeprecationWarning)
        super().__init__()
        self._db_path = self._get_db_path()
        conn = sqlite3.connect(self._db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        cur = conn.cursor()
        # We load the full presets table (full DB is usually a few dozens of megabytes)
        self.all_presets_df = self._load_presets_df_multiprocess(conn, cur, num_workers)
        # 20 megabytes for 30 000 presets
        self.presets_mat = self.all_presets_df['pickled_params_np_array'].values
        self.presets_mat = np.stack(self.presets_mat)
        # Memory save: param values are removed from the main dataframe
        self.all_presets_df.drop(columns='pickled_params_np_array', inplace=True)
        # Algorithms are also separately stored
        self._preset_algos = self.presets_mat[:, 4]
        self._preset_algos = np.asarray(np.round(1.0 + self._preset_algos * 31.0), dtype=int)
        conn.close()

    def _load_presets_df_multiprocess(self, conn, cur, num_workers):
        if num_workers is None:
            num_workers = os.cpu_count() // 2
        cur.execute('SELECT COUNT(1) FROM preset')
        presets_count = cur.fetchall()[0][0]
        num_workers = np.minimum(presets_count, num_workers)
        # The last process might have a little more work to do
        rows_count_by_proc = presets_count // num_workers
        row_index_limits = list()
        for n in range(num_workers-1):
            row_index_limits.append([n * rows_count_by_proc, (n+1) * rows_count_by_proc - 1])
        # Last proc takes the remaining
        row_index_limits.append([(num_workers-1)*rows_count_by_proc, presets_count-1])
        if sys.gettrace() is not None:  # PyCharm debugger detected (should work with others)
            with ThreadPool(num_workers) as p:  # multiproc breaks PyCharm remote debug
                partial_presets_dfs = p.map(get_partial_presets_df, row_index_limits)
        else:
            with multiprocessing.Pool(num_workers) as p:
                partial_presets_dfs = p.map(get_partial_presets_df, row_index_limits)
        return pd.concat(partial_presets_dfs)

    def __str__(self):
        return "{} DX7 presets in database '{}'.".format(len(self.all_presets_df), self._db_path)

    def get_preset_values(self, idx, plugin_format=False):  # FIXME move to ABC mother class
        """ Returns a preset from the DB.

        :param idx: the preset 'row line' in the DB (not the index_preset value, which is an ID)
        :param plugin_format: if True, returns a list of (param_index, param_value) tuples. If False, returns the
            numpy array of param values. """
        preset_values = self.presets_mat[idx, :]
        if plugin_format:
            return self.get_params_in_plugin_format(preset_values)
        else:
            return preset_values

    def get_param_names(self):
        return self._param_names

    def get_preset_indexes_for_algorithms(self, algos):
        """ Returns a list of indexes of presets using the given algorithms in [[1 ; 32]] """
        indexes = []
        for i in range(self._preset_algos.shape[0]):
            if self._preset_algos[i] in algos:
                indexes.append(i)
        return indexes

    def get_size_info(self):
        """ Prints a detailed view of the size of this class and its main elements """
        main_df_size = self.all_presets_df.memory_usage(deep=True).values.sum()
        preset_values_size = self.presets_mat.size * self.presets_mat.itemsize
        return "Dexed Presets Database class size: " \
               "preset values matrix {:.1f} MB, presets dataframe {:.1f} MB"\
            .format(preset_values_size/(2**20), main_df_size/(2**20))

    @staticmethod
    def _get_presets_folder():
        return pathlib.Path(__file__).parent.absolute().joinpath('dexed_presets')

    def write_all_presets_to_files(self, verbose=True):
        """ Write all presets' parameter values to separate pickled files, for multi-processed multi-worker
        DataLoader. File names are presetXXXXXX_params.pickle where XXXXXX is the preset UID (it is not
        its row index in the SQLite database).

        Presets' names will be written to presetXXXXXX_name.txt,
        and comma-separated labels to presetXXXXXX_labels.txt.

        Performs consistency checks (e.g. labels, ...). TODO implement all consistency checks

        All files will be written to ./dexed_presets/ """
        presets_folder = self._get_presets_folder()
        if not os.path.exists(presets_folder):
            os.makedirs(presets_folder)
        for i in range(len(self.presets_mat)):
            preset_UID = self.all_presets_df.iloc[i]['index_preset']
            param_values = self.presets_mat[i, :]
            base_name = "preset{:06d}_".format(preset_UID)
            # ((un-)pickling has been done far too many times for these presets... could have been optimized)
            with open(presets_folder.joinpath(base_name + "params.pickle"), 'wb') as f:
                pickle.dump(param_values, f)
            with open(presets_folder.joinpath(base_name + "name.txt"), 'w') as f:
                f.write(self.all_presets_df.iloc[i]['name'])
            with open(presets_folder.joinpath(base_name + "labels.txt"), 'w') as f:
                labels = self.all_presets_df.iloc[i]['labels']
                labels_list = labels.split(',')
                for l in labels_list:
                    if not any([l == l_ for l_ in self.get_available_labels()]):  # Checks if any is True
                        raise ValueError("Label '{}' should not be available in self.all_presets_df".format(l))
                f.write(labels)
        if verbose:
            print("[dexed.PresetDatabase] Params, names and labels from SQLite DB written as .pickle and .txt files")

    @staticmethod
    def get_preset_params_values_from_file(preset_UID):
        warnings.warn("PresetDatabase class uses the original SQLite database, which is very slow and "
                      "is not up to date. Please use PresetDfDatabase instead", DeprecationWarning)
        return np.load(PresetDatabase._get_presets_folder()
                       .joinpath( "preset{:06d}_params.pickle".format(preset_UID)), allow_pickle=True)

    @staticmethod
    def get_preset_name_from_file(preset_UID):
        warnings.warn("PresetDatabase class uses the original SQLite database, which is very slow and "
                      "is not up to date. Please use PresetDfDatabase instead", DeprecationWarning)
        with open(PresetDatabase._get_presets_folder()
                  .joinpath( "preset{:06d}_name.txt".format(preset_UID)), 'r') as f:
            name = f.read()
        return name

    @staticmethod
    def get_available_labels():
        raise DeprecationWarning("These labels were extracted from the HPSS analysis and are obsolete.")
        return 'harmonic', 'percussive', 'sfx'

    @staticmethod
    def get_preset_labels_from_file(preset_UID):
        """ Return the preset labels as a list of strings. """
        with open(PresetDatabase._get_presets_folder()
                  .joinpath("preset{:06d}_labels.txt".format(preset_UID)), 'r') as f:
            labels = f.read()
        return labels.split(',')



class PresetDfDatabase(PresetDatabaseABC):
    def __init__(self, include_extra_random_presets=False):
        super().__init__()
        with open(PresetDfDatabase._get_dataframe_db_path(), 'rb') as f:
            self._presets_df = pickle.load(f)
        # remove randomly generated presets (data augmentation) UIDs if not requested
        if not include_extra_random_presets:
            self._presets_df = self._presets_df[self._presets_df['preset_UID'] < 400000]
        # Build UID -> local idx dict, for faster access to data by UID
        self._UID_to_local_idx = {self._presets_df.iloc[idx]['preset_UID']: idx
                                  for idx in range(len(self._presets_df))}
        # TODO get available labels

    @property
    def nb_presets(self) -> int:
        return len(self._presets_df)

    def get_preset_name(self, preset_UID: int, long_name=False) -> str:
        df_idx = self._UID_to_local_idx[preset_UID]
        name = self._presets_df.at[df_idx, 'name']
        if long_name:
            name += ' ({})'.format(self._presets_df.at[df_idx, 'cartridge_name'])
        return name

    def get_cartridge_name_from_preset_UID(self, preset_UID: int) -> str:
        return self._presets_df.at[self._UID_to_local_idx[preset_UID], 'cartridge_name']

    def get_preset_params_values(self, preset_UID: int):
        return self._presets_df.at[self._UID_to_local_idx[preset_UID], 'params_values']

    def get_labels_str_from_UID(self, preset_UID: int) -> str:
        return self._presets_df.at[self._UID_to_local_idx[preset_UID], 'instrument_labels_str']

    def get_labels_array_from_UID(self, preset_UID: int):
        # at is approx. 30x faster than .iloc then select col
        return self._presets_df.at[self._UID_to_local_idx[preset_UID], 'instrument_labels_array']

    @property
    def nb_params_per_preset(self) -> int:
        return self._presets_df.at[0, 'params_values'].shape[0]

    @property
    def all_preset_UIDs(self):
        return self._presets_df['preset_UID'].values

    @staticmethod
    def _get_manual_instr_labels_path():
        return pathlib.Path(__file__).parent.joinpath('dexed_manual_instr_labels.json')

    @staticmethod
    def _get_dataframe_db_path():
        return pathlib.Path(__file__).parent.joinpath('dexed_presets.df.pickle')

    @staticmethod
    def save_sqlite_to_df(verbose=False):
        """ Saves the reference .sqlite presets database into an equivalent pandas dataframe. """
        t_start = datetime.now()
        conn = PresetDfDatabase._get_db_connection()
        presets_df = pd.read_sql_query("SELECT * FROM preset", conn)
        cartridges_df = pd.read_sql_query("SELECT * FROM cartridge", conn)
        params_info_df = pd.read_sql_query("SELECT * FROM param", conn)  # Contains params' names
        conn.close()
        if verbose:
            print("[PresetDfDatabase] SQLite tables were read in {:.1f}s ({} presets)"
                  .format((datetime.now() - t_start).total_seconds(), len(presets_df)))
        # Post-processing: remove/rename SQLite columns, add cartridges names
        presets_df = presets_df.rename(columns={"index_preset": "preset_UID"})  # Corresponding Python variable name
        presets_df = presets_df.rename(columns={"pickled_params_np_array": "params_values"})
        presets_df = presets_df.drop(columns=['other_names'])
        presets_df = presets_df.rename(columns={"labels": "hpss_labels"})  # DAFx21 paper: Harmonic-Percussive labels
        cartridge_names = list()
        for cart_idx in presets_df['index_cart'].values:  # Very slow search... but done only once
            name = cartridges_df[cartridges_df['index_cart'] == cart_idx]['name'].values  # array
            if len(name) != 1:
                raise ValueError('index_cart should be unique in the database')
            cartridge_names.append(name[0])
        presets_df['cartridge_name'] = cartridge_names
        # Save the full df
        with open(PresetDfDatabase._get_dataframe_db_path(), 'wb') as f:
            pickle.dump(presets_df, f)  # Pickle Reloading takes < 0.0 ms from the SSD

    @staticmethod
    def update_labels_in_pickled_df(available_labels: List[str], labels_per_UID: Dict[int, List[str]]):
        with open(PresetDfDatabase._get_dataframe_db_path(), 'rb') as f:
            presets_df = pickle.load(f)
        # Also load the manually-encoded labels - they will override any provided input arg
        with open(PresetDfDatabase._get_manual_instr_labels_path(), 'r') as f:
            manual_instr_labels = json.load(f)  # JSON keys must be string, to be parsed as int
        manual_instr_labels = {int(k): v for k, v in manual_instr_labels.items()}

        # check if labels column already exists (discard if yes)
        if 'instrument_labels_str' in presets_df.columns:
            presets_df.drop(columns=['instrument_labels_str', 'instrument_labels_array'], inplace=True)

        # add labels to the existing dataframe - also convert labels to numpy arrays
        labels_str, labels_arrays = list(), list()
        for row in presets_df.iterrows():  # 1-by-1 processing, in case a reordering had happened
            # handle exception? All labels should be available at this point... (even for rejected presets)
            # Add SFX here
            row = row[1]  # Row is a Tuple(int, Series)
            current_UID = row['preset_UID']
            if current_UID in manual_instr_labels.keys():  # Manual high-confidence labels
                current_labels = manual_instr_labels[current_UID]
            else:  # Labels automatically extracted from name and HPSS
                current_labels = labels_per_UID[current_UID]
                if row['hpss_labels'] is not None and 'sfx' in row['hpss_labels']:
                    if 'sfx' not in current_labels:
                        current_labels.append('sfx')
            labels_str.append(current_labels)
            labels_arrays.append(
                np.asarray([(ref_label in current_labels) for ref_label in available_labels], dtype=np.uint8)
            )
        presets_df['instrument_labels_str'] = labels_str
        presets_df['instrument_labels_array'] = labels_arrays
        with open(PresetDfDatabase._get_dataframe_db_path(), 'wb') as f:  # Write the update DataFrame
            pickle.dump(presets_df, f)



class Dexed(synth.dexedbase.DexedCharacteristics):
    """ A Dexed (DX7) synth that can be used through RenderMan for offline wav rendering. """

    def __init__(self, output_Fs, render_Fs=48000,
                 plugin_relative_path="Dexed.so",
                 midi_note_duration_s=3.0, render_duration_s=4.0,
                 buffer_size=512, fft_size=512,
                 fadeout_duration_s=0.0,  # Default: disabled,
                 filter_plugin_loading_errors=True,
                 ):
        """
        :param filter_plugin_loading_errors: Filters the numerous JUCE "No protocol specific" warnings, which are
            displayed when using the VST plug-in. Use False if the synth does not load (e.g. from a Jupyter Notebook).
        """
        super().__init__()
        self.fadeout_duration_s = fadeout_duration_s  # To reduce STFT discontinuities with long-release presets
        self.midi_note_duration_s = midi_note_duration_s
        self.render_duration_s = render_duration_s

        self.plugin_path = str(pathlib.Path(__file__).parent.joinpath(plugin_relative_path))
        self.render_Fs = render_Fs
        self.reduced_Fs = output_Fs
        self.buffer_size = buffer_size
        self.fft_size = fft_size  # FFT not used

        self.engine = rm.RenderEngine(self.render_Fs, self.buffer_size, self.fft_size)
        with utils.text.hidden_prints(filter_stderr=True) if filter_plugin_loading_errors else contextlib.nullcontext():
            self.engine.load_plugin(self.plugin_path)  # filter the "No protocol specified" double error msg

        # A generator preset is a list of (int, float) tuples.
        self.preset_gen = rm.PatchGenerator(self.engine)  # 'RenderMan' generator
        self.current_preset = None

    def __str__(self):
        return "Plugin loaded from {}, Fs={}Hz (output downsampled to {}Hz), buffer {} samples."\
               "MIDI note on duration: {:.1f}s / {:.1f}s total."\
            .format(self.plugin_path, self.render_Fs, self.reduced_Fs, self.buffer_size,
                    self.midi_note_duration_s, self.render_duration_s)

    def render_note(self, midi_note, midi_velocity, normalize=False):
        """ Renders a midi note (for the currently set patch) and returns the (normalized) float array and
         associated sampling rate. """
        self.engine.render_patch(midi_note, midi_velocity, self.midi_note_duration_s, self.render_duration_s)
        audio_out = self.engine.get_audio_frames()
        audio = np.asarray(audio_out)
        fadeout_len = int(np.floor(self.render_Fs * self.fadeout_duration_s))
        if fadeout_len > 1:  # fadeout might be disabled if too short
            fadeout = np.linspace(1.0, 0.0, fadeout_len)
            audio[-fadeout_len:] = audio[-fadeout_len:] * fadeout
        if normalize:
            audio = audio * (0.99 / np.abs(audio).max())  # to prevent 16-bit conversion clipping
        audio = librosa.resample(audio, self.render_Fs, self.reduced_Fs, res_type="kaiser_best")
        return audio, self.reduced_Fs

    def assign_preset(self, preset):
        """ :param preset: List of tuples (param_idx, param_value) """
        self.current_preset = preset
        self.engine.set_patch(self.current_preset)

    def get_random_preset(self, short_release=False, seed=0):
        """ Returns a uniformely random, properly quantized preset in the VST format (list of (idx, value) tuples). """
        if short_release:
            raise NotImplementedError()
        rng = np.random.default_rng(seed=seed)
        # We don't use the randomly generated patches because they might assign a lower probability to extreme values
        n_params = len(self.preset_gen.get_random_patch())  #
        params_cardinality = np.asarray([self.get_param_cardinality(i) for i in range(n_params)], dtype=int)
        preset_np = rng.integers(params_cardinality * 0, params_cardinality, endpoint=False).astype(float)
        preset_np = preset_np / (params_cardinality.astype(float) - 1.0)
        return [(i, v) for i, v in enumerate(preset_np)]

    def assign_random_preset_short_release(self):  # TODO integrate into the assign_random_preset method
        """ Generates a random preset with a short release time - to ensure a limited-duration
         audio recording, and configures the rendering engine to use that preset. """
        self.current_preset = self.preset_gen.get_random_patch()
        self.set_release_short()
        self.engine.set_patch(self.current_preset)

    def set_release_short(self, eg_4_rate_min=0.5):
        raise AssertionError()  # deprecated - should return the modified params as well
        for i, param in enumerate(self.current_preset):
            idx, value = param  # a param is actually a tuple...
            # Envelope release level: always to zero (or would be an actual hanging note)
            if idx == 30 or idx == 52 or idx == 74 or idx == 96 or idx == 118 or idx == 140:
                self.current_preset[i] = (idx, 0.0)
            # Envelope release time: quite short (higher float value: shorter release)
            elif idx == 26 or idx == 48 or idx == 70 or idx == 92 or idx == 114 or idx == 136:
                self.current_preset[i] = (idx, max(eg_4_rate_min, value))
        self.engine.set_patch(self.current_preset)

    def set_default_general_filter_and_tune_params(self):
        """ Internally sets the modified preset, and returns the list of parameter values. """
        assert self.current_preset is not None
        self.current_preset[0] = (0, 1.0)  # filter cutoff
        self.current_preset[1] = (1, 0.0)  # filter reso
        self.current_preset[2] = (2, 1.0)  # output vol
        self.current_preset[3] = (3, 0.5)  # master tune
        self.current_preset[13] = (13, 0.5)  # Sets the 'middle-C' note to the default C3 value
        self.engine.set_patch(self.current_preset)
        return [v for _, v in self.current_preset]

    @staticmethod
    def set_default_general_filter_and_tune_params_(preset_params):
        """ Modifies some params in-place for the given numpy array """
        preset_params[[0, 1, 2, 3, 13]] = np.asarray([1.0, 0.0, 1.0, 0.5, 0.5])

    def set_all_oscillators_on(self):
        """ Internally sets the modified preset, and returns the list of parameter values. """
        assert self.current_preset is not None
        for idx in [44, 66, 88, 110, 132, 154]:
            self.current_preset[idx] = (idx, 1.0)
        self.engine.set_patch(self.current_preset)
        return [v for _, v in self.current_preset]

    @staticmethod
    def set_all_oscillators_on_(preset_params: np.ndarray):
        """ Modifies some params of the given numpy array to ensure that all operators (oscillators) are ON.
        Data is modified in place. """
        preset_params[[44, 66, 88, 110, 132, 154]] = np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    @staticmethod
    def set_all_oscillators_off_(preset_params: np.ndarray):
        """ Modifies some params of the given numpy array to ensure that all operators (oscillators) are OFF.
        Data is modified in place. """
        preset_params[[44, 66, 88, 110, 132, 154]] = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    @staticmethod
    def set_oscillators_on_(preset_params: np.ndarray, operators_to_turn_on):
        """ Modifies some params of the given numpy array to turn some operators ON. Data is modified in place.

        :param preset_params: Numpy Array of preset parameters values.
        :param operators_to_turn_on: List of integers in [1, 6]
        """
        Dexed.set_all_oscillators_off_(preset_params)
        for op_number in operators_to_turn_on:
            preset_params[44 + 22 * (op_number-1)] = 1.0

    def prevent_SH_LFO(self):
        """ If the LFO Wave is random S&H, transforms it into a square LFO wave to get deterministic
        results. Internally sets the modified preset, and returns the list of parameter values.  """
        if self.current_preset[12][1] > 0.95:  # S&H wave corresponds to a 1.0 param value
            self.current_preset[12] = (12, 4.0 / 5.0)  # Square wave is number 4/6
        self.engine.set_patch(self.current_preset)
        return [v for _, v in self.current_preset]

    @staticmethod
    def prevent_SH_LFO_(preset_params):
        """ Modifies some params in-place for the given numpy array """
        if preset_params[12] > 0.95:
            preset_params[12] = 4.0 / 5.0


if __name__ == "__main__":

    if False:  # TODO remove after usage
        import pickle

        with open("/home/gwendal/Jupyter/spinvae_notebooks/random_presets_mat.np.pickle", 'rb') as f:
            random_presets_mat = pickle.load(f)
        cart_name, cart_idx = "DATA_AUG", 12533

        conn = PresetDatabaseABC._get_db_connection()
        cur = conn.cursor()

        presets_df = pd.read_sql_query("SELECT * FROM preset", conn)

        cartridges_df = pd.read_sql_query("SELECT * FROM cartridge", conn)
        #cur.execute(f"INSERT INTO cartridge (name) VALUES ('{cart_name}')")
        #cartridges_df_after = pd.read_sql_query("SELECT * FROM cartridge", conn)

        for i in range(random_presets_mat.shape[0]):
            # Index offset: 400 000
            preset_UID = 400000 + i
            preset_np = random_presets_mat[i, :].astype(np.float32)
            sql_params = (preset_UID, cart_idx, i, f"RAND{i:06d}", preset_np, "random_preset__noise_0.1")
            cur.execute(
                "INSERT INTO preset (index_preset, index_cart, index_in_cartridge, name, pickled_params_np_array, info)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                sql_params)

        #for uid in UIDs_to_delete:
        #    assert uid in presets_df['index_preset'].values, uid
        #    cur.execute(f"DELETE FROM preset WHERE index_preset = {uid}")
        #conn.commit()
#
        #presets_df_after = pd.read_sql_query("SELECT * FROM preset", conn)

        conn.close()

    if False:
        # Test du synth lui-même
        _dexed = Dexed(16000, filter_plugin_loading_errors=False)
        print(_dexed)
        print("Plugin params: ")
        print(_dexed.engine.get_plugin_parameters_description())

        _preset = _dexed.get_random_preset()
        # pres = dexed.preset_db.get_preset_values(0, plugin_format=True)
        # dexed.assign_preset_from_db(100)
        # print(dexed.current_preset)

        # dexed.render_note(57, 100, filename="Test.wav")

        # Compute the total number of logits, if all param are learned as categorical (full-resolution)
        logits_count = 0
        for _ in Dexed.get_numerical_params_indexes():
            logits_count += 100
        for _i in Dexed.get_categorical_params_indexes():
            logits_count += Dexed.get_param_cardinality(_i)
        print("{} logits, if all param are learned as categorical (full-resolution)".format(logits_count))

        print("{} presets use algo 5".format(len(dexed_db.get_preset_indexes_for_algorithm(5))))


