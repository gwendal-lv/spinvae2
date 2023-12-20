import copy
import json
import multiprocessing
import pathlib
import pickle
import threading
import time
import warnings
from datetime import datetime
from typing import Union, Optional, List

import numpy as np
import pandas as pd
import soundfile

from utils.timbretoolbox import ToolboxLogger, TimbreToolboxProcess, TimbreToolboxResults
import utils.timbral_models


class InterpolationTimbreProcessorBase:
    def __init__(self, data_root_path: Union[str, pathlib.Path], timbre_name: str, n_workers: int, verbose=True):
        self.data_root_path = pathlib.Path(data_root_path)
        self.timbre_name, self.n_workers, self.verbose = timbre_name, n_workers, verbose

        # Are supposed to be interpolation sequence names
        self.audio_sub_folders = sorted([f for f in self.data_root_path.iterdir() if f.is_dir()])
        # Get indices of sequences, using the existing folder structure only
        self.sequence_names = [sub_dir.name for sub_dir in self.audio_sub_folders]
        try:
            self.sequence_indices = [int(name) for name in self.sequence_names]
        except ValueError:
            raise NameError("All sub-folders inside '{}' must be named as integers (indices of interpolation sequences)"
                            .format(self.data_root_path))
        if self.sequence_indices != list(range(len(self.sequence_indices))):
            raise NameError("Sub-folders inside '{}' are not named as a continuous range of indices. Found indices: {}"
                            .format(self.data_root_path, self.sequence_indices))

    def save_raw_sequences_features(self, all_seqs_features: List[pd.DataFrame]):
        """ Stores features from all sequences into 2 pickle files: 1 easy-to-use list of dataframes, 1 dataframe"""
        with open(self.data_root_path.joinpath(f'raw_{self.timbre_name}_seqs_features.list.pkl'), 'wb') as f:
            pickle.dump(all_seqs_features, f)
        # dataframe (for easy post-processing and analysis/visualization)
        seqs_features_df = [copy.copy(seq_features) for seq_features in all_seqs_features]  # not a df yet...
        for seq_index, seq_df in enumerate(seqs_features_df):
            seq_df.insert(0, 'seq_index', seq_index)  # in-place
        seqs_features_df = pd.concat(seqs_features_df, axis=0, ignore_index=True, copy=False)
        with open(self.data_root_path.joinpath(f'raw_{self.timbre_name}_seqs_features.df.pkl'), 'wb') as f:
            pickle.dump(seqs_features_df, f)

    def load_raw_sequences_features_list(self):
        with open(self.data_root_path.joinpath(f'raw_{self.timbre_name}_seqs_features.list.pkl'), 'rb') as f:
            return pickle.load(f)

    def load_raw_sequences_features_df(self):
        with open(self.data_root_path.joinpath(f'raw_{self.timbre_name}_seqs_features.df.pkl'), 'rb') as f:
            return pickle.load(f)


class InterpolationTimbreToolbox(InterpolationTimbreProcessorBase, ToolboxLogger):
    def __init__(self, timbre_toolbox_path: Union[str, pathlib.Path], data_root_path: Union[str, pathlib.Path],
                 verbose=True, remove_matlab_csv_after_usage=True, num_matlab_proc=1):
        """
        Can run the TimbreToolbox (in Matlab), then properly converts the computed audio features into Python
        data structures.
        :param num_matlab_proc: Number of Matlab instances created to process the entire dataset (must be >= 1).
        """
        InterpolationTimbreProcessorBase.__init__(self, data_root_path, 'tt', num_matlab_proc, verbose)
        ToolboxLogger.__init__(self)
        self.timbre_toolbox_path = pathlib.Path(timbre_toolbox_path)
        self._remove_csv_after_usage = remove_matlab_csv_after_usage
        self._logging_lock = threading.RLock()  # Re-entrant lock (with owner when acquired)

    def log_and_print(self, log_str: str, erase_file=False, force_print=False):
        with self._logging_lock:
            open_mode = 'w' if erase_file else 'a'
            with open(self.data_root_path.joinpath('timbre_matlab_log.txt'), open_mode) as f:
                f.write(log_str + '\n')  # Trailing spaces and newlines should have been removed
            if self.verbose or force_print:
                print(log_str)

    def _clean_folders(self):
        """
        Removes all .csv and .mat files (written by TimbreToolbox) from sub-folders of the data_root_path folder.
        """
        for sub_dir in self.audio_sub_folders:
            files_to_remove = list(sub_dir.glob('*.csv'))
            files_to_remove += list(sub_dir.glob('*.mat'))
            for f in files_to_remove:
                f.unlink(missing_ok=False)
        for f in list(self.data_root_path.glob("*_input_args.txt")):
            f.unlink(missing_ok=False)  # list of folders to be analyzed by each Matlab subprocess

    def _get_directories_list_file(self, proc_index: int):
        """ Returns the path to the file that contains the folders that a Matlab instance will analyze. """
        return self.data_root_path.joinpath('matlab{}_input_args.txt'.format(proc_index))

    def run(self):
        t_start = datetime.now()
        self.log_and_print("Deleting all previously written .csv and .mat files...")
        self._clean_folders()

        # split list of audio folders, store them into files
        split_audio_sub_folders = np.array_split(self.audio_sub_folders, self.n_workers)
        for proc_index, proc_audio_folders in enumerate(split_audio_sub_folders):
            with open(self._get_directories_list_file(proc_index), 'w') as f:
                for audio_dir in proc_audio_folders:
                    f.write(str(audio_dir) + '\n')
        # run all processors in their own thread
        threads = [threading.Thread(target=self._run_matlab_proc, args=(i, self._get_directories_list_file(i)))
                   for i in range(self.n_workers)]
        for t in threads:
            t.start()
            time.sleep(1.0)  # delay thread start in order to prevent mixed console outputs
        for t in threads:
            t.join()

        # Read CSVs from all interp sequences and all audio files
        all_sequences_features = list()
        num_interp_steps = None
        for sub_dir in self.audio_sub_folders:
            sequence_descriptors = list()
            csv_files = sorted([f for f in sub_dir.glob('*.csv')])
            if num_interp_steps is None:
                num_interp_steps = len(csv_files)
            elif len(csv_files) != num_interp_steps:
                raise RuntimeError("Inconsistent number of interpolation steps: found {} in {} but {} in the "
                                   "previous folder".format(len(csv_files), sub_dir, num_interp_steps))

            for step_index, csv_file in enumerate(csv_files):
                csv_index = int(csv_file.name.replace('audio_step', '').replace('_stats.csv', ''))
                if step_index != csv_index:
                    raise ValueError("Inconsistent indices. Expected step index: {}; found: {} in {}"
                                     .format(step_index, csv_index, csv_file))
                sequence_descriptors.append(TimbreToolboxResults.read_stats_csv(csv_file))  # might append None (rare)
            # Aggregate per-file descriptors and stats, into a per-sequence dict
            aggregated_seq_data = dict()
            aggregated_seq_data['step_index'] = list(range(len(sequence_descriptors)))
            # Retrieve descriptors' name - very rarely, some (zero-only) audio files can't be processed at all
            all_descriptors_names = None
            for audio_file_data in sequence_descriptors:
                if audio_file_data is not None:
                    all_descriptors_names = audio_file_data.keys()
                    break
            for descriptor_name in all_descriptors_names:
                # use zero-only descriptors values is TT could not compute actual values
                aggregated_seq_data[descriptor_name] = [
                    audio_file_data[descriptor_name] if audio_file_data is not None else 0.0
                    for audio_file_data in sequence_descriptors
                ]
            # convert data for each sequence to DF
            aggregated_seq_data = pd.DataFrame(aggregated_seq_data)
            with open(sub_dir.joinpath('tt_sequence_features.pkl'), 'wb') as f:
                pickle.dump(aggregated_seq_data, f)
            all_sequences_features.append(aggregated_seq_data)
        self.save_raw_sequences_features(all_sequences_features)

        if self._remove_csv_after_usage:
            self._clean_folders()
        delta_t = (datetime.now() - t_start).total_seconds()
        print("[InterpolationTimbreToolbox] Processed {} interpolation sequences in {:.1f} minutes ({:.2f}s/sequence)"
              .format(len(self.audio_sub_folders), delta_t/60.0, delta_t/len(self.audio_sub_folders)))

    def _run_matlab_proc(self, proc_index: int, directories_list_file: pathlib.Path):
        timbre_processor = TimbreToolboxProcess(
            self.timbre_toolbox_path, directories_list_file, verbose=self.verbose, logger=self, process_index=proc_index
        )
        timbre_processor.run()


class InterpolationAudioCommonsTimbre(InterpolationTimbreProcessorBase):
    def __init__(self, data_root_path: Union[str, pathlib.Path], n_workers: int, verbose=True):
        super().__init__(data_root_path, 'ac', n_workers, verbose)

    def _clean_folders(self):
        for audio_dir in self.audio_sub_folders:
            for json_file_path in audio_dir.glob("*.ac.json"):
                json_file_path.unlink(missing_ok=False)

    def run(self):
        if self.verbose:
            print(f"[{self.__class__.__name__}] Computing features...")
        t_start = datetime.now()
        self._clean_folders()

        # Send the list of audio directories to be processed to the pool of workers
        ctx = multiprocessing.get_context("spawn")
        with multiprocessing.pool.Pool(self.n_workers, context=ctx, initializer=_proc_no_warnings) as p:
            p.map(_process_dir__multiproc, self.audio_sub_folders)
        warnings.filterwarnings("default")

        # read all files, build the list of dataframes
        num_interp_steps = len([f for f in self.audio_sub_folders[0].glob("*.wav")])
        all_seqs_features = list()
        for audio_dir in self.audio_sub_folders:
            json_files = sorted([f for f in audio_dir.glob("*.ac.json")])
            seq_features = [{'step_index': step_index} for step_index in range(len(json_files))]
            assert len(seq_features) == num_interp_steps, f"Wrong number of interpolation steps in folder {audio_dir}"
            for step_index, p in enumerate(json_files):
                with open(p, 'r') as f:
                    timbre_features = json.load(f)
                seq_features[step_index].update(timbre_features)
            all_seqs_features.append(pd.DataFrame(seq_features))
        self.save_raw_sequences_features(all_seqs_features)

        self._clean_folders()
        delta_t = (datetime.now() - t_start).total_seconds()
        if self.verbose:
            print(f"[{self.__class__.__name__}] Processed {len(self.audio_sub_folders)} interpolation sequences in "
                  f"{delta_t/60.0:.1f} minutes ({delta_t/len(self.audio_sub_folders):.2f}s/sequence)")


def _proc_no_warnings():
    warnings.filterwarnings("ignore")


def _process_dir__multiproc(dir: pathlib.Path):
    for audio_file_path in dir.glob("*.wav"):
        audio, Fs = soundfile.read(audio_file_path)
        timbre_features = utils.timbral_models.timbral_extractor(audio, Fs, exclude_reverb=True)
        # Save all features into a json file
        with open(audio_file_path.with_suffix('.ac.json'), 'w') as f:
            json.dump(timbre_features, f)


if __name__ == "__main__":

    _timbre_toolbox_path = '~/Documents/MATLAB/timbretoolbox'
    _root_path = pathlib.Path(__file__).resolve().parent.parent
    _data_root_path = _root_path.parent.joinpath('Data_SSD/Logs/RefInterp/LinearNaive/interp9_valid')

    #timbre_proc = InterpolationTimbreToolbox(_timbre_toolbox_path, _data_root_path, num_matlab_proc=8)
    #timbre_proc.run()

    ac_timbre_proc = InterpolationAudioCommonsTimbre(_data_root_path, n_workers=8)
    ac_timbre_proc.run()



