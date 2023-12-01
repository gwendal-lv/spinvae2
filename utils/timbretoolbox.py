import json
import multiprocessing.pool
import os
import queue
import shutil
import subprocess
import pathlib
import tempfile
import threading
import time
import warnings
from datetime import datetime
from typing import Optional
from abc import ABC, abstractmethod

import numpy as np


class ToolboxLogger(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()  # ignore any args or kwargs

    @abstractmethod
    def log_and_print(self, log_str: str, erase_file=False, force_print=False):
        pass


class TimbreToolboxProcess:
    def __init__(self, timbre_toolbox_path: pathlib.Path, directories_list_file: pathlib.Path, verbose=True,
                 logger: Optional[ToolboxLogger] = None, process_index: Optional[int] = None):
        """
        Runs the TimbreToolbox to process folders of audio files (folders' paths given in a separate text file).
        The 'matlab' command must be available system-wide.

        :param timbre_toolbox_path: Path to the TimbreToolbox https://github.com/VincentPerreault0/timbretoolbox.
                                    Timbre Toolbox must have been properly compiled (see instructions in their readme).
        :param directories_list_file: Text file which contains a list of directories to be analyzed by the toolbox.
        :param verbose:
        """
        self.process_index = process_index
        self.logger = logger
        self.verbose = verbose
        self.current_path = pathlib.Path(__file__).parent
        self.matlab_commands = "addpath(genpath('{}')); " \
                               "cd '{}'; " \
                               "timbre('{}'); " \
                               "exit " \
            .format(str(timbre_toolbox_path),
                    str(self.current_path),  # Path to the local timbre.m file
                    str(directories_list_file)
                    )
        self.continue_queue_threads = False

    def _get_process_str(self):
        return '' if self.process_index is None else ' #{}'.format(self.process_index)

    def _enqueue_std_output(self, std_output, q: queue.Queue):
        """
        To be launched as a Thread (contains a blocking readline() call)
        Related question: https://stackoverflow.com/questions/375427/a-non-blocking-read-on-a-subprocess-pipe-in-python

        :param std_output: std::cout or std::cerr
        """
        while self.continue_queue_threads:
            line = std_output.readline()
            if line:
                q.put(line)

    def run(self):

        # Matlab args From https://arc.umich.edu/software/matlab/
        # and https://stackoverflow.com/questions/38723138/matlab-execute-script-from-command-linux-line/38723505
        proc_args = ['matlab', '-nodisplay', '-nodesktop', '-nosplash', '-r', self.matlab_commands]
        log_str = '============ Launching matlab commands (will block if a Matlab error happens) ============\n' \
                  '{}\n' \
                  'Subprocess args: {}\n'.format(datetime.now().strftime("%Y/%m/%d, %H:%M:%S"), proc_args)
        if self.process_index is not None:
            log_str = '[#{}]'.format(self.process_index) + log_str
        self._log_and_print(log_str)

        # Poll process.stdout to show stdout live
        # FIXME this seems quite useless.... subprocess.run seems to do exactly this
        proc = subprocess.Popen(proc_args, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Retrieve std::cout and std::cerr from Threads (to raise an exception if any Matlab error happens)
        std_out_queue, std_err_queue = queue.Queue(), queue.Queue()
        self.continue_queue_threads = True
        std_out_thread = threading.Thread(target=self._enqueue_std_output, args=(proc.stdout, std_out_queue))
        std_err_thread = threading.Thread(target=self._enqueue_std_output, args=(proc.stderr, std_err_queue))
        std_out_thread.start(), std_err_thread.start()

        matlab_error_time = None
        # We keep pooling the queues until the process ends, or an error happens
        keep_running = True
        while keep_running:
            while not std_out_queue.empty():  # Does not guarantee get will not block
                try:
                    line = std_out_queue.get_nowait()
                except queue.Empty:
                    break
                self._log_and_print('[MATLAB{}] {}'.format(self._get_process_str(), line.decode('utf-8').rstrip()))
            while not std_err_queue.empty():  # Does not guarantee get will not block
                try:
                    line = std_err_queue.get_nowait()
                except queue.Empty:
                    break
                self._log_and_print('[MATLAB{} ERROR] {}'.format(self._get_process_str(),
                                                                 line.decode('utf-8').rstrip()), force_print=True)
                if matlab_error_time is None:  # Write this only once
                    matlab_error_time = datetime.now()
            time.sleep(0.001)

            if proc.poll() is not None:  # Natural ending (when script has been fully executed)
                if self.verbose:
                    print("Matlab process{} has ended by itself.".format(self._get_process_str()))
                keep_running = False
            if matlab_error_time is not None:  # Forced ending (after a small delay, to retrieve all std err data)
                if (datetime.now() - matlab_error_time).total_seconds() > 2.0:
                    keep_running = False

        if matlab_error_time is not None:
            raise RuntimeError("Matlab{} has raised an error - please check console outputs above"
                               .format(self._get_process_str()))
        rc = proc.poll()
        if rc != 0:
            warnings.warn('Matlab{} exit code was {}. Please check console outputs.'
                          .format(self._get_process_str(), rc))

        self.continue_queue_threads = False
        std_out_thread.join()
        std_err_thread.join()

        self._log_and_print("\n==================== Matlab subprocess{} has ended ========================\n"
                            .format(self._get_process_str()))

    def _log_and_print(self, log_str: str, force_print=False):
        """ Use the logger attribute if available, otherwise just print """
        if self.logger is not None:
            self.logger.log_and_print(log_str, force_print=force_print)
        elif self.verbose or force_print:
            print(log_str)


class TimbreToolboxResults:
    def __init__(self):
        pass

    @staticmethod
    def _to_float(value: str, csv_file: Optional[pathlib.Path] = None, csv_line: Optional[int] = None):
        try:  # detection of casting errors
            float_value = float(value)
        except ValueError as e:
            if csv_line is not None and csv_line is not None:
                warnings.warn("Cannot convert {} to float, will use complex module instead. File '{}' "
                              "line {}: {}".format(value, csv_file, csv_line, e))
            else:
                warnings.warn("Cannot convert {} to float, will use complex module instead. {}".format(value, e))
            float_value = np.abs(complex(value.replace('i', 'j')))  # Matlab uses 'i', Python uses 'j'
        return float_value

    @staticmethod
    def read_stats_csv(csv_file: pathlib.Path):
        """
        :return: A Dict of descriptors, or None if the Evaluation could not be performed by TimbreToolbox
        """
        descr_data = dict()
        with open(csv_file, 'r') as f:
            lines = [line.rstrip() for line in f.readlines()]
            # if lines is a single-item array, the CSV is supposed to correspond to an "Evaluation Error" case
            if len(lines) == 1 or (len(lines) == 2 and len(lines[1]) <= 2):
                if lines[0] == 'Evaluation Error':
                    warnings.warn("File {} contains 'Evaluation Error' - the Matlab script could not "
                                  "evaluate the associated audio file".format(csv_file))
                    return None
                else:
                    raise ValueError("{} contains only 1 line: '{}' (should be audio features, or 'Evaluation Error')"
                                     .format(csv_file, lines[0]))
            # Otherwise, usual case: CSVs written by TimbreToolbox are not tabular data, but 1-item-per-line CSVs
            current_repr, current_descr = None, None
            for i, line in enumerate(lines):
                if len(line) > 0:  # Non-empty line
                    try:
                        name, value = line.split(',')
                    except ValueError:
                        continue  # e.g. empty Unit, or STFT has oddly formatted data (single line with 'Minimums,')
                    # First: we look for a new descriptor or representation
                    if name.lower() == 'representation':  # New representation: next fields are related to the
                        current_repr = value              # representation, not one of its descriptors
                        current_descr = None
                    elif name.lower() == 'descriptor':  # Next fields will be related to this descriptor
                        current_descr = value
                    else:  # At this point, the line contains a specific field
                        if current_descr is not None:  # Representation fields (min/max/med/iqr + other fields, params?)
                            if name.lower() == 'value':  # Global descriptors: single value
                                descr_data[current_descr] = TimbreToolboxResults._to_float(value, csv_file, i)
                            elif name.lower() == 'minimum':
                                descr_data[current_descr + '_min'] = TimbreToolboxResults._to_float(value, csv_file, i)
                            elif name.lower() == 'maximum':
                                descr_data[current_descr + '_max'] = TimbreToolboxResults._to_float(value, csv_file, i)
                            elif name.lower() == 'median':
                                descr_data[current_descr + '_med'] = TimbreToolboxResults._to_float(value, csv_file, i)
                            elif name.lower() == 'interquartile range':
                                descr_data[current_descr + '_IQR'] = TimbreToolboxResults._to_float(value, csv_file, i)
                            else:  # Other descriptor fields (e.g. unit, parameters) are discarded
                                pass
                else:  # Empty line indicates a new descriptor (maybe a new representation)
                    current_descr = None
        return descr_data


class TimbreToolboxSingleDir:
    def __init__(self, audio_dir: pathlib.Path,
                 n_workers=1, max_audio_files_per_worker=2000,
                 timbre_toolbox_path='~/Documents/MATLAB/timbretoolbox', verbose=False, json_files_suffix=''):
        """ Allows to easily perform a Timbre Toolbox analysis for all .wav audio files than can be found
        in a given directory. Uses multiple Matlab instances in parallel (a single instance can use only 1 CPU).
        Works for huge directories, which will be split into small directories with symlink to the original.

        JSON files (instead of raw .CSV from Matlab) will be written directly in the given audio_dir. """

        self.audio_dir, self.n_workers, self.timbre_toolbox_path, self.verbose, self.json_files_suffix = \
            audio_dir, n_workers, pathlib.Path(timbre_toolbox_path), verbose, json_files_suffix

        # look for all files in audio_dir, separate them into new directories using symlinks
        #    because 1 Matlab instance will process all files (or symlinks) available in a given directory
        self.wav_files = sorted([p for p in self.audio_dir.glob('*.wav') if p.is_file()])
        # assign wav files to audio dirs - split using the appropriate chunk size (not the number of chunks);
        # we'll usually have more chunks (of data) than workers
        n_chunks = 1 + (len(self.wav_files) // max_audio_files_per_worker)
        if n_chunks < n_workers:  # Small dataset (few chunks) -> use a larger number of chunks with less datapoints
            n_chunks = n_workers
        self.original_wav_files_split = np.array_split(np.asarray(self.wav_files), n_chunks)
        self.original_wav_files_split = [list(a) for a in self.original_wav_files_split]
        # Fill the temp audio dirs (inside an easy-to-remove temp directory)
        self.main_temp_audio_dir = self.audio_dir.parent.joinpath(self.audio_dir.name + "_temp")
        if os.path.exists(self.main_temp_audio_dir):
            shutil.rmtree(self.main_temp_audio_dir)
        self.main_temp_audio_dir.mkdir(exist_ok=False)
        self.temp_split_audio_dirs = [self.main_temp_audio_dir.joinpath(f'{i:03d}') for i in range(self.n_splits())]
        # create temporary directories and symlinks
        # more temp dirs than workers: each directory should contain 2000 sounds maximum... to prevent the error:
        #     Exception in thread "AWT-EventQueue-0" java.lang.OutOfMemoryError: Java heap space
        #     (TimbreToolbox bug: seems to happen after approx. 13k audio files have been processed)
        self.wav_symlinks_split = [[] for _ in self.original_wav_files_split]
        for i, temp_audio_dir in enumerate(self.temp_split_audio_dirs):
            temp_audio_dir.mkdir(exist_ok=False)
            for original_wav_file in self.original_wav_files_split[i]:
                self.wav_symlinks_split[i].append(temp_audio_dir.joinpath(original_wav_file.name))
                os.symlink(original_wav_file, self.wav_symlinks_split[i][-1])

        # Will be auto-deleted (if not killed...)
        self.temp_matlab_arg_file_wrappers = [tempfile.NamedTemporaryFile('w+') for _ in range(self.n_splits())]
        for i, temp_audio_dir in enumerate(self.temp_split_audio_dirs):
            self.temp_matlab_arg_file_wrappers[i].file.write(str(temp_audio_dir) + "\n")
            self.temp_matlab_arg_file_wrappers[i].file.flush()
        if self.verbose:
            print(f"[{self.__class__.__name__}] Matlab input args stored in "
                  f"{[p.name for p in self.temp_matlab_arg_file_wrappers]}")

    def n_splits(self):
        return len(self.original_wav_files_split)

    def run(self):
        # run all processors in their own thread
        threads_args = [(i, pathlib.Path(temp_arg_file_wrapper.name))
                        for i, temp_arg_file_wrapper in enumerate(self.temp_matlab_arg_file_wrappers)]
        with multiprocessing.pool.ThreadPool(self.n_workers) as p:
            p.map(self._run_single_dir_processor, threads_args)

        # read results (CSVs in all temp sub-dirs) and store them to json
        csv_files_path = []
        for i in range(len(self.wav_symlinks_split)):
            for wav_symlink in self.wav_symlinks_split[i]:
                csv_files_path.append(wav_symlink.parent.joinpath(wav_symlink.name.replace('.wav', '_stats.csv')))
        all_raw_descriptors_values = [TimbreToolboxResults.read_stats_csv(p) for p in csv_files_path]  # list of dicts
        json_files_path = [self.audio_dir.joinpath(p.name.replace('_stats.csv', f'{self.json_files_suffix}.json'))
                           for p in csv_files_path]
        for i, p in enumerate(json_files_path):
            with open(p, 'w') as f:
                json.dump(all_raw_descriptors_values[i], f)
        # erase temp folders (will erase all symlinks and CSVs)
        shutil.rmtree(self.main_temp_audio_dir)

    def _run_single_dir_processor(self, args):
        proc_index, temp_arg_file = args
        timbre_processor = TimbreToolboxProcess(
            self.timbre_toolbox_path, temp_arg_file, self.verbose, process_index=proc_index)
        # This launches Matlab and the script, but nothing else (no cleanup, no post-processing, etc.)
        timbre_processor.run()


