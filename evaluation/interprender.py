"""
Base class for the InterpBase class itself, which focuses on rendering audio files using multiple processes.

This file should remain as simple as possible to prevent any issue with multiprocessing features.
"""
import sys
from abc import ABC, abstractmethod
import multiprocessing
from typing import List, Optional

import numpy as np


class InterpMultiprocRenderer(ABC):  # ABC will be used by the child class
    def __init__(self, num_steps: int, dataset, generated_note_duration):
        super().__init__()
        self.num_steps, self.dataset, self.generated_note_duration = num_steps, dataset, generated_note_duration

        # Slave (child) processes. This one is considered as the master.
        self._slave_procs = None
        # Master-endpoint connections to send args and to receive results
        self._args_connections: Optional[List[multiprocessing.connection.Connection]] = None
        self._results_connections: Optional[List[multiprocessing.connection.Connection]] = None

        self._seq_number = 0  # TCP-like sequence number, to ensure worker's tasks ordering (maybe be unnecessary...)
        self._procs_current_seq_number: Optional[List[int]] = None

        self._debug = sys.gettrace() is not None  # works for pycharm - we won't use multiprocessing when debugging

    @property
    def _n_workers(self):
        return self.num_steps  # On small/medium systems, may be greater (e.g. 7 or 9) than the number of CPU cores

    def on_render_audio_begins(self):
        if self._debug:  # just return if not using procs
            return

        self._slave_procs, self._args_connections, self._results_connections = list(), list(), list()
        self._procs_current_seq_number = list()

        # Create all processes
        context = multiprocessing.get_context('spawn')
        for i in range(self._n_workers):
            # Pipe to send args (for a slave to know what to compute)
            #     should be used in the master->slave direction only
            master_args_connection, slave_args_connection = context.Pipe()  # 1 pipe is made of 2 endpoint connections
            self._args_connections.append(master_args_connection)
            # Pipe to send results (from a slave)
            #     should be used in the master->slave direction only
            master_results_connection, slave_results_connection = context.Pipe()
            self._results_connections.append(master_results_connection)
            # We send the data set and note duration now (not to have to pickle them for each new audio render)
            self._slave_procs.append(context.Process(
                target=audio_render_proc,
                args=(i, slave_args_connection, slave_results_connection, self.dataset, self.generated_note_duration)
            ))
            self._slave_procs[-1].start()  # Each process will wait for a message in the Pipe
            self._procs_current_seq_number.append(-1)

        # wait for all processes to be properly started - we don't use another synchronization object
        #  (e.g. semaphore) because the Pipes are already setup
        # Send commands to the workers
        for i in range(self._n_workers):
            self._procs_current_seq_number[i] = self._seq_number
            self._args_connections[i].send((self._seq_number, "START_SYNC", ))
            self._seq_number += 1
        # Then retrieve all results - using the sequence number to ensure that the result is exactly what we expect
        for i in range(self._n_workers):
            slave_results = self._results_connections[i].recv()
            result_seq_number = slave_results[0]
            assert result_seq_number == self._procs_current_seq_number[i], f"Worker {i}: sequence ordering mismatch"
            assert slave_results[1] == f"WORKER_{i}_START_SYNC_OK"

    def on_render_audio_ends(self, _: int):
        if self._debug:  # don't use mp when debugging
            return

        # Nicely close all workers and pipes
        for i in range(self._n_workers):
            self._args_connections[i].send("STOP")
            self._slave_procs[i].join(timeout=60.0)  # timeout does NOT raise an exception is it has expired
            assert self._slave_procs[i].exitcode == 0, f"Audio process #{i} could not be joined after the configured " \
                                                       f"timeout. It might be stuck somewhere."
            self._slave_procs[i].close()
            # here we close our (master) pipes' endpoints; workers will close their own ends of the pipes
            self._args_connections[i].close()
            self._results_connections[i].close()

        self._slave_procs, self._args_connections, self._results_connections = None, None, None
        self._procs_current_seq_number = None

    def generate_audio__multiproc(self, vst_presets: np.ndarray, disable_multiprocessing=False):
        """ Renders audio files using 1 process for each interpolation step (workers were already created
            and are waiting for some data). """
        if self._debug or disable_multiprocessing:  # if we are debugging: don't use processes....
            midi_pitch, midi_vel = self.dataset.default_midi_note
            audio_renders = [
                self.dataset._render_audio(vst_presets[i, :], midi_pitch, midi_vel, self.generated_note_duration)
                for i in range(vst_presets.shape[0])
            ]

        else:  # Actual multi-processing
            assert self._n_workers == vst_presets.shape[0], \
                f'Multi-processing requires n_workers ({self._n_workers}) == num_step ({vst_presets.shape})'
            # Send commands to the workers  TODO add sequence id
            for i in range(self._n_workers):
                self._procs_current_seq_number[i] = self._seq_number
                self._args_connections[i].send((self._seq_number, vst_presets[i, :], ))
                self._seq_number += 1
            # Then retrieve all results - using the sequence number to ensure that the result is exactly what we expect
            audio_renders = list()
            for i in range(self._n_workers):
                slave_results = self._results_connections[i].recv()
                result_seq_number = slave_results[0]
                assert result_seq_number == self._procs_current_seq_number[i], f"Worker {i}: sequence ordering mismatch"
                audio_renders.append(slave_results[1])
        return audio_renders



def audio_render_proc(
        idx: int,
        args_connection: multiprocessing.connection.Connection,
        results_connection: multiprocessing.connection.Connection,
        dataset, note_duration
):
    midi_pitch, midi_vel = dataset.default_midi_note

    # Wait for the start 'signal' from the master
    args = args_connection.recv()  # args[0] is a sequence number
    assert args[1] == "START_SYNC", "The first message in the pipe from master should be 'START_SYNC'"
    results_connection.send((args[0], f"WORKER_{idx}_START_SYNC_OK"))

    continue_worker = True
    while continue_worker:
        args = args_connection.recv()
        if isinstance(args, str) and args == "STOP":  # string args (not a tuple, no sequence number)
            continue_worker = False

        elif isinstance(args, tuple):
            seq_number, vst_preset = args[0], args[1]
            assert isinstance(seq_number, int) and isinstance(vst_preset, np.ndarray) and len(vst_preset.shape) == 1
            audio_render = dataset._render_audio(vst_preset, midi_pitch, midi_vel, note_duration)
            results_connection.send((seq_number, audio_render, ))  # TODO send seq ID in response

        else:
            raise ValueError(f"Received unexpected args from the Pipe from the master process: {args}")

    args_connection.close()  # Master won't send anything anymore
    results_connection.close()  # This slave does not need to send anything either: return 0 will be checked
    return 0

