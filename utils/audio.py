"""
Audio utils, mostly based on librosa functionalities

Do not import torch_spectrograms to prevent multi-processing issues
"""
import copy
import multiprocessing
import os
import typing
from datetime import datetime
from typing import Iterable, Sequence, Optional
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import librosa
import librosa.display
import soundfile as sf

from data.abstractbasedataset import AudioDataset


class SequencesMFCC:
    def __init__(self, storage_path: typing.Union[str, pathlib.Path],
                 n_mfcc: int, n_fft=2048, hop_length=1024, fmin=0.0, start_end_only=False, verbose=False):
        """
        Utility class to compute MFCCs from .wav files stored in subdirectories of storage_path.

        :param start_end_only: If True, only the "start" and "end" files (of the sorted(...) list of files for
            each subdirectory) will be considered.
        """
        self.storage_path = pathlib.Path(storage_path)
        self.start_end_only, self.verbose = start_end_only, verbose
        self.n_mfcc, self.n_fft, self.hop_length, self.fmin = n_mfcc, n_fft, hop_length, fmin
        self.mfccs_df: Optional[pd.DataFrame] = None

    @property
    def mfcc_description(self):
        return f"mfcc{self.n_mfcc}_nfft{self.n_fft}_hop{self.hop_length}_fmin{self.fmin:.0f}"

    def compute(self):
        sr = -1
        mfccs = list()
        subdirs = [f for f in self.storage_path.glob('*') if f.is_dir()]
        progress_steps = np.round(len(subdirs) * np.linspace(0.0, 1.0, 10, endpoint=False)).astype(int)
        for i, f in enumerate(subdirs):
            wav_files = sorted([str(wav_f) for wav_f in f.glob('*.wav')])
            if self.start_end_only:
                if len(wav_files) < 2:
                    raise FileNotFoundError(f"Directory {f} should contain at least 2 .wav files")
                wav_files = [wav_files[0], wav_files[-1]]
            wav_files = [pathlib.Path(wav_f) for wav_f in wav_files]
            for wav_f in wav_files:
                x, _sr = sf.read(wav_f)
                if sr < 0:  # not set yet
                    sr = _sr
                else:
                    assert sr == _sr, "Sampling rates must be the same for all samples (in all subdirectories)"
                mfcc = librosa.feature.mfcc(
                    x, sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length, fmin=self.fmin,
                )
                mfccs.append({'dir': f.name, 'file': wav_f.name, f'mfcc{self.n_mfcc}': mfcc})
            if self.verbose and i in progress_steps:
                print(f"[{self.__class__.__name__}] "
                      f"Progress: {100.0 * (i+1)/len(subdirs):.0f}% (subdirectory {i+1}/{len(subdirs)})")
        self.mfccs_df = pd.DataFrame(mfccs).sort_values(by=['dir', 'file'], ignore_index=True)
        save_path = self.storage_path.joinpath(f"{self.mfcc_description}.df.pkl")
        self.mfccs_df.to_pickle(save_path)
        if self.verbose:
            print(f"[{self.__class__.__name__}] Progress: 100% (done). Saved to: {save_path}")

    def load(self):
        self.mfccs_df = pd.read_pickle(self.storage_path.joinpath(f"{self.mfcc_description}.df.pkl"))
        return self.mfccs_df

    def mean_euclidean_distance(self, other_storage_path: typing.Union[str, pathlib.Path]):
        other_storage_path = pathlib.Path(other_storage_path)
        other_mfccs_df = pd.read_pickle(other_storage_path.joinpath(f"{self.mfcc_description}.df.pkl"))
        assert np.all(self.mfccs_df['dir'] == other_mfccs_df['dir'])
        assert np.all(self.mfccs_df['file'] == other_mfccs_df['file'])
        distances_df = copy.deepcopy(self.mfccs_df)
        distances_df[f'mfcc{self.n_mfcc}_other'] = copy.deepcopy(other_mfccs_df[f'mfcc{self.n_mfcc}'])
        distances_df[f'mfcc{self.n_mfcc}_distance'] = distances_df.apply(
            lambda x: np.mean(np.sqrt(np.square(x[f'mfcc{self.n_mfcc}'] - x[f'mfcc{self.n_mfcc}_other']))),
            # lambda x: np.mean(x[f'mfcc{self.n_mfcc}']),
            axis=1
        )
        return distances_df


class SimilarityEvaluator:
    """ Class for evaluating audio similarity between audio samples through various criteria. """
    def __init__(self, x_wav: Sequence[Iterable], n_fft=1024, fft_hop=256, sr=22050, n_mfcc=13):
        """

        :param x_wav: List or Tuple which contains the 2 audio signals (arrays) to be compared.
        :param n_fft:
        :param fft_hop:
        :param sr:
        :param n_mfcc:
        """
        assert len(x_wav) == 2  # This class requires exactly 2 input audio signals
        self.x_wav = x_wav
        self.n_fft = n_fft
        self.fft_hop = fft_hop
        self.sr = sr
        self.n_mfcc = n_mfcc
        # Pre-compute STFT (used in mae log and spectral convergence)
        self.stft = [np.abs(librosa.stft(x, self.n_fft, self.fft_hop)) for x in self.x_wav]

    def get_mae_log_stft(self, return_spectrograms=True):
        """ Returns the Mean Absolute Error on log(|STFT|) spectrograms of input sounds, and the two spectrograms
        themselves (e.g. for plotting them later). """
        eps = 1e-4  # -80dB  (un-normalized stfts)
        log_stft = [np.maximum(s, eps) for s in self.stft]
        log_stft = [np.log10(s) for s in log_stft]
        mae = np.abs(log_stft[1] - log_stft[0]).mean()
        return (mae, log_stft) if return_spectrograms else mae

    def display_stft(self, s, log_scale=True):
        """ Displays given spectrograms s (List of two |STFT|) """
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        im = librosa.display.specshow(s[0], shading='flat', ax=axes[0], cmap='magma')
        im = librosa.display.specshow(s[1], shading='flat', ax=axes[1], cmap='magma')
        if log_scale:
            axes[0].set(title='Reference $\log_{10} |STFT|$')
        else:
            axes[0].set(title='Reference $|STFT|$')
        axes[1].set(title='Inferred synth parameters')
        fig.tight_layout()
        return fig, axes

    def get_spectral_convergence(self, return_spectrograms=True):
        """ Returns the Spectral Convergence of input sounds, and the two linear-scale spectrograms
            used to compute SC (e.g. for plotting them later). SC: see https://arxiv.org/abs/1808.06719 """
        # Frobenius norm is actually the default numpy matrix norm
        # TODO check for 0.0 frob norm of stft[0]
        sc = np.linalg.norm(self.stft[0] - self.stft[1], ord='fro') / np.linalg.norm(self.stft[0], ord='fro')
        return (sc, self.stft) if return_spectrograms else sc

    def get_mae_mfcc(self, return_mfccs=True, n_mfcc: Optional[int] = None):
        """ Returns the Mean Absolute Error on MFCCs, and the MFCCs themselves.
        Uses librosa default MFCCs configuration: TODO precise
        """
        mfcc = [librosa.feature.mfcc(x, sr=self.sr, n_mfcc=(self.n_mfcc if n_mfcc is None else n_mfcc))
                for x in self.x_wav]
        mae = np.abs(mfcc[0] - mfcc[1]).mean()
        return (mae, mfcc) if return_mfccs else mae

    def display_mfcc(self, mfcc):
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        im = librosa.display.specshow(mfcc[0], shading='flat', ax=axes[0], cmap='viridis')
        im = librosa.display.specshow(mfcc[1], shading='flat', ax=axes[1], cmap='viridis')
        axes[0].set(title='{}-bands MFCCs'.format(self.n_mfcc))
        axes[1].set(title='Inferred synth parameters')
        fig.tight_layout()
        return fig, axes



class SimpleSampleLabeler:
    def __init__(self, x_wav, Fs, hpss_margin=3.0, perc_duration_ms=250.0):
        """ Class to attribute labels or a class to sounds, mostly based on librosa hpss and empirical thresholds.

        :param x_wav:
        :param Fs:
        :param hpss_margin: see margin arg of librosa.decompose.hpss
        :param perc_duration_ms: The duration of a percussion sound - most of the percussive energy should be found
            before that time (in the percussive separated spectrogram).
        """
        assert Fs == 22050  # Librosa defaults must be used at the moment
        self.x_wav = x_wav
        self.Fs = Fs  # FIXME 22 or 16 kHz
        self.hpss_margin = hpss_margin
        self.perc_duration_ms = perc_duration_ms
        # Pre-computation of spectrograms and energies
        self.specs = self._get_hpr_specs()
        self.energy, self.energy_ratio = self._get_energy_ratios()
        # Energies on attack (to identify perc sounds)
        # Perc content supposed to be found in the first 10s of ms. Hop: default librosa 256
        limit_index = int(np.ceil(self.perc_duration_ms * self.Fs / 256.0 / 1000.0))  # FIXME 22 or 16 kHz
        self.attack_specs = dict()
        self.attack_energies = dict()
        for k in self.specs:
            self.attack_specs[k] = self.specs[k][:, 0:limit_index]  # indexes: f, t
            self.attack_energies[k] = np.abs(self.attack_specs[k]).sum()
        # Labels pre-computation... so it's done
        self.is_harmonic = self._is_harmonic()
        self.is_percussive = self._is_percussive()

    def has_label(self, label):
        if label == 'harmonic':
            return self.is_harmonic
        elif label == 'percussive':
            return self.is_percussive
        elif label == 'sfx':
            return not self.is_harmonic and not self.is_percussive
        else:
            raise ValueError("Label '{}' is not valid.".format(label))

    def _get_hpr_specs(self):
        D = librosa.stft(self.x_wav)  # TODO custom fft params
        H, P = librosa.decompose.hpss(D, margin=self.hpss_margin)
        R = D - (H + P)
        return {'D': D, 'H': H, 'P': P, 'R': R}

    def _get_energy_ratios(self):
        energy = dict()
        for k in self.specs:
            energy[k] = np.abs(self.specs[k]).sum()
        return energy, {'D': 1.0, 'H': energy['H'] / energy['D'], 'P': energy['P'] / energy['D'],
                        'R': energy['R'] / energy['D']}

    def plot_hpr_specs(self, figsize=(8, 6)):
        fig, axes = plt.subplots(4, 1, figsize=figsize)
        axes = [axes]  # Unqueeze - to prepare for multi-cols display
        for col in range(1):
            im = librosa.display.specshow(librosa.amplitude_to_db(np.abs(self.specs['D']), ref=np.max), y_axis='log',
                                          ax=axes[col][0])
            fig.colorbar(im, format='%+2.0f dB', ax=axes[col][0])
            axes[col][0].set(title='Full power spectrogram')
            im = librosa.display.specshow(librosa.amplitude_to_db(np.abs(self.specs['H']), ref=np.max), y_axis='log',
                                          ax=axes[col][1])
            fig.colorbar(im, format='%+2.0f dB', ax=axes[col][1])
            axes[col][1].set(title='Harmonic power spectrogram ({:.1f}% of total spectral power)'.format(
                100.0 * self.energy_ratio['H']))
            im = librosa.display.specshow(librosa.amplitude_to_db(np.abs(self.specs['P']), ref=np.max), y_axis='log',
                                          ax=axes[col][2])
            fig.colorbar(im, format='%+2.0f dB', ax=axes[col][2])
            axes[col][2].set(title='Percussive power spectrogram ({:.1f}% of total spectral power)'.format(
                100.0 * self.energy_ratio['P']))
            im = librosa.display.specshow(librosa.amplitude_to_db(np.abs(self.specs['R']), ref=np.max), y_axis='log',
                                          ax=axes[col][3])
            fig.colorbar(im, format='%+2.0f dB', ax=axes[col][3])
            axes[col][3].set(title='Residuals power spectrogram ({:.1f}% of total spectral power)'.format(
                100.0 * self.energy_ratio['R']))
        fig.tight_layout()
        return fig, axes

    def get_harmonic_sound(self):
        return librosa.istft(self.specs['H'])

    def get_percussive_sound(self):
        return librosa.istft(self.specs['P'])

    def get_residual_sound(self):
        return librosa.istft(self.specs['R'])

    def _is_harmonic(self):
        if self.energy_ratio['H'] > 0.40:
            return True
        elif self.energy_ratio['H'] > 0.35:  # Harmonic with percussive attack
            return (self.attack_energies['P'] / self.energy['P']) > 0.9
        return False

    def _is_percussive(self):
        # Mostly percussive sound
        if self.energy_ratio['P'] > 0.40:
            return (self.attack_energies['P'] / self.energy['P']) > 0.9
        # Percussive with harmonic attack
        elif self.energy_ratio['P'] > 0.35 and self.energy_ratio['H'] > 0.15:
            return (self.attack_energies['P'] / self.energy['P']) > 0.9\
                   and (self.attack_energies['H'] / self.energy['H']) > 0.8
        return False

    def print_labels(self):
        print("is_harmonic={}   is_percussive={}".format(self.is_harmonic, self.is_percussive))



def average_f0(dataset: AudioDataset, midi_pitch, midi_vel, num_workers: Optional[int] = None, print_time=True):
    """ Computes the f0 pitch of the given note, for each preset of the dataset.
    -1 pitches correspond to no pitch detected by librosa.pyin.

    Uses multiprocessing even if num_workers == 1. """
    t_start = datetime.now()
    presets_indexes = np.arange(len(dataset))
    if num_workers is None:
        num_workers = os.cpu_count()
    split_presets_indexes = np.array_split(presets_indexes, num_workers)
    workers_args = list()
    for indexes in split_presets_indexes:
        workers_args.append((dataset, midi_pitch, midi_vel, indexes))
    with multiprocessing.Pool(num_workers) as p:
        f0_split = p.map(_average_f0, workers_args)
    f0_merged = list()
    for f0 in f0_split:
        f0_merged += f0
    delta_t = (datetime.now() - t_start).total_seconds()
    if print_time:
        print("{} note f0 pitch estimatates computed in {:.1f} min ({:.1f} ms / note)"
              .format(len(presets_indexes), delta_t/60.0, 1000.0*delta_t/len(presets_indexes)))
    return f0_merged


def _average_f0(worker_args):
    dataset, midi_pitch, midi_vel, presets_indexes = worker_args
    f0_list = list()
    for preset_index in presets_indexes:
        preset_UID = dataset.valid_preset_UIDs[preset_index]
        audio, Fs = dataset.get_wav_file(preset_UID, midi_pitch, midi_vel, variation=0)
        f0, voiced_flag, voiced_prob = librosa.pyin(audio, fmin=librosa.midi_to_hz(midi_pitch - 24),
                                                    fmax=librosa.midi_to_hz(midi_pitch + 24), sr=dataset.Fs)
        if f0[voiced_flag].shape[0] > 0:
            f0_list.append(f0[voiced_flag].mean())
        else:
            f0_list.append(-1.0)
    return f0_list



def write_wav_and_mp3(
        base_path: pathlib.Path, base_name: str, samples, sr,
        ffmpeg_command="ffmpeg", ffmpeg_options="-hide_banner -loglevel error"):
    """ Writes a .wav file and converts it to .mp3 using command-line ffmpeg (which must be available). """
    wav_path_str = "{}".format(base_path.joinpath(base_name + '.wav'))
    sf.write(wav_path_str, samples, sr)
    mp3_path_str = "{}".format(base_path.joinpath(base_name + '.mp3'))
    # mp3 320k will be limited to 160k for mono audio - still too much loss for HF content
    os.system(f"{ffmpeg_command} {ffmpeg_options} -i '{wav_path_str}' -b:a 320k -y '{mp3_path_str}'")
    return pathlib.Path(wav_path_str), pathlib.Path(mp3_path_str)



def dataset_samples_rms(dataset: AudioDataset, outliers_min: Optional[int] = None,
                        num_workers=os.cpu_count(), print_analysis_duration=True):
    """ Computes a list of RMS stats (min/avg/max) for each audio file available from the given dataset.
    Also returns outliers (each outlier as a (UID, pitch, vel, var) tuple) if min/max values are given, else None.

    :returns: (rms_stats list of dicts, outliers list of tuples) """
    if print_analysis_duration:
        print("Starting dataset RMS computation...")
    t_start = datetime.now()
    if num_workers < 1:
        num_workers = 1
    if num_workers == 1:
        audio_rms_stats, outliers = _dataset_samples_rms((dataset, dataset.valid_preset_UIDs, outliers_min))
    else:
        split_preset_UIDs = np.array_split(dataset.valid_preset_UIDs, num_workers)
        workers_args = [(dataset, UIDs, outliers_min) for UIDs in split_preset_UIDs]
        with multiprocessing.Pool(num_workers) as p:  # automatically closes and joins all workers
            results = p.map(_dataset_samples_rms, workers_args)
            # Structure of results:
            #     list of length num_workers
            #         dict of 3 lists (1 list for 'min', etc....)
            #         list of outliers (1 tuple / outlier)
        audio_rms_stats = dict()
        for k in results[0][0]:  # keys detected automatically (defined in _dataset_samples_rms)
            audio_rms_stats[k] = list()
        outliers = list()
        for r in results:
            for k in audio_rms_stats:
                audio_rms_stats[k] += r[0][k]
            outliers += r[1]  # merges the 2 lists
    delta_t = (datetime.now() - t_start).total_seconds()
    if print_analysis_duration:
        print("Dataset RMS computation finished. {:.1f} min total ({:.1f} ms / wav, {} audio files)"
              .format(delta_t / 60.0, 1000.0 * delta_t / dataset.nb_valid_audio_files, dataset.nb_valid_audio_files))
    return audio_rms_stats, outliers


def _dataset_samples_rms(worker_args):
    dataset, preset_UIDs, outliers_min = worker_args
    """ Auxiliary function for dataset_samples_rms: computes a single batch (multiproc or not). """
    audio_rms_stats = {'min': list(), 'avg': list(), 'max': list()}
    outliers = list()
    for preset_UID in preset_UIDs:
        for midi_note in dataset.midi_notes:
            midi_pitch, midi_vel = midi_note
            for variation in range(dataset.get_nb_variations_per_note(preset_UID)):
                audio, Fs = dataset.get_wav_file(preset_UID, midi_pitch, midi_vel, variation=variation)
                rms_frames = librosa.feature.rms(audio)  # type: np.ndarray
                rms_frames = rms_frames[0]
                audio_rms_stats['min'].append(rms_frames.min())
                audio_rms_stats['avg'].append(rms_frames.mean())
                audio_rms_stats['max'].append(rms_frames.max())
                if outliers_min is not None:
                    if audio_rms_stats['max'][-1] < outliers_min:  # If max RMS is very small: outlier
                        outliers.append((preset_UID, midi_pitch, midi_vel, variation))
    return audio_rms_stats, outliers


def trim(_audio, _sr, _db_th=30, _min_len_s=1.5, _fade_ms=20.0):
    assert _min_len_s > _fade_ms / 1000.0
    _, _non_silent_indices = librosa.effects.trim(_audio, top_db=_db_th)
    # keep a minimal length
    _new_n_samples = max(_non_silent_indices[1], int(_min_len_s * _sr))
    # and don't trim the beginning
    _audio_trimmed = copy.deepcopy(_audio[0:_new_n_samples])
    # apply fade at the end (otherwise a click almost always occurs)
    _n_fade = int(_sr * _fade_ms / 1000.0)
    _fade_ramp = np.linspace(1.0, 0.0, num=_n_fade)
    _audio_trimmed[-_n_fade:] = _audio_trimmed[-_n_fade:] * _fade_ramp
    return _audio_trimmed

