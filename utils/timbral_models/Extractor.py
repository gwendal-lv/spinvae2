from __future__ import division

import copy
import multiprocessing
import warnings
from typing import Sequence

import librosa.util.exceptions
import soundfile as sf
import numpy as np
import six
import time
from . import timbral_util, timbral_hardness, timbral_depth, timbral_brightness, timbral_roughness, timbral_warmth, \
    timbral_sharpness, timbral_booming, timbral_reverb


def timbral_extractor(fname, fs=0, dev_output=False, phase_correction=False, clip_output=False,
                      exclude_reverb=False, output_type='dictionary'):
    """
      The Timbral Extractor will extract all timbral attribute sin one function call, returning the results as either
      a list or dictionary, depending on input definitions.

      Version 0.4

      Simply calls each function with

      Required parameter
      :param fname:             string or numpy array
                                string, audio filename to be analysed, including full file path and extension.
                                numpy array, array of audio samples, requires fs to be set to the sample rate.

     Optional parameters
      :param fs:                int/float, when fname is a numpy array, this is a required to be the sample rate.
                                Defaults to 0.
      :param phase_correction:  bool, perform phase checking before summing to mono.  Defaults to False.
      :param dev_output:        bool, when False return the depth, when True return all extracted
                                features.  Default to False.
      :param clip_output:             bool, force the output to be between 0 and 100.
      :param output_type:       string, defines the type the output should be formatted in.  Accepts either
                                'dictionary' or 'list' as parameters.  Default to 'dictionary'.

      :return: timbre           the results from all timbral attributes as either a dictionary or list, depending
                                on output_type.

      Copyright 2019 Andy Pearce, Institute of Sound Recording, University of Surrey, UK.
    """
    '''
      Check output_type before calculating anything
    '''
    if output_type != 'dictionary' and output_type != 'list':
        raise ValueError('output_type must be \'dictionary\' or \'list\'.')

    timbre = {
        'hardness': None,
        'depth': None,
        'brightness': None,
        'roughness': None,
        'warmth': None,
        'sharpness': None,
        'boominess': None
    }
    if not exclude_reverb:
        timbre['reverb'] = None

    '''
      Basic audio reading
    '''
    if isinstance(fname, six.string_types):
        # read audio file only once and pass arrays to algorithms
        try:
            audio_samples, fs = sf.read(fname)
            # making an array again for copying purposes
            multi_channel_audio = np.array(audio_samples)
        except:
            print('Soundfile failed to load: ' + str(fname))
            raise TypeError('Unable to read audio file.')
    elif hasattr(fname, 'shape'):
        if fs==0:
            raise ValueError('If giving function an array, \'fs\' must be specified')
        audio_samples = fname
        multi_channel_audio = np.array(fname)
    else:
        raise ValueError('Input must be either a string or a numpy array.')

    # channel reduction
    audio_samples = timbral_util.channel_reduction(audio_samples)

    # resample audio file if sample rate is less than 44100
    audio_samples, fs = timbral_util.check_upsampling(audio_samples, fs)
    # In the original code (2019), all already-loaded files are 'read' a second time, but with some normalization
    try:
        audio_samples_2nd_read_pass, _fs = timbral_util.file_read(audio_samples, fs, phase_correction=phase_correction)
    except timbral_util.ZeroVolumeError as e:
        warnings.warn(str(e) + "\nAll AudioCommons timbre features will be set to zero.")
        for k in timbre:
            timbre[k] = 0.0
        return timbre if output_type == "dictionary" else [v for k, v in timbre.items()]

    # Pre-computed audio data, which is going to be used by several individual feature extractors
    windowed_audio = timbral_util.window_audio(audio_samples_2nd_read_pass)  # Original: always 4096 window size
    # specific_loudness is computed for each windowed audio frame
    windows_specific_loudness = list()
    windows_RMS = list()
    for i in range(windowed_audio.shape[0]):
        N_entire, N_single = timbral_util.specific_loudness(windowed_audio[i, :], fs=fs)
        windows_specific_loudness.append((N_entire, N_single, ))
        windows_RMS.append(np.sqrt(np.mean(windowed_audio[i, :] * windowed_audio[i, :])))
    # 20Hz highpass audio - run 3 times to get -18dB per octave - unstable filters produced when using a 6th order
    hp20Hz_audio_samples = timbral_util.filter_audio_highpass(audio_samples_2nd_read_pass, crossover=20, fs=fs)
    hp20Hz_audio_samples = timbral_util.filter_audio_highpass(hp20Hz_audio_samples, crossover=20, fs=fs)
    hp20Hz_audio_samples = timbral_util.filter_audio_highpass(hp20Hz_audio_samples, crossover=20, fs=fs)
    # Store all of those into a dict
    audio_data = {
        'audio_samples': audio_samples_2nd_read_pass,
        'fs': fs,
        'windowed_audio': windowed_audio,
        'windows_specific_loudness': windows_specific_loudness,
        'windows_RMS': np.asarray(windows_RMS),
        'hp20Hz_audio_samples': hp20Hz_audio_samples
    }

    # TODO maybe pre-compute librosa HPSS here? And add a general "Percussive" timbre feature
    #   also: harmonic_med, harmonic_IQR, etc... might contain must less noise than timbretoolbox's estimations?
    #   also compute residuals? Less noisy than TT's 'noise' computations?

    # functions can be given audio samples as well
    try:
        timbre['hardness'] = timbral_hardness(audio_data, dev_output=dev_output, clip_output=clip_output)
        timbre['depth'] = timbral_depth(audio_data, dev_output=dev_output, clip_output=clip_output)
        timbre['brightness'] = timbral_brightness(audio_data, dev_output=dev_output, clip_output=clip_output)
        timbre['roughness'] = timbral_roughness(audio_data, dev_output=dev_output, clip_output=clip_output)
        try:
            timbre['warmth'] = timbral_warmth(audio_data, dev_output=dev_output, clip_output=clip_output)
        except ValueError as e:  # observed: error when computing a minimum on an empty set of peaks
            warnings.warn(str(e) + "\nAudioCommons 'warmth' feature will be set to zero.")
            timbre['warmth'] = 0.0
        timbre['sharpness'] = timbral_sharpness(audio_data, dev_output=dev_output, clip_output=clip_output)
        timbre['boominess'] = timbral_booming(audio_data, dev_output=dev_output, clip_output=clip_output)
    except librosa.util.exceptions.ParameterError as e:
        # librosa.util.exceptions.ParameterError: Audio buffer is not finite everywhere
        warnings.warn(str(e) + "\nAll AudioCommons timbre features will be set to zero.")
        for k in timbre:
            timbre[k] = 0.0
    # reverb calculated on all channels
    if not exclude_reverb:
        timbre['reverb'] = timbral_reverb(multi_channel_audio, fs=fs)

    # Format output
    return timbre if output_type == "dictionary" else [v for k, v in timbre.items()]


def timbral_extractor_multiprocessing(files_names: Sequence[str], n_workers: int, **kwargs):

    list_of_args = [(s, kwargs) for s in files_names]
    with multiprocessing.Pool(8) as mp_pool:
        ac_raw_features_df = mp_pool.map(_timbral_extractor__mp, list_of_args)
    return ac_raw_features_df


def _timbral_extractor__mp(args):
    fname, kwargs = args[0], args[1]
    return timbral_extractor(fname, **kwargs)

