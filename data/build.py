"""
Utility function for building datasets and dataloaders using given configuration arguments.
"""
import copy
import sys
import warnings
from typing import Optional

import numpy as np

import torch.utils.data
from torch.utils.data import DataLoader

from data.abstractbasedataset import AudioDataset
from . import dataset
import data.sampler

import config


def get_dataset(model_config: config.ModelConfig, train_config: config.TrainConfig):
    """
    Returns the full (main) dataset.
    """
    # Compatibility with legacy models
    try:
        n_extra_random_presets = train_config.n_extra_random_presets
    except AttributeError:
        warnings.warn("Can't find train_config.n_extra_random_presets from the given configuration (this is OK "
                      "for older models). Will use the default 0 value instead.")
        n_extra_random_presets = 0
    # THis function handles Dexed only
    if model_config.synth.startswith('dexed'):
        full_dataset = dataset.DexedDataset(
            ** dataset.model_config_to_dataset_kwargs(model_config),
            restrict_to_labels=model_config.dataset_labels,
            n_extra_random_presets=n_extra_random_presets,
        )
    else:
        raise NotImplementedError("No dataset available for '{}': unrecognized synth.".format(model_config.synth))
    # config.py direct dirty modifications - number of learnable params depends on the synth and dataset arguments
    model_config.synth_params_count = full_dataset.learnable_params_count
    return full_dataset


def get_pretrain_datasets(model_config: config.ModelConfig, train_config: config.TrainConfig):
    kwargs = {
        'dummy_synth_params_tensor': True,
        'k_folds_count': train_config.k_folds, 'test_holdout_proportion': train_config.test_holdout_proportion,
        'n_extra_random_presets': train_config.n_extra_random_presets,
    }
    train_ds = dataset.MergedDataset(model_config, dataset_type='train', **kwargs)
    valid_ds = dataset.MergedDataset(model_config, dataset_type='validation', **kwargs)
    return train_ds, valid_ds


def get_num_workers(train_config: config.TrainConfig):
    """ Returns the appropriate number of multi-processing workers (might be 0) depending on the current config
     and debugging status. """
    if sys.gettrace() is not None:
        print("[data/build.py] Debugger detected - num_workers=0 for all DataLoaders")
        num_workers = 0  # PyCharm debug behaves badly with multiprocessing...
    else:  # We should use an higher CPU count for real-time audio rendering
        # 4*GPU count: optimal w/ light dataloader (e.g. (mel-)spectrogram computation)
        num_workers = min(train_config.minibatch_size, torch.cuda.device_count() * 4)
    return num_workers


def get_split_dataloaders(train_config: config.TrainConfig, full_dataset: AudioDataset,
                          num_workers: Optional[int] = None, persistent_workers=True):
    """ Returns a dict of train/validation/test DataLoader instances, and a dict which contains the
    length of each sub-dataset. """
    # Num workers might be zero (no multiprocessing)
    if num_workers is None:
        num_workers = get_num_workers(train_config)
    # Dataloader easily build from samplers
    subset_samplers = data.sampler.build_subset_samplers(full_dataset, verbose=(train_config.verbosity >= 1))
    dataloaders = dict()
    sub_datasets_lengths = dict()
    batch_size = train_config.minibatch_size
    for k, sampler in subset_samplers.items():
        # Last train minibatch must be dropped to help prevent training instability. Worst case example, last minibatch
        # contains only 8 elements, mostly sfx: these hard to learn (or generate) item would have a much higher
        # equivalent learning rate because all losses are minibatch-size normalized. No issue for eval though
        drop_last = (k.lower() == 'train')
        ds = copy.deepcopy(full_dataset)
        ds._data_augmentation = (k.lower() == 'train')
        # Dataloaders based on previously built samplers (don't need to set shuffle to True)
        dataloaders[k] = torch.utils.data.DataLoader(ds, batch_size=batch_size, drop_last=drop_last,
                                                     sampler=sampler, num_workers=num_workers,
                                                     pin_memory=train_config.dataloader_pin_memory,
                                                     persistent_workers=((num_workers > 0) and persistent_workers))
        # actual nb of dataloader items length depends on drop last, or not
        if drop_last:
            sub_datasets_lengths[k] = (len(sampler.indices) // batch_size) * batch_size
        else:
            sub_datasets_lengths[k] = len(sampler.indices)
        if train_config.verbosity >= 1:
            print("[data/build.py] Dataset '{}' contains {}/{} samples ({:.1f}%). num_workers={}"
                  .format(k, sub_datasets_lengths[k], len(full_dataset),
                          100.0 * sub_datasets_lengths[k]/len(full_dataset), num_workers))
    return dataloaders, sub_datasets_lengths


def get_pretrain_dataloaders(model_config, train_config,
                             train_ds: dataset.MergedDataset, valid_ds: dataset.MergedDataset):
    """ Returns a dict with 'train' and 'validation' dataloaders, and the length or corresponding datasets
     Return is consistent with get_split_dataloaders(...). """
    use_wsampler = train_config.pretrain_synths_max_imbalance_ratio > 0.0  # Only for the training dataloader
    imbalance_ratio = train_config.pretrain_synths_max_imbalance_ratio if use_wsampler else None
    train_dl, train_nb_items = \
        train_ds.get_dataloader(batch_size=train_config.minibatch_size, use_weighted_sampler=use_wsampler,
                                max_imbalance_ratio=imbalance_ratio, num_workers=get_num_workers(train_config),
                                pin_memory=train_config.dataloader_pin_memory,
                                persistent_workers=train_config.dataloader_persistent_workers)
    valid_dl, valid_nb_items = \
        valid_ds.get_dataloader(batch_size=train_config.minibatch_size, use_weighted_sampler=False,
                                max_imbalance_ratio=None,
                                num_workers=get_num_workers(train_config),
                                pin_memory=train_config.dataloader_pin_memory,
                                persistent_workers=train_config.dataloader_persistent_workers)
    if train_config.verbosity >= 1:
        print("[data/build.py] MergedDataset 'train' contains {}/{} samples ({:.1f}% of train dataset)"
              .format(train_nb_items, len(train_ds), 100.0 * train_nb_items / len(train_ds)))
        print("[data/build.py] MergedDataset 'validation' contains {}/{} samples ({:.1f}% of validation dataset)"
              .format(valid_nb_items, len(valid_ds), 100.0 * valid_nb_items / len(valid_ds)))
    return ({'train': train_dl, 'validation': valid_dl},
            {'train': train_nb_items, 'validation': valid_nb_items})

