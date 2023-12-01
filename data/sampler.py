"""
Samplers for any abstract PresetDataset class, which can be used as train/valid/test samplers.
Support k-fold cross validation and subtleties of multi-note (multi-layer spectrogram) preset datasets.

"""
import warnings
from collections.abc import Iterable
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.utils.data

from data.abstractbasedataset import AudioDataset


_SEED_OFFSET = 6357396522630986725  # because PyTorch recommends to use a seed "with a lot of 0 and 1 bits"


#
class SubsetDeterministicSampler(torch.utils.data.Sampler):
    r""" Samples elements from a given list of indices, without replacement. Deterministic subset sampler for
    validation and test datasets (to replace PyTorch's SubsetRandomSampler)
    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(list(self.indices))

    def __len__(self):
        return len(self.indices)


def get_subsets_indexes(dataset: AudioDataset,
                        k_fold=0, k_folds_count=5, test_holdout_proportion=0.2,
                        random_seed=0
                        ) -> Dict[str, List[int]]:
    """
    Builds 'train', 'validation' and 'test' arrays of indexes

    :param dataset: Required to properly separate dataset items indexes by preset UIDs (not to split
        a multi-note preset into multiple subsets).
    :param k_fold: Current k-fold cross-validation fold index. If -1, builds a 'special fold' that randomly assigns
        indexes from 'normal' folds to the train and validation sets (test items are never part of those 'special'
        train/valid sets.). The -1 value can be useful for pre-training part of a model, before doing some proper
        k-fold cross-validation and test.
    :param k_folds_count: Total number of k-folds
    :param test_holdout_proportion: Proportion of 'test' data, excluded from cross-validation folds.
    :param random_seed: For reproducibility, always use the same seed

    :returns: dict of arrays of indexes (not UIDs)
    """
    from data.dexeddataset import DexedDataset
    if isinstance(dataset, DexedDataset):
        assert False, "Automatic subsets building should not be used for Dexed"

    presets_count = dataset.valid_presets_count
    all_preset_indexes = np.arange(presets_count)
    preset_indexes = dict()
    rng = np.random.default_rng(seed=random_seed)
    # Shuffle preset indexes, and separate them into subsets
    rng.shuffle(all_preset_indexes)  # in-place shuffling
    first_test_idx = int(np.floor(presets_count * (1.0 - test_holdout_proportion)))
    non_test_preset_indexes, preset_indexes['test'] = np.split(all_preset_indexes, [first_test_idx])
    # All folds are retrieved - we'll choose only one of these as validation subset, and merge the others
    #    'special fold' -1 for pre-training a single model: the 'validation' fold contains samples for all
    #    normal k-folds (test held-out set is of course always excluded)
    #    We just shuffle the indexes one more time before splitting, then arbitrarily use the 'new' first fold
    if k_fold == -1:
        rng.shuffle(non_test_preset_indexes)
        k_fold = 0
    preset_indexes_folds = np.array_split(non_test_preset_indexes, k_folds_count)
    preset_indexes['validation'] = preset_indexes_folds[k_fold]
    preset_indexes['train'] = np.hstack([preset_indexes_folds[i] for i in range(k_folds_count) if i != k_fold])
    # Final indexes
    if dataset.midi_notes_per_preset == 1 or dataset.multichannel_stacked_spectrograms:
        final_indexes = preset_indexes
    else:  # multi-note, single-layer spectrogram dataset: dataset indexes are not preset indexes
        final_indexes = dict()
        # We don't need to shuffle again these groups (SubsetRandomSampler will do it)
        for k in preset_indexes:  # k: train, valid or test
            final_indexes[k] = list()
            for preset_idx in preset_indexes[k]:
                final_indexes[k] += [preset_idx * dataset.midi_notes_per_preset + i
                                     for i in range(dataset.midi_notes_per_preset)]
    # ask the dataset to save those indexes
    dataset.save_indices_splits(final_indexes)
    return final_indexes


def build_subset_samplers(dataset: AudioDataset, random_seed=0, verbose=False) -> Dict[str, torch.utils.data.Sampler]:
    """
    Builds 'train', 'validation' and 'test' subset samplers

    Args description: see get_subsets_indexes(...)
    """
    splits_names = ['train', 'validation', 'test']
    final_indexes = {k: dataset.get_split_indices(k, exclude_zero_volume_items=True) for k in splits_names}
    subset_samplers = dict()
    for k in final_indexes:
        if k.lower() == 'train':
            torch_rng = torch.Generator().manual_seed(_SEED_OFFSET + random_seed)
            subset_samplers[k] = torch.utils.data.SubsetRandomSampler(final_indexes[k], generator=torch_rng)
        else:
            subset_samplers[k] = SubsetDeterministicSampler(final_indexes[k])
    return subset_samplers



