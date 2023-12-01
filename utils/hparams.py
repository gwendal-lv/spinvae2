from typing import Sequence, Optional
from abc import ABC, abstractmethod

import numpy as np


class DynamicParam(ABC):
    def __init__(self, current_epoch=0):
        self.current_epoch = current_epoch - 1  # This value will be incremented when epoch actually starts

    def on_new_epoch(self):
        self.current_epoch += 1

    @abstractmethod
    def get(self, current_epoch: Optional[int] = None):
        pass

    @property
    def value(self):
        return self.get()


class LinearDynamicParam(DynamicParam):
    """ Hyper-Parameter which is able to automatically increase or decrease at each epoch.
    It provides the same methods as a metric (see logs/metrics.py) and can be easily used with tensorboard. """
    def __init__(self, start_value, end_value, start_epoch=0, end_epoch=10, current_epoch=0):
        super().__init__(current_epoch)
        self.start_value = start_value
        self.end_value = end_value
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        assert self.end_epoch >= self.start_epoch

    def get(self, current_epoch: Optional[int] = None):
        """ Returns an interpolated value. Current epoch was automatically incremented by calling on_new_epoch()
        but can be passed as argument to this method. """
        epoch = self.current_epoch if current_epoch is None else current_epoch
        if epoch >= self.end_epoch:
            return self.end_value
        elif epoch <= self.start_epoch:
            return self.start_value
        else:
            offset_epochs = epoch - self.start_epoch
            return self.start_value + (self.end_value - self.start_value) * offset_epochs\
                                            / (self.end_epoch - self.start_epoch)

    @property
    def has_reached_final_value(self):
        return self.current_epoch >= self.end_epoch


class CyclicalParam(DynamicParam):
    def __init__(self, cycle_coordinates: Sequence[Sequence[int]], cycle_n_epochs: int, current_epoch=0):
        """
        Hyperparameter whose values are defined by a periodic piecewise-linear curve.

        :param cycle_coordinates: sequence of (x, value) pairs, where x-coordinates are normalized (in [0.0, 1.0])
        :param cycle_n_epochs: Number of epochs for each period (first period starts at epoch 0)
        """
        super().__init__(current_epoch)
        cycle_np = np.asarray(cycle_coordinates)
        assert len(cycle_np.shape) == 2
        assert cycle_np.shape[1] == 2
        assert cycle_np.shape[0] >= 2, "A cycle must be made of at least 2 points."

        self.cycle_times = cycle_np[:, 0]
        self.cycle_values = cycle_np[:, 1]
        self.cycle_n_epochs = cycle_n_epochs

        assert np.all(np.diff(self.cycle_times) > 0), "x-coordinates must be strictly increasing"
        assert np.isclose(self.cycle_times[0], 0.0) and np.isclose(self.cycle_times[-1], 1.0)

    def get(self, current_epoch: Optional[int] = None):
        epoch = self.current_epoch if current_epoch is None else current_epoch
        normalized_x = (epoch % self.cycle_n_epochs) / self.cycle_n_epochs  # 0.0 <= x < 1.0
        return np.interp([normalized_x], self.cycle_times, self.cycle_values).item()

