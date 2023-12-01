"""
Functions and classes to easily load models or data for evaluation.
"""
import pathlib
import typing
import warnings
from pathlib import Path
import pickle

import torch

import config
import data.build
import model.hierarchicalvae


class ModelLoader:
    def __init__(self, model_folder_path: typing.Union[Path, str], device='cpu', dataset_type='validation'):
        """
        Loads model_config and train_config from the given folder, then builds the model and loads the last
        available checkpoint.
        The dataset as well as the required dataloader are also built.

        :param model_folder_path: e.g. Path('/home/user/saved/MMD_tests/model1')
        :param dataset_type: Usually 'validation' or 'test', but 'train' is accepted.
        """
        self.device = device
        self.path_to_model_dir = pathlib.Path(model_folder_path)
        self.model_config, self.train_config = self.get_model_train_configs(self.path_to_model_dir)
        if device == 'cpu':
            self.train_config.main_cuda_device_idx = -1

        # Build dataset, dataloaders
        self.dataset_type = dataset_type
        if self.train_config.pretrain_audio_only:
            raise NotImplementedError("Only fine-tuned and complete models can be loaded by this class.")
        else:
            # Parts of train.py code
            self.dataset = data.build.get_dataset(self.model_config, self.train_config)
            dataloaders, dataloaders_nb_items = data.build.get_split_dataloaders(self.train_config, self.dataset)
            self.dataloader = dataloaders[self.dataset_type]
            self.dataloader_num_items = dataloaders_nb_items[self.dataset_type]

        # Then build model and load its weights
        try:
            self.ae_model = model.hierarchicalvae.HierarchicalVAE(
                self.model_config, self.train_config, self.dataset.preset_indexes_helper)
            self.legacy_model = False
        except AttributeError as e:
            self.ae_model, self.legacy_model = None, True  # indicates that this old model can't be loaded anymore
            warnings.warn(f"[{self.__class__.__name__}] Can't load {model_folder_path}, will be considered a legacy "
                          f"model. The exception raised during loading was: {str(e)}")
        if self.ae_model is not None:
            self.ae_model.load_checkpoints(
                self.path_to_model_dir.joinpath("checkpoint.tar"), map_location=torch.device(device))

        if device == 'cpu':
            torch.cuda.empty_cache()  # Checkpoints were usually GPU tensors (originally)

    @staticmethod
    def get_model_train_configs(model_dir: Path):
        with open(model_dir.joinpath("config.pickle"), 'rb') as f:
            checkpoint_configs = pickle.load(f)
        model_config: config.ModelConfig = checkpoint_configs['model']
        train_config: config.TrainConfig = checkpoint_configs['train']
        return model_config, train_config


if __name__ == "__main__":
    _model_path = Path(__file__).resolve().parent.parent
    _model_path = _model_path.joinpath('../Data_SSD/Logs/preset-vae/pAE/combin_CEsmooth0.00_beta1.6e-04')
    loader = ModelLoader(_model_path)
    print("OK")

