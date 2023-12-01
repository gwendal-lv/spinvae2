"""
Allows easy modification of all configuration parameters required to perform a series of models evaluations.
This script is not intended to be run, it only describes parameters.
"""
import copy
from pathlib import Path
from typing import List, Dict, Union, Any, Optional

from utils import config_confidential


class InterpEvalConfig:
    def __init__(self, dataset_type='test'):
        self.device = 'cpu'  # even 'cpu' uses some CUDA memory because we load a model that was on GPU
        self.cpu_usage = 'moderate'  # 'low', 'high' or 'moderate'
        self.dataset_type = dataset_type  # 'validation' or 'test'
        self.num_steps = 9
        self.verbose = True  # General information about progress, CPU time / item, ...
        self.verbose_postproc = False  # Detailed (Matlab) post-processing outputs

        self.use_reduced_dataset = False  # fast debugging (set to False during actual eval)
        self.skip_audio_render = False  # don't re-render audio, recompute interpolation features/metrics only
        self.force_re_eval_all = False  # skip_audio remains usable (only interp features will be force-evaluated)

        # Audio features and interpolation metrics
        # Features to be rejected
        #    - Noisiness features seem badly estimated for the DX7 (because of the FM?). They are quite constant
        #      equal to 1.0 (absurd) and any slightly < 1.0 leads to diverging values after 1-std normalization
        #    - __high_correlation__: remove features which are highly correlated to another (correlation values
        #          from Dexed's training dataset) - see utils/timbrefeatures.py
        self.excluded_interp_features = ('Noisiness', '__high_correlation__')

        # Reference model
        self.logs_root_dir = Path(config_confidential.logs_root_dir)
        # Reference data can be stored anywhere (they don't use a trained NN)
        self.reference_model_path = self.logs_root_dir.parent.joinpath('RefInterp/LinearNaive')
        self.ref_model_interp_path: Optional[Path] = None
        self.ref_model_force_re_eval = False

        # List of models and eval configs for each model
        #    - the config of the first model will be used to load the dataset used by the reference model
        #    - optional fields for any config: 'u_curve': 'linear', 'latent_interp': 'linear'

        self.other_models: List[Dict[str, Any]] = [

            # Spinvae et Spinvae2 (as references)
            {'base_model_name': 'pAE/spinvae_icassp23_copy_neweval_2023_08'},  # test eval only, regen not available anymore
            {'base_model_name': 'pAEmel_train2/noAudIn_ARzCks_Dz512_cnn8x1_big_mixt3_audLR2e-04_b5e-05_g1e-02'},  # SPINVAE 2

            # No attr reg
            {'base_model_name': 'pAEmel_train2/noAudIn_noAttrRegDz256_cnn8x1_big_mixt3_audLR2e-04_b5e-04'},

        ]
        # Auto duplicate everything to try arcsin u curves
        if False:
            other_models_duplicates = copy.deepcopy(self.other_models)
            for m_config in other_models_duplicates:
                m_config['u_curve'] = 'arcsin'
            self.other_models += other_models_duplicates
        # Auto duplicate everything to try all z refinement options FIXME REMOVE, deprecated
        if False:
            other_models_backup = copy.deepcopy(self.other_models)
            for refine_lvl in [1, 2]:
                other_models_duplicates = copy.deepcopy(other_models_backup)
                for m_config in other_models_duplicates:
                    m_config['refine_level'] = refine_lvl
                self.other_models += other_models_duplicates

        self.set_default_config_values()
        self.build_models_storage_path()

    @property
    def default_interp_curve(self):
        return 'linear'

    @property
    def default_refine_level(self):
        return 0

    def set_default_config_values(self):
        """ Sets default values for some argument that can be omitted """
        for m_config in self.other_models:
            # u (interp variable) and latent interp (z) curves
            for curve in ['u_curve', 'latent_interp']:
                try:
                    curve_type = m_config[curve]
                except KeyError:
                    curve_type = self.default_interp_curve
                m_config[curve] = curve_type
            # refine level: default is 0
            try:
                refine_lvl = m_config['refine_level']
            except KeyError:
                refine_lvl = self.default_refine_level
            m_config['refine_level'] = refine_lvl

    def build_models_storage_path(self):
        """ auto build eval data paths from the model name and interp-hyperparams """
        self.ref_model_interp_path = self.reference_model_path.joinpath(
            'interp{}_{}'.format(self.num_steps, self.dataset_type[0:5]))
        for m_config in self.other_models:
            m_config['base_model_path'] = self.logs_root_dir.joinpath(m_config['base_model_name'])
            interp_name = 'interp{}'.format(self.num_steps)
            interp_name += '_' + self.dataset_type[0:5]
            interp_name += '_u' + m_config['u_curve'][0:3].capitalize()
            interp_name += '_z' + m_config['latent_interp'][0:3].capitalize()
            refine_lvl = m_config['refine_level']
            interp_name += '_refi{}'.format(refine_lvl) if refine_lvl > 0 else ''
            # Set paths and names
            m_config['interp_storage_path'] = m_config['base_model_path'].joinpath(interp_name)
            m_config['model_interp_name'] = m_config['base_model_name'] + '/' + interp_name

