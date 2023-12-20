"""
Evaluate audio features, then interpolation smoothness and nonlinearity, from directories containing
audio files only (does not require any given model to render audio interpolations beforehand).
"""
import pathlib

import evaluation.interpbase


class AudioOnlyEvaluation(evaluation.interpbase.InterpBase):

    def __init__(self, num_steps: int,
                 storage_path: pathlib.Path, reference_storage_path: pathlib.Path,
                 **kwargs):
        """
        Class to help compute audio features and interpolation metric for pre-rendered audio sequences.
        The sequences are considered to be interpolations.
        """
        super().__init__(
            dataset=None,
            num_steps=num_steps, reference_storage_path=reference_storage_path,
            **kwargs
        )
        self._storage_path = storage_path

    @property
    def storage_path(self) -> pathlib.Path:
        return self._storage_path

    def render_audio(self):
        assert False, "This class is intended to work with pre-rendered audio sequences.."


def run_evaluation(num_steps, storage_path, reference_storage_path):
    # TODO ok now???
    _audio_eval = AudioOnlyEvaluation(num_steps, storage_path, reference_storage_path, cpu_usage="moderate")
    _audio_eval.compute_and_save_interpolation_metrics()


if __name__ == "__main__":
    # FIXME ?  multiproc spawn issue when data is in the __main__ section?

    _storage_path = pathlib.Path(
        pathlib.Path(__file__).joinpath(
            "../../../spinvae_notebooks/generated_data/" 
            "export_audio__noAudIn_ARzCks_Dz512_cnn8x1_big_mixt3_audLR2e-04_b5e-05_g1e-02__interp7_test_audio2.0_0.5_trimmed"
        )
    )
    run_evaluation(7, _storage_path, _storage_path)
