import unittest
import numpy as np

import synth.dexed


class DexedQuantification(unittest.TestCase):
    def test_random_preset_quantification(self):
        sr = 16000
        midi_note = (56, 75)
        for seed in range(10):
            dexed_synth = synth.dexed.Dexed(sr, filter_plugin_loading_errors=False)
            preset = dexed_synth.get_random_preset(seed=seed)
            dexed_synth.assign_preset(preset)
            audio_ref, _ = dexed_synth.render_note(*midi_note)

            dexed_synth = synth.dexed.Dexed(sr, filter_plugin_loading_errors=False)
            preset = dexed_synth.get_random_preset(seed=seed)
            for param_idx in range(len(preset)):
                card = dexed_synth.get_param_cardinality(param_idx)
                # TODO EXPLAIN : even a VERY small noise leads to different audio renders...
                #    BUT the results are very consistent
                #    explanation:? maybe Dexed (as opposed to the true DX7) actually allows for true-continuous params?
                #        (not limited to 100 different values, but uses true floats internally?)
                noise = 0.001
                preset[param_idx] = (param_idx, preset[param_idx][1] + noise)
            dexed_synth.assign_preset(preset)
            audio_noisy_params, _ = dexed_synth.render_note(*midi_note)
            self.assertTrue(np.all(audio_ref == audio_noisy_params))


if __name__ == '__main__':
    unittest.main()
