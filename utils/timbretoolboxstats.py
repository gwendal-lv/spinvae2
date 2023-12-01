"""
Statistics and other information about TimbreToolbox features
"""

# list of excluded features
excluded_features = ["Noisiness_"]  # Seems broken, at least for DX7 sound

# Features that should be projected onto a perceptual log scale
log_scale_features = [
    # 2022 log-scale features
    # Values in Hz
    'F0_', 'SpecCent_', 'SpecSpread_', 'SpecRollOff_'
    # ratio explodes (seems to be exponentially distributed)
    'OddEvenRatio_',

    # New log scales used in 2023
    # Energies (perceived on a log scale)
    'FrameErg_', 'RMSEnv_', 'AmpMod', 'HarmErg_', 'NoiseErg_', 'HarmDev_',
    # Also to the following features:
    #    Spectral Flatness is the ratio geometricmean/arithmeticmean (of amplitudes)
    #    SpecKurt : 4th order moment ; homogeneous to Hz^4
    #    Spectral Variation (Spectral Flux): variation of amplitude between time frames; perceptually on a log scale
    #    In-Harmonicity: is a ratio of Hz x a^2 / (Hz x a^2) (log scale seems appropriate, and we observe it improves
    #       the repartition of values)
    'SpecFlat_', 'SpecKurt_', 'SpecVar_', 'InHarm_',
]

# Eps to be added inside the log, for numerical stability (Values OK for the Dexed dataset)
log_scale_dynamic_eps = {
    'FrameErg_med': 7.8423e-05, 'FrameErg_IQR': 0.0001094,
    'SpecFlat_med': 1.5612e-06, 'SpecFlat_IQR': 2.3056e-06,
    'SpecKurt_med': 0.434259, 'SpecKurt_IQR': 0.823675,
    'SpecVar_med': 1.0158000000000002e-05, 'SpecVar_IQR': 4.5427000000000004e-05,
    'HarmDev_med': 3.0108499999999997e-07, 'HarmDev_IQR': 5.570400000000001e-07,
    'HarmErg_med': 1.36225e-08, 'HarmErg_IQR': 5.32195e-08,
    'InHarm_med': 1.5094499999999998e-06, 'InHarm_IQR': 1.40565e-05,
    'NoiseErg_med': 0.0007494800000000001, 'NoiseErg_IQR': 0.0010178,
    'AmpMod': 0.00047665,
    'RMSEnv_med': 0.0004319, 'RMSEnv_IQR': 0.00029309000000000003
}
