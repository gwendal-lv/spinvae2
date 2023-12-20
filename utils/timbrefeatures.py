import copy
import pathlib
import pickle
from typing import Sequence

import pandas as pd
import numpy as np

import utils.timbretoolboxstats


class TimbreFeatures:
    def __init__(self, raw_features_file: pathlib.Path):
        """
        Class for loading raw timbre features (AudioCommons and TimbreToolbox) which must have been computed and
        stored as a table-like file,
        and for post-processing them.
        """
        # load from pandas or from csv
        if raw_features_file.name.endswith(".df.pickle"):
            with open(raw_features_file, "rb") as f:
                self.raw_df = pickle.load(f)
                assert isinstance(self.raw_df, pd.DataFrame)
        elif raw_features_file.name.endswith(".csv"):
            with open(raw_features_file, "r") as f:
                self.raw_df = pd.read_csv(raw_features_file)
        else:
            assert False, f"Can't identify the type of file {raw_features_file} from its extension"

        # Build subsets of columns
        self.non_feature_cols = [c for c in self.raw_df if not (c.startswith("ac_") or c.startswith("tt_"))]
        self.raw_ac_cols = [c for c in self.raw_df if c.startswith("ac_")]
        self.raw_tt_cols = [c for c in self.raw_df if c.startswith("tt_")]

        # Values pre-processing: we pre-process all columns (even the ones we'll eventually discard)
        self.preproc_df = copy.deepcopy(self.raw_df)  # For the raw_df to be displayable
        self.preproc_df = self.preproc_df.replace([np.inf, -np.inf], np.nan)  # We discard infs and use nans instead
        # identify NaNs and not-computed values: we build a new bool
        #    dataframe to indicate which cells should be replaced by the column's median value
        # Default 0.0 when AudioCommmons computation has failed. AC feature values are based on regression
        #     models, so the probability of a value to be 0.0 is 0.0 (it was a default 'failed' value).
        #     In particular, the ac_roughness feature computation often fails
        self._preproc_invalid_bool_mask = (self.preproc_df[self.raw_feature_cols] == 0.0)
        self._preproc_invalid_bool_mask[self.raw_tt_cols] = False  # 0.0 values from AC features only
        # Add NaNs to this mask
        self._preproc_invalid_bool_mask = self._preproc_invalid_bool_mask | self.preproc_df[self.raw_feature_cols].isna()
        # Now: actually replace w/ the median value (column-by-column)
        for c in self.raw_feature_cols:
            col_values = self.preproc_df[c]
            mask = self._preproc_invalid_bool_mask[c]
            median_value = col_values[~mask].median()
            # "SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame"
            #     -> pd doc says: "care must be taken to avoid what is called chained indexing"
            # We use .loc to access the DF by rows (first) and columns (last), using label(s) or bool mask
            self.preproc_df.loc[mask, c] = median_value

        # Post-processing: discard inappropriate columns, apply log scales, then normalize
        self.ac_cols = self.raw_ac_cols  # keep all audio commons columns
        self.tt_cols = list()
        for c in self.raw_tt_cols:
            # We remove _min and _max values (keep only the median and IQR)
            if not (c.endswith('_min') or c.endswith('_max')):
                is_excluded = False
                for excluded_name in utils.timbretoolboxstats.excluded_features:
                    if excluded_name in c:
                        is_excluded = True
                if not is_excluded:
                    self.tt_cols.append(c)
        self.postproc_df = copy.deepcopy(self.preproc_df[self.non_feature_cols + self.feature_cols])
        freq_eps = 1.0
        # Apply log scales to TT features only
        for col in self.tt_cols:
            # Modifications applied to 2022 interpolation timbre extraction
            # values in Hz
            if col.startswith('tt_F0_') or col.startswith('tt_SpecCent_') \
                    or col.startswith('tt_SpecSpread_') or col.startswith('tt_SpecRollOff_'):
                self.postproc_df[col] = np.log(freq_eps + self.postproc_df[col])
            # ratio explodes (seems to be exponentially distributed)
            elif col.startswith('tt_OddEvenRatio_'):
                self.postproc_df[col] = np.log(1.0 + self.postproc_df[col])

            # New (april 2023) log scales applied to energies
            elif col.startswith('tt_FrameErg_') or col.startswith('tt_RMSEnv_') or col.startswith('tt_AmpMod') \
                    or col.startswith('tt_HarmErg_') or col.startswith('tt_NoiseErg_') or col.startswith('tt_HarmDev_'):
                    dynamic_eps = utils.timbretoolboxstats.log_scale_dynamic_eps[col.replace('tt_', '')]
                    self.postproc_df[col] = np.log(dynamic_eps + self.postproc_df[col])
            # Also to the following features:
            #    Spectral Flatness is the ratio geometricmean/arithmeticmean (of amplitudes)
            #    SpecKurt : 4th order moment ; homogeneous to Hz^4
            #    Spectral Variation (Spectral Flux) : variation of amplitude between time frames;
            #       perceptually on a log scale
            #    In-Harmonicity: is a ratio of Hz x a^2 / (Hz x a^2) (log scale seems appropriate, and we
            #       observe it improves the repartition of values)
            elif col.startswith('tt_SpecFlat_') or col.startswith('tt_SpecKurt_') \
                    or col.startswith('tt_SpecVar_') or col.startswith('tt_InHarm_'):
                dynamic_eps = utils.timbretoolboxstats.log_scale_dynamic_eps[col.replace('tt_', '')]
                self.postproc_df[col] = np.log(dynamic_eps + self.postproc_df[col])
        # Finally: normalize (don't normalize using ALL preset UIDs: use training subset stats only)
        mean, std = pd.Series(_post_distorsion_stats['mean']), pd.Series(_post_distorsion_stats['std'])
        self.postproc_df[self.feature_cols] = (self.postproc_df[self.feature_cols] - mean) / std

    @property
    def raw_feature_cols(self):
        return self.raw_ac_cols + self.raw_tt_cols

    @property
    def feature_cols(self):
        return self.ac_cols + self.tt_cols


# Features with a very high correlation with another
#     removing these features ensure all Spearman correlations are < 0.9
highly_correlated_features = [
    'ac_sharpness',  # Correlated to ac_brightness but the latter has a nicer distribution

    # Spec Skewness and Slope and really weird distribution, and are correlated to other features
    #    Skew <-> Kurtosis (as expected...) but Kurtosis is nicely distributed
    'tt_SpecSkew_med', 'tt_SpecSkew_IQR',
    'tt_SpecSlope_med', 'tt_SpecSlope_IQR',

    # We could have removed RollOff or Centroid, but Centroid seems to contain more outliers
    'tt_SpecCent_med',  # RollOff/Centroid IQRs are not as correlated, though

    # Highly correlated to both Effective Duration and Temporal Centroid
    'tt_Rel',

    # keep RMSEnv or FrameErg? RMS, computed from the temporal energy envelope
    'tt_FrameErg_med', 'tt_FrameErg_IQR',

    # idem LAT
    'tt_AttSlope',

    # Harmonic/Noise contents:
    'tt_NoiseErg_med', 'tt_NoiseErg_IQR',
    'tt_HarmErg_med', 'tt_HarmDev_med',  # medians are correlated to RMS, but IQRs are not
    'tt_HarmDev_IQR',  # (but are correlated to the other)
]


def parse_timbre_features_arguments(timbre_arguments: Sequence[str]):
    """ Parses a list of arguments and/or timbre feature names, in order to easily build a long list of features. """
    explicit_timbre_feat_list = []
    available_features_names = list(_post_distorsion_stats['mean'].keys())
    remove_high_corr = False
    for timbre_arg in timbre_arguments:
        if timbre_arg == 'ac_*':
            explicit_timbre_feat_list += [f for f in available_features_names if f.startswith('ac_')]
        elif timbre_arg == 'tt_*':
            explicit_timbre_feat_list += [f for f in available_features_names if f.startswith('tt_')]
        elif timbre_arg == '__no_high_corr__':
            remove_high_corr = True
        else:
            assert timbre_arg in available_features_names, f"Argument {timbre_arg} is not a valid feature name"
            explicit_timbre_feat_list.append(timbre_arg)
    # Remove highly correlated, if requested
    if remove_high_corr:
        explicit_timbre_feat_list = [f for f in explicit_timbre_feat_list if f not in highly_correlated_features]
    # checks: duplicates
    assert len(set(explicit_timbre_feat_list)) == len(explicit_timbre_feat_list)
    return explicit_timbre_feat_list


# Normalization values (extracted from Dexed train dataset, after log scales have been applied to features)
_post_distorsion_stats = {
    'mean': {
        'ac_hardness': 46.25082999726012,
        'ac_depth': 52.994353757269884,
        'ac_brightness': 51.957319354268435,
        'ac_roughness': 42.427876218870274,
        'ac_warmth': 51.380621548202896,
        'ac_sharpness': 37.37998058489299,
        'ac_boominess': 32.210331509854406,
        'tt_FrameErg_med': -5.349435478986149,
        'tt_FrameErg_IQR': -4.869288264013964,
        'tt_SpecCent_med': 5.814110022286996,
        'tt_SpecCent_IQR': 4.459029358930566,
        'tt_SpecCrest_med': 92.8425140773873,
        'tt_SpecCrest_IQR': 30.863566039453026,
        'tt_SpecDecr_med': -0.03771411046491118,
        'tt_SpecDecr_IQR': 0.16035011540603247,
        'tt_SpecFlat_med': -8.024369637995669,
        'tt_SpecFlat_IQR': -7.719016685312941,
        'tt_SpecKurt_med': 4.364715936274617,
        'tt_SpecKurt_IQR': 4.829726920096817,
        'tt_SpecRollOff_med': 6.454392136178145,
        'tt_SpecRollOff_IQR': 4.958185640390218,
        'tt_SpecSkew_med': 11.885978159911046,
        'tt_SpecSkew_IQR': 11.712902626726114,
        'tt_SpecSlope_med': -1.2669591793349869e-06,
        'tt_SpecSlope_IQR': 8.510235375082157e-08,
        'tt_SpecSpread_med': 5.287426124956959,
        'tt_SpecSpread_IQR': 4.571149101848172,
        'tt_SpecVar_med': -6.758642822036731,
        'tt_SpecVar_IQR': -5.517316781760121,
        'tt_F0_med': 4.935726860617339,
        'tt_F0_IQR': 1.077489130266382,
        'tt_HarmDev_med': -10.884678767239231,
        'tt_HarmDev_IQR': -10.183231079914714,
        'tt_HarmErg_med': -13.717060451314953,
        'tt_HarmErg_IQR': -12.371201147187032,
        'tt_InHarm_med': -9.04315744124908,
        'tt_InHarm_IQR': -7.218958095573689,
        'tt_NoiseErg_med': -3.090975134853777,
        'tt_NoiseErg_IQR': -2.6383382683925434,
        'tt_OddEvenRatio_med': 3.5236072578710473,
        'tt_OddEvenRatio_IQR': 5.148128004737989,
        'tt_AmpMod': -3.295489739659602,
        'tt_Att': 0.06542898946684286,
        'tt_AttSlope': 9.534402833223368,
        'tt_Dec': 0.36540522815996046,
        'tt_DecSlope': -1.111063791736328,
        'tt_EffDur': 2.0588298778628356,
        'tt_FreqMod': 3.509300499319708,
        'tt_LAT': -0.9827694551120438,
        'tt_RMSEnv_med': -3.5859992051019027,
        'tt_RMSEnv_IQR': -3.7915446972163065,
        'tt_Rel': 2.175170428857356,
        'tt_TempCent': 1.2552647103536634
    },
    'std': {
        'ac_hardness': 11.808880827963613,
        'ac_depth': 13.061459041983479,
        'ac_brightness': 11.762881820945017,
        'ac_roughness': 10.415979318575216,
        'ac_warmth': 9.846089989910858,
        'ac_sharpness': 12.001339018931594,
        'ac_boominess': 11.564223616152661,
        'tt_FrameErg_med': 2.241519428645993,
        'tt_FrameErg_IQR': 1.8290137140331038,
        'tt_SpecCent_med': 0.940458212500928,
        'tt_SpecCent_IQR': 1.558317403004967,
        'tt_SpecCrest_med': 34.12033231862072,
        'tt_SpecCrest_IQR': 23.617394608808276,
        'tt_SpecDecr_med': 0.525315653068968,
        'tt_SpecDecr_IQR': 0.5868924041122475,
        'tt_SpecFlat_med': 2.1337394433425914,
        'tt_SpecFlat_IQR': 2.382584727138608,
        'tt_SpecKurt_med': 2.304660438705014,
        'tt_SpecKurt_IQR': 2.5427122033890193,
        'tt_SpecRollOff_med': 1.0436534451600026,
        'tt_SpecRollOff_IQR': 2.222893724408737,
        'tt_SpecSkew_med': 21.46576552252813,
        'tt_SpecSkew_IQR': 19.72383740900833,
        'tt_SpecSlope_med': 2.6514023476074534e-07,
        'tt_SpecSlope_IQR': 1.6665176050805084e-07,
        'tt_SpecSpread_med': 1.0214841337690819,
        'tt_SpecSpread_IQR': 1.3251252679435352,
        'tt_SpecVar_med': 2.8839447383346077,
        'tt_SpecVar_IQR': 2.4169653690576314,
        'tt_F0_med': 1.0768639126308746,
        'tt_F0_IQR': 1.396259041506176,
        'tt_HarmDev_med': 2.4155490591220015,
        'tt_HarmDev_IQR': 2.0575592347519014,
        'tt_HarmErg_med': 3.4121732850766704,
        'tt_HarmErg_IQR': 3.000375537223636,
        'tt_InHarm_med': 3.2885640016510984,
        'tt_InHarm_IQR': 2.5685981397223476,
        'tt_NoiseErg_med': 2.2345549551152324,
        'tt_NoiseErg_IQR': 1.8157738591247137,
        'tt_OddEvenRatio_med': 5.512294128852568,
        'tt_OddEvenRatio_IQR': 6.265602703417261,
        'tt_AmpMod': 1.539758307730887,
        'tt_Att': 0.1451362594830177,
        'tt_AttSlope': 2.9031665059204625,
        'tt_Dec': 0.465129433600626,
        'tt_DecSlope': 2.2796958271283545,
        'tt_EffDur': 1.1342995400451468,
        'tt_FreqMod': 1.8725004475760514,
        'tt_LAT': 0.24492245362287682,
        'tt_RMSEnv_med': 1.5394277427921712,
        'tt_RMSEnv_IQR': 1.2214295250025398,
        'tt_Rel': 1.1647385557237713,
        'tt_TempCent': 0.5191677955238353
    }
}




if __name__ == "__main__":
    timbre_features = TimbreFeatures(
        pathlib.Path(__file__).joinpath('../../../Data_SSD/Datasets/Dexed/raw_timbre_features.df.pickle').resolve()
    )
