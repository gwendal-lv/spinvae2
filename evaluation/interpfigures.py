
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings

from evalconfig import InterpEvalConfig
from evaluation.interpbase import InterpBase
import utils.stat


def interp_results_boxplots(
        storage_paths: List[Path], models_names: Optional[List[str]] = None,
        metrics_to_plot=('smoothness', 'nonlinearity'),
        eval_config: Optional[InterpEvalConfig] = None,
        reference_model_idx=0,
        display_wilcoxon_tests=False,
        compact_display=False, figsize=None, legend_ax_idx=None,
):
    """

    :param storage_paths:
    :param models_names:
    :param metrics_to_plot:
    :param exclude_min_max:
    :param exclude_features:
    :param eval_config: if given, will be used to exclude_min_max, exclude_features, ....
    :param reference_model_idx: The model to be considered as the reference for normalization of all metrics.
    :param compact_display: Smaller optimized graph for paper print
    :param legend_ax_idx: if not None, only the axis designated by this index will have a legend
    :return:
    """
    # auto create model names if not given (use parent's name)
    if models_names is None:
        models_names = [p.parent.name + '/' + p.name for p in storage_paths]
    else:
        assert len(models_names) == len(storage_paths)
    # load data
    #    1st index: model index
    #    2nd index: metric type (e.g. smoothness, RSS, ...)
    #    3rd and 4th "dims": actual DataFrame whose index is an interp sequence index, columns are metrics' names
    models_interp_results = [InterpBase.get_interp_results(p, eval_config) for p in storage_paths]

    # for each feature, compute normalisation factors from the 1st model, to be used for all models
    #   mean "without outliers" gives the best boxplots
    reference_results = models_interp_results[reference_model_idx]
    reference_norm_factors = {k: utils.stat.means_without_outliers(results_df)
                              for k, results_df in reference_results.items()}

    # Detailed boxplots: each metric has its own subplots
    fig, axes = plt.subplots(
        len(metrics_to_plot), 1,
        figsize=((12, len(metrics_to_plot) * 5) if figsize is None else figsize),
        sharex=('col' if compact_display else 'none')
    )
    # TODO if needed: rename models in the dataframe itself
    if len(metrics_to_plot) == 1:
        axes = [axes]  # Add singleton dimension, for compatibility
    for metric_idx, metric_name in enumerate(metrics_to_plot):
        models_melted_results = list()
        for model_idx, interp_results in enumerate(models_interp_results):
            results_df = interp_results[metric_name]
            results_df = results_df / reference_norm_factors[metric_name]
            # https://stackoverflow.com/questions/49554139/boxplot-of-multiple-columns-of-a-pandas-dataframe-on-the-same-figure-seaborn
            melted_results_df = pd.melt(results_df)  # 2-cols DF: 'variable' (feature name) and 'value'
            melted_results_df['model_name'] = models_names[model_idx]
            models_melted_results.append(melted_results_df)
        models_melted_results = pd.concat(models_melted_results)
        # use bright colors (pastel palette) such that the black median line is easily visible
        sns.boxplot(data=models_melted_results, x="variable", y="value", hue="model_name",
                    ax=axes[metric_idx], showfliers=False, linewidth=1.0, palette="pastel")
        axes[metric_idx].set_ylim(ymin=0.0)
        '''
        sns.pointplot(
            data=models_melted_results, x="variable", y="value", hue="model_name", ax=axes[metric_idx],
            errwidth=1.0, marker='.', scale=0.5, ci="sd", dodge=0.4, join=False,  # SD instead of 95% CI
        )
        '''
        axes[metric_idx].set(xlabel='', ylabel=(metric_name.title() + ' (scaled)'))
        axes[metric_idx].tick_params(axis='x', labelrotation=90)
        if compact_display:
            axes[metric_idx].get_legend().set(title=None)
        if legend_ax_idx is not None and metric_idx != legend_ax_idx:
            axes[metric_idx].get_legend().remove()
        # If 2 models only, we may perform the wilcoxon paired test
        if len(models_interp_results) == 2:
            if display_wilcoxon_tests:
                if not compact_display:
                    # p_value < 0.05 if model [1] has significantly LOWER values than model [0] (reference)
                    p_values, has_improved = utils.stat.wilcoxon_test(
                        models_interp_results[0][metric_name], models_interp_results[1][metric_name])
                    p_values, has_deteriorated = utils.stat.wilcoxon_test(
                        models_interp_results[1][metric_name], models_interp_results[0][metric_name])
                    axes[metric_idx].set(
                        title="{} - Wilcoxon test: {}/{} improved, {}/{} deteriorated features"
                        .format(
                            metric_name, np.count_nonzero(has_improved.values), len(has_improved),
                            np.count_nonzero(has_deteriorated.values), len(has_deteriorated)
                        )
                    )
                else:
                    warnings.warn("Can't display wilcoxon test results because compact_display is True")
        else:
            if display_wilcoxon_tests:
                warnings.warn("The Wilcoxon test requires to provide only 2 models (reference and another)")
    fig.tight_layout()

    # TODO grouped boxplots - or maybe edit figures after they have been saved as PDF?

    return fig, axes


def plot_improvements_vs_ref(
        improvements_df: pd.DataFrame,
        measurements_to_plot=('wilcoxon_test___n_improved', 'wilcoxon_test___n_deteriorated',
                              'variation_vs_ref___median', 'variation_vs_ref___mean'),
        hparams: Optional[List[str]] = None,
):
    # 1) Plot numeric general improvement
    cols = list(improvements_df.columns)
    for m in measurements_to_plot:
        cols.remove(m)
    melted_df = pd.melt(improvements_df, id_vars=cols, var_name='var')  # value_vars will be all non-id cols
    g = sns.catplot(
        data=melted_df, y="model___name", x='value', col='var', hue="metric___name",
        kind="point", sharex=False,
    )
    fig1, axes1 = g.fig, g.axes
    for ax in axes1[0]:
        ax.grid(axis='y')
    fig1.set_size_inches(20.0, len(improvements_df) / 7.0)

    # 2) plot improvements vs model/train/interp hparams - scatter plots only
    if hparams is not None and len(hparams) >= 1:
        fig2, axes2 = plt.subplots(
            len(hparams), len(measurements_to_plot),
            figsize=(1 + 3 * len(measurements_to_plot), 1 + 3 * len(hparams)), sharex='col', sharey='row')
        axes2 = np.expand_dims(axes2, axis=0) if len(hparams) == 1 else axes2
        for i, hparam in enumerate(hparams):
            log_scale = hparam.endswith('___LOGSCALE')
            if log_scale:
                hparam = hparam.replace('___LOGSCALE', '')
            for j, measurement_name in enumerate(measurements_to_plot):
                # draw the evolution of means using lines (as done in comet.ml)
                # TODO hparams: we must manually handle the 'empty' case when computing means, for non-numeric
                if np.issubdtype(improvements_df[hparam].values.dtype, np.number):
                    h_params_values = np.unique(improvements_df[hparam].values)  # float equality: works OK with hparams
                else:  # non-numeric: force-concert everything to str
                    h_params_values = np.unique([str(v) for v in improvements_df[hparam].values])  # also does a sort
                mean_per_hparam = [improvements_df.loc[improvements_df[hparam] == v][measurement_name].values.mean()
                                   for v in h_params_values]
                axes2[i, j].plot(mean_per_hparam, h_params_values, color='k')  # 'vertical' plot
                # Then draw the actual scatter plot
                sns.scatterplot(data=improvements_df, x=measurement_name, y=hparam, ax=axes2[i, j],
                                hue='model___name', legend=False)
                if log_scale:
                    axes2[i, j].set(yscale='log')
        fig2.tight_layout()
    else:
        fig2, axes2 = None, None

    return fig1, axes1, fig2, axes2


def plot_interpolation_vs_reconstruction(improvements_df: pd.DataFrame, figsize=(10, 10)):
    """ Plots the reconstruction errors as a function of the interpolation quality measurements.
    Smoothness- and non-linearity-performance will be plotted into two different figures.
    TODO legend in a 3rd figure?   """
    interp_measurements = ['wilcoxon_test___n_improved', 'wilcoxon_test___n_deteriorated',
                           'variation_vs_ref___median', 'variation_vs_ref___mean',]
    interp_measurements_short_names = ['n_improved', 'n_deteriorated', 'variation_median', 'variation_mean']
    recons_measurements = ['reconstruction_error___timbre_mean', 'reconstruction_error___timbre_med',
                           'reconstruction_error___param_acc_mean', 'reconstruction_error___param_acc_med',
                           'reconstruction_error___param_l1err_mean', 'reconstruction_error___param_l1err_med',]
    recons_measurements_short_names = ['timbre_error_mean', 'timbre_error_median',
                                       'param_acc_mean', 'param_acc_median', 'param_l1err_mean', 'param_l1err_median']
    fig1, axes1 = plt.subplots(len(recons_measurements), len(interp_measurements), figsize=figsize,
                               sharex='col', sharey='row')
    fig2, axes2 = plt.subplots(len(recons_measurements), len(interp_measurements), figsize=figsize,
                               sharex='col', sharey='row')
    all_figs, all_axes = (fig1, fig2), (axes1, axes2)
    # TODO build a dict of markers (different marker for each model_name)
    models_names = list(improvements_df[improvements_df['metric___name'] == 'smoothness']['model___name'])
    markers = ['o', '<', 's', 'p', 'P', '*', 'X', 'd']
    i, markers_by_model = 0, {}
    for name in models_names:
        markers_by_model[name] = markers[i]
        i = (i + 1 if i < (len(markers) - 1) else 0)
    # Plot the smoothness and nonlinearity, scatter plots + linear regressions
    for fig_idx, metric in enumerate(['smoothness', 'nonlinearity']):
        df = improvements_df[improvements_df['metric___name'] == metric]
        fig, axes = all_figs[fig_idx], all_axes[fig_idx]
        for col, interp_m in enumerate(interp_measurements):
            for row, recons_m in enumerate(recons_measurements):
                # TODO try scatter with custom markers
                sns.regplot(data=df, x=interp_m, y=recons_m, ax=axes[row, col], scatter=False, color='lightgray')
                axes[row, col].set_xlabel("")  # regplot forces labels on all sub-axes
                axes[row, col].set_ylabel("")
                sns.scatterplot(
                    data=df, x=interp_m, y=recons_m, hue='model___name', style='model___name', markers=markers_by_model,
                    legend=False, ax=axes[row, col])
        # improved axes labels (shorter)
        for col, short_name in enumerate(interp_measurements_short_names):
            axes[-1, col].set_xlabel(short_name)
        for row, short_name in enumerate(recons_measurements_short_names):
            axes[row, 0].set_ylabel(short_name)
        fig.suptitle(metric)
        fig.tight_layout()
        # TODO height proportionnal to the number of models
    dummy_df = improvements_df[improvements_df['metric___name'] == 'smoothness']
    dummy_df.insert(loc=1, column='dummy_x', value=0)
    dummy_df.insert(loc=1, column='dummy_y', value=0)
    #dummy_df.insert(loc=1, column='dummy_y', value=np.arange(0, -len(dummy_df), step=-1))
    fig3, ax3 = plt.subplots(1, 1, figsize=(figsize[0], 0.3 * len(dummy_df)))
    sns.scatterplot(
        data=dummy_df, x='dummy_x', y='dummy_y', hue='model___name', style='model___name', markers=markers_by_model,
        legend=True, ax=ax3)
    ax3.set_axis_off()
    ax3.set(xlabel="", ylabel="")



if __name__ == "__main__":
    # use for debugging only
    from evalconfig import InterpEvalConfig

    if False:  # test BOXPLOT
        _base_path = Path(__file__).resolve().parent.parent.parent.joinpath("Data_SSD/Logs")
        _storage_paths = [
            _base_path.joinpath('RefInterp/LinearNaive/interp9_valid'),
            _base_path.joinpath('preset-vae/presetAE/combined_vae_beta1.60e-04_presetfactor0.20/interp9_valid_uLin_zLin')
        ]
        interp_results_boxplots(_storage_paths, eval_config=InterpEvalConfig(), display_wilcoxon_tests=True)

    _improvements_df, _improvements_df_multiindex = InterpBase.compute_interp_improvement_vs_ref(InterpEvalConfig())

    #plot_improvements_vs_ref(
    #    _improvements_df,
    #    hparams=['model_config___dim_z', 'train_config___beta']  # TODO try others
    #)

    plot_interpolation_vs_reconstruction(improvements_df=_improvements_df)



    plt.show()

