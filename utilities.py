import numpy as np
import pandas as pd
import logging

def calculate_posterior_mc_rate(mc_da, cov_da, normalize_per_cell=True, clip_norm_value=10):
    raw_rate = mc_da / cov_da
    cell_rate_mean = np.nanmean(raw_rate, axis=1)[:, None]  # this skip na
    cell_rate_var = np.nanvar(raw_rate, axis=1)[:, None]  # this skip na
    # based on beta distribution mean, var
    # a / (a + b) = cell_rate_mean
    # a * b / ((a + b) ^ 2 * (a + b + 1)) = cell_rate_var
    # calculate alpha beta value for each cell
    cell_a = (1 - cell_rate_mean) * (cell_rate_mean ** 2) / cell_rate_var - cell_rate_mean
    cell_b = cell_a * (1 / cell_rate_mean - 1)
    # cell specific posterior rate
    post_rate = (mc_da + cell_a) / (cov_da + cell_a + cell_b)
    if normalize_per_cell:
        # there are two ways of normalizing per cell, by posterior or prior mean:
        # prior_mean = cell_a / (cell_a + cell_b)
        # posterior_mean = post_rate.mean(dim=var_dim)
        # Here I choose to use prior_mean to normalize cell,
        # therefore all cov == 0 features will have normalized rate == 1 in all cells.
        # i.e. 0 cov feature will provide no info
        prior_mean = cell_a / (cell_a + cell_b)
        post_rate = post_rate / prior_mean
        if clip_norm_value is not None:
            post_rate[post_rate > clip_norm_value] = clip_norm_value
    return post_rate

def highly_variable_methylation_feature(X, feature_mean_cov, var_names, min_disp=0.5, max_disp=None, min_mean=0, max_mean=5, n_top_feature=None, bin_min_features=5, mean_binsize=0.05, cov_binsize=100):
    """
    Adapted from Scanpy, see license above
    The main difference is that, this function normalize dispersion based on both mean and cov bins.
    """
    # RNA is only scaled by mean, but methylation is scaled by both mean and cov
    log = logging.getLogger()
    log.info('extracting highly variable features')
    if n_top_feature is not None:
        log.info('If you pass `n_top_feature`, all cutoffs are ignored.')
    # warning for extremely low cov
    low_cov_portion = (feature_mean_cov < 30).sum() / feature_mean_cov.size
    if low_cov_portion > 0.2:
        log.warning(f'{int(low_cov_portion * 100)}% feature with < 10 mean cov, '
                    f'consider filter by cov before find highly variable feature. '
                    f'Otherwise some low coverage feature may be elevated after normalization.')
    cov = feature_mean_cov
    mean = np.mean(X, axis=0)
    var = np.var(X, axis=0, ddof=1)
    # now actually compute the dispersion
    if 0 in mean:
        mean[mean == 0] = 1e-12  # set entries equal to zero to small value
    # raw dispersion is the variance normalized by mean
    dispersion = var / mean
    if 0 in dispersion:
        dispersion[dispersion == 0] = np.nan
    dispersion = np.log(dispersion)
    # all of the following quantities are "per-feature" here
    df = pd.DataFrame(index=var_names)
    df['mean'] = mean
    df['dispersion'] = dispersion
    df['cov'] = cov
    # instead of n_bins, use bin_size, because cov and mc are in different scale
    df['mean_bin'] = (df['mean'] / mean_binsize).astype(int)
    df['cov_bin'] = (df['cov'] / cov_binsize).astype(int)
    # save bin_count df, gather bins with more than bin_min_features features
    bin_count = df.groupby(['mean_bin', 'cov_bin']).apply(lambda i: i.shape[0]).reset_index().sort_values(0, ascending=False)
    bin_count.head()
    bin_more_than = bin_count[bin_count[0] > bin_min_features]
    if bin_more_than.shape[0] == 0:
        raise ValueError(f'No bin have more than {bin_min_features} features, uss larger bin size.')
    # for those bin have too less features, merge them with closest bin in manhattan distance
    # this usually don't cause much difference (a few hundred features), but the scatter plot will look more nature
    index_map = {}
    for _, (mean_id, cov_id, count) in bin_count.iterrows():
        if count > 1:
            index_map[(mean_id, cov_id)] = (mean_id, cov_id)
        manhattan_dist = (bin_more_than['mean_bin'] - mean_id).abs() + (bin_more_than['cov_bin'] - cov_id).abs()
        closest_more_than = manhattan_dist.sort_values().index[0]
        closest = bin_more_than.loc[closest_more_than]
        index_map[(mean_id, cov_id)] = tuple(closest.tolist()[:2])
    # apply index_map to original df
    raw_bin = df[['mean_bin', 'cov_bin']].set_index(['mean_bin', 'cov_bin'])
    raw_bin['use_mean'] = pd.Series(index_map).apply(lambda i: i[0])
    raw_bin['use_cov'] = pd.Series(index_map).apply(lambda i: i[1])
    df['mean_bin'] = raw_bin['use_mean'].values
    df['cov_bin'] = raw_bin['use_cov'].values
    # calculate bin mean and std, now disp_std_bin shouldn't have NAs
    disp_grouped = df.groupby(['mean_bin', 'cov_bin'])['dispersion']
    disp_mean_bin = disp_grouped.mean()
    disp_std_bin = disp_grouped.std(ddof=1)
    # actually do the normalization
    _mean_norm = disp_mean_bin.loc[list(zip(df['mean_bin'], df['cov_bin']))]
    _std_norm = disp_std_bin.loc[list(zip(df['mean_bin'], df['cov_bin']))]
    df['dispersion_norm'] = (df['dispersion'].values - _mean_norm.values) / _std_norm.values
    dispersion_norm = df['dispersion_norm'].values.astype('float32')
    log.info('    finished')
    return dispersion_norm

