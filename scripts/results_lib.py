import math

import numpy as np
from scipy import stats


def summary_stats(scores):
    scores = list(scores)
    n = len(scores)
    arr = np.array(scores, dtype=float)

    mean = float(arr.mean())
    stdev = float(arr.std(ddof=1))
    sorted_arr = np.sort(arr)
    median = float(np.median(sorted_arr))

    if n > 1:
        excl = np.delete(sorted_arr, -1)
        mean_excl_top_outlier = float(excl.mean())
    else:
        mean_excl_top_outlier = mean

    se = stdev / math.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1) if n > 1 else float('nan')
    ci_low = mean - t_crit * se
    ci_high = mean + t_crit * se

    return {
        'n': n,
        'mean': mean,
        'median': median,
        'mean_excl_top_outlier': mean_excl_top_outlier,
        'stdev': stdev,
        'ci95_low': ci_low,
        'ci95_high': ci_high,
    }


def mannwhitney_pair(a, b):
    a = np.array(list(a), dtype=float)
    b = np.array(list(b), dtype=float)
    result = stats.mannwhitneyu(a, b, alternative='two-sided')
    u_stat = float(result.statistic)
    p_value = float(result.pvalue)

    n1, n2 = len(a), len(b)
    rank_biserial = 1.0 - (2.0 * u_stat) / (n1 * n2)

    return u_stat, p_value, rank_biserial


def bonferroni_threshold(alpha=0.05, n_tests=4):
    return alpha / n_tests
