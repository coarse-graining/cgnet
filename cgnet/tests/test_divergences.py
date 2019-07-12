# Authors: Brooke Husic and Nick Charron

import numpy as np

from cgnet.feature import kl_divergence
from cgnet.feature import js_divergence
from cgnet.feature import compute_intersection


def _get_random_distr():
    length = np.random.randint(5, 50)
    n_zeros = np.random.randint(1, 10)
    zeros = np.zeros(n_zeros)
    dist1 = np.abs(np.concatenate([np.random.randn(length), zeros]))
    dist2 = np.abs(np.concatenate([np.random.randn(length), zeros]))
    np.random.shuffle(dist1)
    np.random.shuffle(dist2)
    dist1 /= np.sum(dist1)
    dist2 /= np.sum(dist2)

    return dist1, dist2


dist1, dist2 = _get_random_distr()


def test_zero_kl_divergence():
    # Tests the calculation of KL divergence for histograms drawn from
    # uniform distributions
    div = kl_divergence(dist1, dist1)
    assert div == 0


def test_kl_divergence():
    manual_div = 0.
    for i, entry in enumerate(dist1):
        if dist1[i] > 0 and dist2[i] > 0:
            manual_div += entry * np.log(entry / dist2[i])

    cgnet_div = kl_divergence(dist1, dist2)
    np.testing.assert_allclose(manual_div, cgnet_div)

# def test_js_divergence():
#     # Tests the calculation of SL divergence for histograms drawn from
#     # uniform distributions
#     nbins = np.random.randint(0, high=50)
#     bins = np.linspace(0, 1, nbins)
#     hist1, bins = np.histogram(np.random.uniform(size=nbins), bins=bins,
#                                density=True)
#     hist2, bins = np.histogram(np.random.uniform(size=nbins), bins=bins,
#                                density=True)

#     div = js_divergence(hist1, hist1)
#     np.testing.assert_allclose(0, div)

#     hist1 = np.ma.masked_where(hist1 == 0, hist1)
#     hist2 = np.ma.masked_where(hist2 == 0, hist2)
#     mix = 0.5 * (hist1 + hist2)
#     summand = 0.5 * (hist1 * np.ma.log(hist1/mix))
#     summand += 0.5 * (hist2 * np.ma.log(hist2/mix))
#     div_0 = np.ma.sum(summand)
#     div = js_divergence(hist1, hist2)
#     assert div_0 == div

# def test_compute_intersection():
#     # Tests the calculation of intersection for histograms drawn from
#     # uniform distributions
#     nbins = np.random.randint(0, high=50)
#     bins = np.linspace(0, 1, nbins)
#     hist1, bins = np.histogram(np.random.uniform(size=nbins), bins=bins,
#                                density=True)
#     hist2, bins = np.histogram(np.random.uniform(size=nbins), bins=bins,
#                                density=True)

#     inter = compute_intersection(hist1, hist1, bins)
#     np.testing.assert_allclose(1, inter)

#     inter_0 = 0.00
#     intervals = np.diff(bins)
#     for i in range(len(intervals)):
#         inter_0 += min(intervals[i] * hist1[i], intervals[i] * hist2[i])
#     inter = compute_intersection(hist1, hist2, bins)
#     assert inter_0 == inter
