# Authors: Brooke Husic and Nick Charron

import numpy as np

from cgnet.feature import kl_divergence, js_divergence, histogram_intersection


def _get_random_distr():
    length = np.random.randint(0, 50)
    n_zeros = np.random.randint(0, 10)
    zeros = np.zeros(n_zeros)
    dist1 = np.abs(np.concatenate([np.random.randn(length), zeros]))
    dist2 = np.abs(np.concatenate([np.random.randn(length), zeros]))
    np.random.shuffle(dist1)
    np.random.shuffle(dist2)
    return dist1, dist2


def _get_uniform_histograms():
    nbins = np.random.randint(2, high=50)
    bins_ = np.linspace(0, 1, nbins)
    hist1, bins1 = np.histogram(np.random.uniform(size=nbins), bins=bins_,
                                density=True)
    hist2, bins2 = np.histogram(np.random.uniform(size=nbins), bins=bins_,
                                density=True)
    np.testing.assert_array_equal(bins1, bins2)
    return hist1, hist2, bins_, bins1


dist1, dist2 = _get_random_distr()
hist1, hist2, bins_, bins = _get_uniform_histograms()


def test_zero_kl_divergence():
    # Tests the calculation of KL divergence for a random distribution from
    # zeros with itself
    div = kl_divergence(dist1, dist1)
    np.testing.assert_allclose(div, 0.)


def test_kl_divergence():
    # Tests the calculation of KL divergence for two random distributions with
    # zeros using a manual calculation
    manual_div = 0.
    for i, entry in enumerate(dist1):
        if dist1[i] > 0 and dist2[i] > 0:
            manual_div += entry * np.log(entry / dist2[i])

    cgnet_div = kl_divergence(dist1, dist2)
    np.testing.assert_allclose(manual_div, cgnet_div)


def test_zero_js_divergence():
    # Tests the calculation of JS divergence for a random distribution from
    # zeros with itself
    div = js_divergence(dist1, dist1)
    np.testing.assert_allclose(div, 0.)


def test_js_divergence():
        # Tests the calculation of JS divergence for two random distributions with
        # zeros using a manual calculation
    dist1m = np.ma.masked_where(dist1 * dist2 == 0, dist1)
    dist2m = np.ma.masked_where(dist1 * dist2 == 0, dist2)
    elementwise_mean = 0.5 * (dist1m + dist2m)
    manual_div_1 = 0.
    for i, entry in enumerate(dist1):
        if dist1[i] > 0 and elementwise_mean[i] > 0:
            manual_div_1 += entry * np.log(entry / elementwise_mean[i])
    manual_div_2 = 0.
    for i, entry in enumerate(dist2):
        if dist2[i] > 0 and elementwise_mean[i] > 0:
            manual_div_2 += entry * np.log(entry / elementwise_mean[i])
    manual_div = np.mean([manual_div_1, manual_div_2])

    cgnet_div = js_divergence(dist1, dist2)
    np.testing.assert_allclose(manual_div, cgnet_div)


def test_js_divergence_2():
    # Tests the calculation of JS divergence for two random distributions with
    # zeros using masked arrays
    dist1m = np.ma.masked_where(dist1 == 0, dist1)
    dist2m = np.ma.masked_where(dist2 == 0, dist2)
    elementwise_mean = 0.5 * (dist1m + dist2m)
    summand = 0.5 * (dist1m * np.ma.log(dist1m/elementwise_mean))
    summand += 0.5 * (dist2m * np.ma.log(dist2m/elementwise_mean))
    manual_div = np.ma.sum(summand)

    cgnet_div = js_divergence(dist1, dist2)
    np.testing.assert_allclose(manual_div, cgnet_div)


def test_full_histogram_intersection():
    # Tests the intersection of a uniform histogram with itself
    cgnet_intersection = histogram_intersection(hist1, hist1, bins)
    np.testing.assert_allclose(cgnet_intersection, 1.)
    np.testing.assert_allclose(cgnet_intersection, 1.)


def test_histogram_intersection():
    # Tests the calculation of intersection for histograms drawn from
    # uniform distributions
    manual_intersection = 0.
    intervals = np.diff(bins_)
    for i in range(len(intervals)):
        manual_intersection += min(intervals[i] * hist1[i],
                                   intervals[i] * hist2[i])

    cgnet_intersection = histogram_intersection(hist1, hist2, bins_)
    np.testing.assert_allclose(manual_intersection, cgnet_intersection)


def test_histogram_intersection_no_bins():
    # Tests the calculation of intersection for histograms drawn from
    # uniform distributions
    cgnet_intersection = histogram_intersection(hist1, hist2, bins_)
    nobins_intersection = histogram_intersection(hist1, hist2, bins=None)
    np.testing.assert_allclose(cgnet_intersection, nobins_intersection)
