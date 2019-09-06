# Authors: Brooke Husic and Nick Charron
# Contributors : Dominik Lemm

import numpy as np

from cgnet.feature import kl_divergence, js_divergence, histogram_intersection


def _get_random_distr():
    # This function produces two random distributions upon which
    # comparisons, overlaps, and divergences can be calculated

    length = np.random.randint(1, 50)  # Range of distribution
    n_zeros = np.random.randint(1, 10)  # Number of bins with zero counts
    zeros = np.zeros(n_zeros)  # corresponding array of zero count bins

    # Here, we create two distributions, and then shuffle the bins
    # so that the zero count bins are distributed randomly along the
    # distribution extent
    distribution_1 = np.abs(np.concatenate([np.random.randn(length), zeros]))
    distribution_2 = np.abs(np.concatenate([np.random.randn(length), zeros]))
    np.random.shuffle(distribution_1)
    np.random.shuffle(distribution_2)
    return distribution_1, distribution_2


def _get_uniform_histograms():
    # This function produces two histograms sampled from uniform
    # distributions, returning the corresponding bins as well
    nbins = np.random.randint(2, high=50)  # Random number of bins
    _bins = np.linspace(0, 1, nbins)  # Equally space bins

    # Here, we produce the two histogram/bin pairs. We explicitly use
    # the same _bins, so we check that bins_1 and bins_2 are both equal
    # to our specified _bins post histogram creation.
    histogram_1, bins_1 = np.histogram(np.random.uniform(size=nbins),
                                       bins=_bins,
                                       density=True)
    histogram_2, bins_2 = np.histogram(np.random.uniform(size=nbins),
                                       bins=_bins,
                                       density=True)

    # We verify that that the two bin arrays are the same as each other and
    # as our input _bins. This is necessary for proper histogram comparison,
    # which is done bin-wise
    np.testing.assert_array_equal(bins_1, bins_2)
    np.testing.assert_array_equal(bins_1, _bins)
    return histogram_1, histogram_2, bins_1


# We use the above two functions to generate random distributions
# and histogram/bin pairs suitable for comparison using CGnet feature tools
dist1, dist2 = _get_random_distr()
hist1, hist2, bins = _get_uniform_histograms()


def test_zero_kl_divergence():
    # Tests the calculation of KL divergence for a random distribution from
    # zeros with itself. The KL divergence of a distribution with itself
    # is equal to zero
    div = kl_divergence(dist1, dist1)
    np.testing.assert_allclose(div, 0.)


def test_kl_divergence():
    # Tests the calculation of KL divergence for two random distributions with
    # zeros using a manual calculation
    manual_div = 0.  # accumulator for the KL divergence

    # Loop through the bins of distribution 1 and accumulate the KL divergence
    for i, entry in enumerate(dist1):
        if dist1[i] > 0 and dist2[i] > 0:
            manual_div += entry * np.log(entry / dist2[i])

    # Here we verify that the manual calculation above matches the same produced
    # by the kl_divergence function
    cgnet_div = kl_divergence(dist1, dist2)
    np.testing.assert_allclose(manual_div, cgnet_div)


def test_zero_js_divergence():
    # Tests the calculation of JS divergence for a random distribution from
    # zeros with itself. The JS divergence of a distribution with itself is
    # equal to zero
    div = js_divergence(dist1, dist1)
    np.testing.assert_allclose(div, 0.)


def test_js_divergence():
    # Tests the calculation of JS divergence for two random distributions with
    # zeros using a manual calculation

    # Here, we mask those mutual bins where the count multiplciation
    # is null, as these do not contribute to the JS divergence
    dist1_masked = np.ma.masked_where(dist1 * dist2 == 0, dist1)
    dist2_masked = np.ma.masked_where(dist1 * dist2 == 0, dist2)

    # Here we produce the elementwise mean of the masked distributions
    # for calculating the JS divergence
    elementwise_mean = 0.5 * (dist1_masked + dist2_masked)

    manual_div_1 = 0.  # accumulator for the first divergence
    # Here, we loop through the bins of the first distribution and calculate
    # the divergence
    for i, entry in enumerate(dist1):
        if dist1[i] > 0 and elementwise_mean[i] > 0:
            manual_div_1 += entry * np.log(entry / elementwise_mean[i])

    manual_div_2 = 0.  # accumulator for the second divergence
    # Here, we loop through the bins of the second distribution and calculate
    # the divergence
    for i, entry in enumerate(dist2):
        if dist2[i] > 0 and elementwise_mean[i] > 0:
            manual_div_2 += entry * np.log(entry / elementwise_mean[i])

    # Manual calculation of the JS divergence
    manual_div = np.mean([manual_div_1, manual_div_2])

    # Here, we verify that the manual calculation matches the
    # output of the js_divergence function
    cgnet_div = js_divergence(dist1, dist2)
    np.testing.assert_allclose(manual_div, cgnet_div)


def test_js_divergence_2():
    # Tests the calculation of JS divergence for two random distributions with
    # zeros using masked arrays

    # This is the same test as test_js_divergence_1, just done using numpy
    # operations rather than loops
    dist1_masked = np.ma.masked_where(dist1 == 0, dist1)
    dist2_masked = np.ma.masked_where(dist2 == 0, dist2)
    elementwise_mean = 0.5 * (dist1_masked + dist2_masked)
    summand = 0.5 * (dist1_masked * np.ma.log(dist1_masked / elementwise_mean))
    summand += 0.5 * (dist2_masked * np.ma.log(dist2_masked / elementwise_mean))
    manual_div = np.ma.sum(summand)

    cgnet_div = js_divergence(dist1, dist2)
    np.testing.assert_allclose(manual_div, cgnet_div)


def test_full_histogram_intersection():
    # Tests the intersection of a uniform histogram with itself
    # The intersection of any histogram with itself should be unity

    cgnet_intersection = histogram_intersection(hist1, hist1, bins)
    np.testing.assert_allclose(cgnet_intersection, 1.)
    np.testing.assert_allclose(cgnet_intersection, 1.)


def test_histogram_intersection():
    # Tests the calculation of intersection for histograms drawn from
    # uniform distributions

    manual_intersection = 0.  # intersection accumulator
    intervals = np.diff(bins)  # intervals betweem histogram bins

    # Here we loop though the common histogram intervals and accumulate
    # the intersection of the two histograms
    for i in range(len(intervals)):
        manual_intersection += min(intervals[i] * hist1[i],
                                   intervals[i] * hist2[i])

    # Here we verify that the manual calculation matches the output of
    # the historam_intersection function
    cgnet_intersection = histogram_intersection(hist1, hist2, bins)
    np.testing.assert_allclose(manual_intersection, cgnet_intersection)


def test_histogram_intersection_no_bins():
    # Tests the calculation of intersection for histograms drawn from
    # uniform distributions. The histogram intersection should fill in
    # the bins uniformly if none are supplied and norm=True (default).

    cgnet_intersection = histogram_intersection(hist1, hist2, bins)
    nobins_intersection = histogram_intersection(hist1, hist2, bin_edges=None,
                                                 norm=True)
    np.testing.assert_allclose(cgnet_intersection, nobins_intersection)


def test_histogram_intersection_norm():
    # Tests the functionality of the norm=True option for histogram
    # intersection calculations when bins do not have a range of 1.

    # Create bins that are spaced by length 1 and offset by an arbitrary
    # constant
    constant = np.random.randint(2, 10)
    bins_no_norm = np.linspace(constant, len(hist1)+constant, len(hist1) + 1)

    norm_intersection = histogram_intersection(hist1, hist2,
                                               bins_no_norm, norm=True)
    cgnet_intersection = histogram_intersection(hist1, hist2, bins)

    np.testing.assert_allclose(norm_intersection, cgnet_intersection)


def test_histogram_intersection_norm_2():
    # Tests norm functionality for computing histogram intersections
    # of uniform distributions multipled by an arbitrary factor with manually
    # specified bins
    factor = np.random.randint(2, 6)
    hist1_mult = factor*hist1
    hist2_mult = factor*hist2

    # Create bins that are spaced by length 1 and offset by an arbitrary
    # constant
    constant = np.random.randint(2, 10)
    bins_no_norm = np.linspace(constant, len(hist1)+constant, len(hist1) + 1)

    intersection_mult = histogram_intersection(hist1_mult, hist2_mult,
                                               bins_no_norm)
    intersection = histogram_intersection(hist1, hist2, bin_edges=None,
                                          norm=True)

    np.testing.assert_allclose(intersection_mult, factor*intersection)


def test_histogram_intersection_norm_3():
    # Test whether the intersection of a histogram with itself is 
    # the length of the histogram when no norm is used on the bins
    self_no_norm = histogram_intersection(hist1, hist1, norm=False)
    np.testing.assert_allclose(self_no_norm, len(hist1))
