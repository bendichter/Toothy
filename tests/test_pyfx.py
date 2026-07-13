"""Tests for pyfx: general numeric helpers, signal filtering, and color conversion."""
import numpy as np
import pytest

import pyfx


###   GENERAL FUNCTIONS   ###

@pytest.mark.parametrize('x, expected', [
    (3, True), (3.5, True), ('3.5', True), ('-2e3', True),
    ('abc', False), (None, False), ([1, 2], False),
])
def test_isnum(x, expected):
    assert pyfx.IsNum(x) is expected


def test_downsample_averages_bins():
    assert pyfx.Downsample(np.arange(10), 2).tolist() == [0.5, 2.5, 4.5, 6.5, 8.5]


def test_downsample_truncates_trailing_partial_bin():
    # 11 samples binned by 2 -> the 11th sample is dropped, not zero-padded
    assert pyfx.Downsample(np.arange(11), 2).tolist() == [0.5, 2.5, 4.5, 6.5, 8.5]


def test_normalize_scales_to_unit_range():
    assert pyfx.Normalize([0, 5, 10]).tolist() == [0.0, 0.5, 1.0]


def test_normalize_constant_input_returns_zeros():
    # guards against a divide-by-zero on flat/dead channels
    assert pyfx.Normalize([4, 4, 4]).tolist() == [0.0, 0.0, 0.0]


def test_normalize_ignores_nans():
    res = pyfx.Normalize([np.nan, 0.0, 10.0])
    assert np.isnan(res[0])
    assert res[1:].tolist() == [0.0, 1.0]


def test_closest_and_idxclosest():
    collection = [1.0, 2.0, 3.0, 4.0]
    assert pyfx.Closest(3.2, collection) == 3.0
    assert pyfx.IdxClosest(3.2, collection) == 2


def test_center_and_idxcenter():
    assert pyfx.Center([10, 20, 30, 40, 50]) == 30
    assert pyfx.IdxCenter([10, 20, 30, 40, 50]) == 2


def test_edges_returns_first_and_last():
    assert pyfx.Edges([7, 8, 9]) == (7, 9)


def test_minmax_excludes_nans():
    assert pyfx.MinMax([np.nan, -2.0, 5.0]) == (-2.0, 5.0)


def test_limit_pads_data_range():
    # range is 10 wide, 10% padding -> 1 unit beyond each end
    assert pyfx.Limit([0, 10], mode=2, pad=0.1) == (-1.0, 11.0)
    assert pyfx.Limit([0, 10], mode=0, pad=0.1) == -1.0
    assert pyfx.Limit([0, 10], mode=1, pad=0.1) == 11.0


def test_limit_negative_sign_pads_inward():
    assert pyfx.Limit([0, 10], mode=2, pad=0.1, sign=-1) == (1.0, 9.0)


def test_limit_empty_collection_returns_none():
    assert pyfx.Limit([]) == (None, None)


def test_symlimit_is_symmetric_about_zero():
    assert pyfx.SymLimit([-3.0, 2.0]) == (-3.0, 3.0)


def test_inrange_is_inclusive_of_bounds():
    assert pyfx.InRange(5, 0, 10) is True
    assert pyfx.InRange(0, 0, 10) is True
    assert pyfx.InRange(11, 0, 10) is False


def test_allinrange():
    assert pyfx.AllInRange([1, 5, 9], 0, 10) is True
    assert pyfx.AllInRange([1, 5, 11], 0, 10) is False


def test_centerwin_even_window():
    ii, vals = pyfx.CenterWin(np.arange(10), 4, total=True)
    assert ii.tolist() == [3, 4, 5, 6]
    assert vals.tolist() == [3, 4, 5, 6]


def test_centerwin_odd_window_keeps_requested_length():
    ii, vals = pyfx.CenterWin(np.arange(10), 5, total=True)
    assert len(ii) == 5
    assert ii.tolist() == [3, 4, 5, 6, 7]


def test_get_sequences_splits_on_breaks():
    seqs = pyfx.get_sequences(np.array([1, 2, 3, 7, 8, 15]))
    assert [s.tolist() for s in seqs] == [[1, 2, 3], [7, 8], [15]]


def test_get_sequences_single_run():
    seqs = pyfx.get_sequences(np.array([4, 5, 6]))
    assert [s.tolist() for s in seqs] == [[4, 5, 6]]


def test_laser_start_end_finds_stim_trains():
    laser = np.zeros(20000)
    laser[1000:2000] = 1
    laser[10000:11000] = 1
    istart, iend = pyfx.laser_start_end(laser, fs=1000., intval=5)
    assert istart.tolist() == [1000, 10000]
    assert iend.tolist() == [1999, 10999]


def test_laser_start_end_no_stim_returns_empty():
    assert pyfx.laser_start_end(np.zeros(100)) == ([], [])


###   SIGNAL PROCESSING   ###

def test_butter_bandpass_returns_sos_coefficients():
    sos = pyfx.butter_bandpass(80, 120, 1000., order=3)
    assert sos.shape == (3, 6)


def test_butter_bandpass_filter_isolates_passband():
    fs = 1000.
    t = np.arange(0, 2, 1 / fs)
    target = np.sin(2 * np.pi * 100 * t)   # in the 80-120 Hz passband
    noise = np.sin(2 * np.pi * 5 * t)      # well below the passband
    filtered = pyfx.butter_bandpass_filter(target + noise, 80, 120, fs)

    # ignore filter edge effects
    mid = slice(200, -200)
    assert np.corrcoef(filtered[mid], target[mid])[0, 1] > 0.99
    assert np.corrcoef(filtered[mid], noise[mid])[0, 1] < 0.1


def test_butter_bandpass_filter_preserves_shape_on_2d_input():
    data = np.random.default_rng(0).normal(size=(4, 1000))
    filtered = pyfx.butter_bandpass_filter(data, 80, 120, 1000., axis=-1)
    assert filtered.shape == data.shape


###   COLOR CONVERSION   ###

@pytest.mark.parametrize('color', ['red', '#ff0000', (255, 0, 0), [1.0, 0.0, 0.0]])
def test_color2rgba_accepts_multiple_formats(color):
    assert np.allclose(pyfx.color2rgba(color), [1.0, 0.0, 0.0, 1.0])


def test_color2rgba_parses_qt_stylesheet_string():
    assert np.allclose(pyfx.color2rgba('rgb(255, 0, 0)'), [1.0, 0.0, 0.0, 1.0])


@pytest.mark.parametrize('bad', ['not-a-color', (1, 2), None, (0, 0, 0, 0, 0)])
def test_color2rgba_rejects_invalid_input(bad):
    assert pyfx.color2rgba(bad) is None


def test_rgb1_and_rgb255():
    assert pyfx.rgb1('red') == (1.0, 0.0, 0.0)
    assert pyfx.rgb1('red', alpha=True) == (1.0, 0.0, 0.0, 1.0)
    assert pyfx.rgb255('red') == (255, 0, 0)


def test_qstyle_rgb_formats_for_stylesheets():
    assert pyfx.qstyle_rgb('red') == 'rgb(255, 0, 0)'
    assert pyfx.qstyle_rgb('red', alpha=True) == 'rgba(255, 0, 0, 255)'


def test_get_hex():
    assert pyfx.get_hex((255, 0, 0)) == '#ff0000'


def test_alpha_like_blends_color_toward_background():
    # 50% black on white is mid-grey
    assert np.allclose(pyfx.alpha_like('black', alpha=0.5, bg='white'), (0.5, 0.5, 0.5))


def test_hue_tints_and_shades():
    # mode=1 lightens toward white, mode=0 darkens toward black
    assert np.allclose(pyfx.hue('black', 0.5, mode=1, cscale=1), (0.5, 0.5, 0.5, 1))
    assert np.allclose(pyfx.hue('white', 0.5, mode=0, cscale=1), (0.5, 0.5, 0.5, 1))


def test_cmap_returns_rgb_or_rgba():
    rgb = pyfx.Cmap(np.array([0.0, 1.0, 2.0]))
    assert rgb.shape == (3, 3)

    rgba = pyfx.Cmap(np.array([0.0, 1.0, 2.0]), alpha=0.4, use_alpha=True)
    assert rgba.shape == (3, 4)
    assert np.allclose(rgba[:, 3], 0.4)


def test_cmap_alpha_appends_alpha_column():
    cmap = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert np.allclose(pyfx.Cmap_alpha(cmap, 0.25)[:, 3], 0.25)


def test_rand_hex_returns_n_valid_hex_codes():
    colors = pyfx.rand_hex(3)
    assert len(colors) == 3
    assert all(c.startswith('#') and len(c) == 7 for c in colors)


###   MISC   ###

def test_dict2ss_builds_stylesheet_string():
    ss = pyfx.dict2ss({'QLabel': {'color': 'red'}})
    assert ss == 'QLabel {color:red;}'


def test_unique_fname_appends_counter_for_existing_names(tmp_path):
    assert pyfx.unique_fname(tmp_path, 'rec') == 'rec'

    (tmp_path / 'rec').touch()
    assert pyfx.unique_fname(tmp_path, 'rec') == 'rec (1)'

    (tmp_path / 'rec (1)').touch()
    assert pyfx.unique_fname(tmp_path, 'rec') == 'rec (2)'
