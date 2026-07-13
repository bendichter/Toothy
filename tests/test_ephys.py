"""Tests for ephys: event detection, channel estimation, event dataframes, and file I/O."""
import numpy as np
import pandas as pd
import pytest

import ephys
from conftest import LFP_FS, gaussian_bump


###   EVENT DETECTION: DENTATE SPIKES   ###

DS_KW = dict(ds_height_thr=1.0, ds_abs_thr=0.5, ds_dist_thr=100.0, ds_wlen=125.0)


def test_get_ds_peaks_finds_known_spikes(ds_lfp, lfp_time):
    df, thresholds = ephys.get_ds_peaks(ds_lfp, lfp_time, LFP_FS, pprint=False, **DS_KW)

    assert len(df) == 3
    assert np.allclose(df['time'].values, [1.0, 4.0, 7.0], atol=0.005)
    assert np.allclose(df['amp'].values, 1.0, atol=0.01)


def test_get_ds_peaks_returns_thresholds_used(ds_lfp, lfp_time):
    _, thresholds = ephys.get_ds_peaks(ds_lfp, lfp_time, LFP_FS, pprint=False, **DS_KW)

    assert set(thresholds.index) == {'peak_height', 'isi', 'min_amp'}
    assert thresholds['min_amp'] == 0.5
    assert thresholds['isi'] == 0.1  # 100 ms -> s
    assert thresholds['peak_height'] == pytest.approx(np.std(ds_lfp) * 1.0)


def test_get_ds_peaks_absolute_threshold_rejects_small_events(ds_lfp, lfp_time):
    # add a 0.1 mV blip, well under the 0.5 mV ds_abs_thr
    lfp = ds_lfp + gaussian_bump(lfp_time, 9.0, amp=0.1)
    df, _ = ephys.get_ds_peaks(lfp, lfp_time, LFP_FS, pprint=False, **DS_KW)

    assert len(df) == 3
    assert 9.0 not in np.round(df['time'].values)


def test_get_ds_peaks_distance_threshold_merges_close_events(lfp_time):
    # two spikes 50 ms apart, with a 100 ms minimum separation -> only one survives
    lfp = gaussian_bump(lfp_time, 3.0, amp=1.0) + gaussian_bump(lfp_time, 3.05, amp=0.8)
    df, _ = ephys.get_ds_peaks(lfp, lfp_time, LFP_FS, pprint=False, **DS_KW)

    assert len(df) == 1
    assert df['time'].iloc[0] == pytest.approx(3.0, abs=0.005)


def test_get_ds_peaks_no_events_returns_empty_df(lfp_time):
    flat = np.zeros(len(lfp_time))
    df, _ = ephys.get_ds_peaks(flat, lfp_time, LFP_FS, pprint=False, **DS_KW)
    assert len(df) == 0


def test_get_ds_peaks_reports_expected_columns(ds_lfp, lfp_time):
    df, _ = ephys.get_ds_peaks(ds_lfp, lfp_time, LFP_FS, pprint=False, **DS_KW)
    expected = {'time', 'amp', 'half_width', 'width_height', 'asym', 'prom',
                'start', 'stop', 'idx', 'idx_peak', 'idx_start', 'idx_stop'}
    assert expected.issubset(df.columns)

    # events are ordered in time and bounded by their own start/stop
    assert df['time'].is_monotonic_increasing
    assert (df['start'] < df['time']).all()
    assert (df['stop'] > df['time']).all()


def test_get_ds_peaks_uses_raw_lfp_for_peak_index(ds_lfp, lfp_time):
    # when LFPraw is supplied, peak idx comes from the raw trace within +/-10 samples
    df, _ = ephys.get_ds_peaks(ds_lfp, lfp_time, LFP_FS, pprint=False,
                               LFPraw=ds_lfp, **DS_KW)
    assert np.all(np.abs(df['idx'].values - df['idx_peak'].values) <= 10)


###   EVENT DETECTION: SHARP-WAVE RIPPLES   ###

SWR_KW = dict(swr_freq=[120, 180], swr_height_thr=2.0, swr_min_thr=1.0,
              swr_dist_thr=100.0, swr_min_dur=20.0, swr_freq_thr=100.0,
              swr_freq_win=8.0, swr_maxamp_win=40.0)


def test_get_swr_peaks_finds_known_ripples(swr_lfp):
    t, lfp = swr_lfp
    df, _ = ephys.get_swr_peaks(lfp, t, LFP_FS, pprint=False, **SWR_KW)

    assert len(df) == 3
    assert np.allclose(df['time'].values, [1.5, 2.5, 3.5], atol=0.03)
    # detected instantaneous frequency should land near the 150 Hz carrier
    assert np.allclose(df['freq'].values, 150.0, atol=15.0)


def test_get_swr_peaks_returns_thresholds_used(swr_lfp):
    t, lfp = swr_lfp
    _, thresholds = ephys.get_swr_peaks(lfp, t, LFP_FS, pprint=False, **SWR_KW)

    assert set(thresholds.index) == {'dur', 'inst_freq', 'peak_height', 'edge_height', 'isi'}
    assert thresholds['dur'] == 0.02        # 20 ms -> s
    assert thresholds['inst_freq'] == 100.0
    assert thresholds['isi'] == 0.1


def test_get_swr_peaks_no_events_returns_empty_df():
    t = np.arange(0, 5, 1 / LFP_FS)
    df, _ = ephys.get_swr_peaks(np.zeros(len(t)), t, LFP_FS, pprint=False, **SWR_KW)
    assert len(df) == 0


def test_get_swr_peaks_events_bounded_by_start_and_stop(swr_lfp):
    t, lfp = swr_lfp
    df, _ = ephys.get_swr_peaks(lfp, t, LFP_FS, pprint=False, **SWR_KW)
    assert (df['start'] <= df['time']).all()
    assert (df['stop'] >= df['time']).all()
    assert (df['dur'] > 0).all()


def test_get_inst_freq_recovers_carrier_frequency():
    import scipy.signal
    t = np.arange(0, 1, 1 / LFP_FS)
    hilb = scipy.signal.hilbert(np.sin(2 * np.pi * 150 * t))
    ifreq = ephys.get_inst_freq(hilb, LFP_FS, swr_freq=[120, 180])

    # ignore Hilbert edge artifacts
    assert np.median(ifreq[100:-100]) == pytest.approx(150.0, abs=1.0)


def test_get_inst_freq_clips_to_ripple_band():
    import scipy.signal
    t = np.arange(0, 1, 1 / LFP_FS)
    hilb = scipy.signal.hilbert(np.sin(2 * np.pi * 5 * t))  # far below the band
    ifreq = ephys.get_inst_freq(hilb, LFP_FS, swr_freq=[120, 180])
    assert np.all((ifreq >= 120) & (ifreq <= 180))


###   WAVEFORM SHAPE   ###

def test_get_asym_zero_for_symmetric_waveform():
    assert ephys.get_asym(ipk=10, istart=5, istop=15) == 0.0


def test_get_asym_positive_when_peak_is_early():
    # peak sits 5 samples after start, 10 before stop -> right side is longer
    assert ephys.get_asym(ipk=10, istart=5, istop=20) == 100.0


def test_get_asym_negative_when_peak_is_late():
    assert ephys.get_asym(ipk=15, istart=5, istop=20) == -100.0


###   CHANNEL ESTIMATION   ###

def test_estimate_theta_chan_picks_max_theta_power(std_df):
    assert ephys.estimate_theta_chan(std_df) == 0


def test_estimate_theta_chan_skips_noise_channels(std_df):
    assert ephys.estimate_theta_chan(std_df, noise_idx=np.array([0])) == 1


def test_estimate_theta_chan_all_noise_returns_zero(std_df):
    assert ephys.estimate_theta_chan(std_df, noise_idx=np.arange(8)) == 0


def test_estimate_ripple_chan_prefers_high_ripple_low_theta(std_df):
    # ch 5 has strong ripple power with modest theta; ch 0-3 are high-theta (fissure)
    assert ephys.estimate_ripple_chan(std_df) == 5


def test_estimate_ripple_chan_skips_noise_channels(std_df):
    assert ephys.estimate_ripple_chan(std_df, noise_idx=np.array([5])) != 5


def test_estimate_hil_chan_weighs_amplitude_and_event_count():
    # ch 2 has the largest spikes but ch 1 has many more of them
    ds_mean = pd.DataFrame({'amp': [0.1, 0.5, 1.0, 0.2],
                            'n_valid': [1, 10, 2, 0]})
    assert ephys.estimate_hil_chan(ds_mean) == 1


###   EVENT DATAFRAMES   ###

def test_clean_event_df_adds_channel_shank_and_status(event_df, std_df):
    probe = ephys.demo_probe()
    df = ephys.clean_event_df(event_df.copy(), std_df, probe)

    assert df['ch'].tolist() == [0, 0, 3]
    assert df['shank'].tolist() == [0, 0, 0]      # demo probe is a single shank
    assert df['status'].tolist() == [1, 1, 1]     # all events start as "original"
    assert df['is_valid'].tolist() == [1, 1, 1]
    # per-channel band power is joined onto every event
    assert set(std_df.columns).issubset(df.columns)
    assert df['n_valid'].tolist() == [2, 2, 1]    # ch 0 has 2 events, ch 3 has 1


def test_clean_event_df_preserves_existing_status(event_df, std_df):
    probe = ephys.demo_probe()
    edf = event_df.copy()
    edf['status'] = [1, 0, 2]   # 0 = user-removed, 2 = user-added
    df = ephys.clean_event_df(edf, std_df, probe)

    assert df['is_valid'].tolist() == [1, 0, 1]
    assert df['n_valid'].tolist() == [1, 1, 1]   # ch 0 now has only 1 valid event


def test_get_mean_event_df_averages_by_channel(event_df, std_df):
    probe = ephys.demo_probe()
    df_all = ephys.clean_event_df(event_df.copy(), std_df, probe)
    df_mean = ephys.get_mean_event_df(df_all, std_df)

    # one row per probe channel, even for channels with no events
    assert len(df_mean) == len(std_df)
    assert df_mean['ch'].tolist() == list(range(8))
    assert df_mean['n_valid'].tolist() == [2, 0, 0, 1, 0, 0, 0, 0]

    assert df_mean.loc[0, 'amp'] == pytest.approx(2.0)   # mean of 1.0 and 3.0
    assert df_mean.loc[3, 'amp'] == pytest.approx(5.0)
    assert np.isnan(df_mean.loc[1, 'amp'])               # no events on ch 1


def test_replace_missing_channels_fills_gaps_with_nan():
    df = pd.DataFrame({'amp': [1.0, 2.0], 'n_valid': [3, 4]}, index=[0, 2])
    filled = ephys.replace_missing_channels(df, np.arange(4))

    assert filled.index.tolist() == [0, 1, 2, 3]
    assert filled.loc[[0, 2], 'amp'].tolist() == [1.0, 2.0]
    assert np.isnan(filled.loc[[1, 3], 'amp']).all()
    # event counts for missing channels are 0, not NaN
    assert filled.loc[[1, 3], 'n_valid'].tolist() == [0.0, 0.0]


def test_replace_missing_channels_noop_when_complete():
    df = pd.DataFrame({'amp': [1.0, 2.0], 'n_valid': [3, 4]}, index=[0, 1])
    assert ephys.replace_missing_channels(df, np.arange(2)) is df


###   HDF5 KEYS   ###

@pytest.mark.parametrize('kwargs, expected', [
    ({}, '/LFP'),
    (dict(iprb=0), '/0/LFP'),
    (dict(iprb=1, ishank=2), '/1/2/LFP'),
])
def test_get_h5_key(kwargs, expected):
    assert ephys.get_h5_key('LFP', **kwargs) == expected


###   PROBES   ###

def test_demo_probe_shape():
    probe = ephys.demo_probe()
    assert probe.get_contact_count() == 8
    assert probe.shank_ids.astype(int).tolist() == [0] * 8


def test_probe_file_roundtrip(tmp_path):
    probe = ephys.demo_probe()
    fpath = tmp_path / 'probe.json'
    ephys.write_probe_file(probe, fpath)

    loaded = ephys.read_probe_file(fpath, raise_exception=True)
    assert loaded.get_contact_count() == probe.get_contact_count()
    assert np.allclose(loaded.contact_positions, probe.contact_positions)


def test_copy_probe_is_independent():
    probe = ephys.demo_probe()
    clone = ephys.copy_probe(probe)
    assert clone is not probe
    assert np.allclose(clone.contact_positions, probe.contact_positions)


###   WHEEL / POSITION   ###

def test_encoder2pos_direction_reverses_when_channels_swap():
    t = np.arange(64)
    chA = ((t // 4) % 2).astype('float64')
    chB = (((t - 2) // 4) % 2).astype('float64')   # 90 degrees out of phase

    forward = ephys.encoder2pos(chA, chB)
    backward = ephys.encoder2pos(chB, chA)

    assert len(forward) == len(chA)
    assert np.sum(forward) != 0
    assert np.sum(forward) == pytest.approx(-np.sum(backward))


def test_pos2speed_preserves_length_and_smooths():
    pos = np.cumsum(np.ones(1000))   # constant forward motion
    speed = ephys.pos2speed(pos, sf=10)
    assert len(speed) == len(pos)
    assert np.all(speed >= 0)


###   NOTES / BASE DIRECTORY FILE I/O   ###

def test_notes_roundtrip(tmp_path):
    fpath = tmp_path / 'notes.txt'
    ephys.write_notes(fpath, 'clean recording; ch 4 noisy')
    assert ephys.read_notes(fpath) == 'clean recording; ch 4 noisy'


def test_read_notes_missing_file_returns_empty_string(tmp_path):
    assert ephys.read_notes(tmp_path / 'nope.txt') == ''


def test_base_dirs_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)   # base_dirs reads default_folders.txt from cwd
    paths = ['/data/raw', '/data/probes', '/data/probe.json', '/data/params.txt']
    ephys.write_base_dirs(paths)

    assert ephys.base_dirs() == paths
    assert ephys.base_dirs(return_keys=True) == [
        ('RAW_DATA', '/data/raw'),
        ('PROBE_FILES', '/data/probes'),
        ('DEFAULT_PROBE', '/data/probe.json'),
        ('DEFAULT_PARAMETERS', '/data/params.txt'),
    ]


def test_write_base_dirs_requires_four_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(AssertionError):
        ephys.write_base_dirs(['/data/raw'])
