"""Tests for data_processing: raw format detection, recording bounds, and chunked loading."""
import numpy as np
import pytest

import data_processing as dp


###   RAW FORMAT VALIDATION   ###

def test_validate_neuronexus_accepts_complete_pair(tmp_path):
    meta = tmp_path / 'rec.xdat.json'
    meta.touch()
    (tmp_path / 'rec_data.xdat').touch()
    assert dp.validate_neuronexus(str(meta)) is True


def test_validate_neuronexus_rejects_missing_data_file(tmp_path):
    meta = tmp_path / 'rec.xdat.json'
    meta.touch()   # metadata present, binary missing
    assert dp.validate_neuronexus(str(meta)) is False


def test_validate_neuronexus_rejects_ambiguous_folder(tmp_path):
    # two recordings in one folder -> can't tell which binary goes with which metadata
    for name in ['a.xdat.json', 'b.xdat.json', 'a_data.xdat', 'b_data.xdat']:
        (tmp_path / name).touch()
    assert dp.validate_neuronexus(str(tmp_path / 'a.xdat.json')) is False


def test_validate_neuronexus_rejects_wrong_extension(tmp_path):
    fpath = tmp_path / 'rec.json'
    fpath.touch()
    assert dp.validate_neuronexus(str(fpath)) is False


def _make_openephys_tree(root):
    """ node/experiment/recording layout expected by the OpenEphys validator """
    rec = root / 'Record Node 101' / 'experiment1' / 'recording1'
    (rec / 'continuous').mkdir(parents=True)
    (root / 'Record Node 101' / 'settings.xml').touch()
    oebin = rec / 'structure.oebin'
    oebin.touch()
    return oebin


def test_validate_openephys_accepts_complete_tree(tmp_path):
    oebin = _make_openephys_tree(tmp_path)
    assert dp.validate_openephys(str(oebin)) is True


def test_validate_openephys_rejects_missing_continuous_dir(tmp_path):
    oebin = _make_openephys_tree(tmp_path)
    (oebin.parent / 'continuous').rmdir()
    assert dp.validate_openephys(str(oebin)) is False


def test_validate_openephys_rejects_missing_settings_xml(tmp_path):
    oebin = _make_openephys_tree(tmp_path)
    (tmp_path / 'Record Node 101' / 'settings.xml').unlink()
    assert dp.validate_openephys(str(oebin)) is False


def test_validate_openephys_rejects_wrong_filename(tmp_path):
    oebin = _make_openephys_tree(tmp_path)
    other = oebin.parent / 'structure.json'
    other.touch()
    assert dp.validate_openephys(str(other)) is False


def test_validate_neuralynx_checks_extension():
    assert dp.validate_neuralynx('CSC1.ncs') is True
    assert dp.validate_neuralynx('CSC1.txt') is False


@pytest.mark.parametrize('fname, expected', [
    ('rec.nwb', 'NWB'),
    ('rec.npy', 'NPY'),
    ('rec.mat', 'MAT'),
    ('CSC1.ncs', 'Neuralynx'),
])
def test_get_data_format_identifies_supported_files(tmp_path, fname, expected):
    fpath = tmp_path / fname
    fpath.touch()
    assert dp.get_data_format(str(fpath)) == expected


def test_get_data_format_identifies_neuronexus(tmp_path):
    meta = tmp_path / 'rec.xdat.json'
    meta.touch()
    (tmp_path / 'rec_data.xdat').touch()
    assert dp.get_data_format(str(meta)) == 'NeuroNexus'


def test_get_data_format_identifies_openephys(tmp_path):
    oebin = _make_openephys_tree(tmp_path)
    assert dp.get_data_format(str(oebin)) == 'OpenEphys'


def test_get_data_format_rejects_unsupported_file(tmp_path):
    fpath = tmp_path / 'rec.txt'
    fpath.touch()
    with pytest.raises(Exception, match='not a supported data file'):
        dp.get_data_format(str(fpath))


def test_get_data_format_rejects_nonexistent_path(tmp_path):
    with pytest.raises(Exception):
        dp.get_data_format(str(tmp_path / 'nope.nwb'))


###   RECORDING BOUNDS   ###

def test_get_rec_bounds_full_recording_by_default():
    assert dp.get_rec_bounds(10000, 1000) == (0, 10000)


def test_get_rec_bounds_converts_seconds_to_samples():
    assert dp.get_rec_bounds(10000, 1000, tstart=2, tend=5) == (2000, 5000)


def test_get_rec_bounds_enforces_minimum_one_second_window():
    # tend only 100 ms after tstart -> pushed out to tstart + 1 s
    assert dp.get_rec_bounds(10000, 1000, tstart=2, tend=2.1) == (2000, 3000)


def test_get_rec_bounds_clamps_start_to_last_second():
    # tstart past the end of the recording -> clamped to NSAMPLES - FS
    assert dp.get_rec_bounds(10000, 1000, tstart=999) == (9000, 10000)


def test_get_rec_bounds_clamps_end_to_recording_length():
    assert dp.get_rec_bounds(10000, 1000, tstart=0, tend=999) == (0, 10000)


def test_get_rec_bounds_negative_start_clamps_to_zero():
    assert dp.get_rec_bounds(10000, 1000, tstart=-5, tend=5) == (0, 5000)


def test_get_rec_bounds_matches_get_rec_bounds2():
    # the two implementations are duplicates; keep them in agreement
    for kw in [dict(), dict(tstart=2, tend=5), dict(tstart=2, tend=2.1), dict(tstart=999)]:
        assert dp.get_rec_bounds(10000, 1000, **kw) == dp.get_rec_bounds2(10000, 1000, **kw)


###   CHUNKED LOADING   ###

def test_get_chunkfunc_steps_through_recording():
    fx, (rec_n, proc_n) = dp.get_chunkfunc(loading_window_s=2, recording_fs=1000,
                                           recording_n_samples=10000, i_start=0, i_end=10000)
    assert (rec_n, proc_n) == (2000, 2000)   # no downsampling -> same chunk size

    (r0, r1), (p0, p1), _ = fx(0)
    assert (r0, r1) == (0, 2000)
    assert (p0, p1) == (0, 2000)

    (r0, r1), _, _ = fx(2)
    assert (r0, r1) == (4000, 6000)


def test_get_chunkfunc_final_chunk_is_clipped_to_recording_end():
    fx, _ = dp.get_chunkfunc(loading_window_s=3, recording_fs=1000,
                             recording_n_samples=10000, i_start=0, i_end=10000)
    # chunk 3 would run 9000-12000; it must stop at the last sample
    (r0, r1), _, _ = fx(3)
    assert (r0, r1) == (9000, 10000)


def test_get_chunkfunc_downsamples_processed_indices():
    fx, (rec_n, proc_n) = dp.get_chunkfunc(loading_window_s=2, recording_fs=2000,
                                           recording_n_samples=20000, target_fs=1000,
                                           i_start=0, i_end=20000)
    assert rec_n == 4000    # 2 s at 2 kHz
    assert proc_n == 2000   # 2 s at 1 kHz

    (r0, r1), (p0, p1), _ = fx(1)
    assert (r0, r1) == (4000, 8000)
    assert (p0, p1) == (2000, 4000)   # half, per the 2x downsampling factor


def test_get_chunkfunc_respects_start_offset():
    fx, _ = dp.get_chunkfunc(loading_window_s=2, recording_fs=1000,
                             recording_n_samples=10000, i_start=1000, i_end=9000)
    (r0, r1), _, _ = fx(0)
    assert (r0, r1) == (1000, 3000)

    # last chunk is clipped to i_end, not to the recording length
    (r0, r1), _, _ = fx(3)
    assert (r0, r1) == (7000, 9000)


def test_get_chunkfunc_returns_progress_text():
    fx, _ = dp.get_chunkfunc(loading_window_s=60, recording_fs=1000,
                             recording_n_samples=600000, i_start=0, i_end=600000)
    _, _, txt = fx(0)
    assert 'Extracting' in txt


def test_resample_chunk_changes_sample_count_not_channel_count():
    snip = np.random.default_rng(0).normal(size=(8, 2000))
    out = dp.resample_chunk(snip, 1000)
    assert out.shape == (8, 1000)


def test_resample_chunk_preserves_low_frequency_signal():
    fs = 2000
    t = np.arange(0, 1, 1 / fs)
    snip = np.sin(2 * np.pi * 5 * t)[None, :]    # 5 Hz, far below either Nyquist
    out = dp.resample_chunk(snip, 1000)

    t_ds = np.arange(0, 1, 1 / 1000)
    expected = np.sin(2 * np.pi * 5 * t_ds)
    assert np.corrcoef(out[0], expected)[0, 1] > 0.99


###   PROCESSED DIRECTORY VALIDATION   ###

def test_validate_processed_ddir_accepts_hdf5_output(tmp_path):
    for name in ['probe_group', 'params.pkl', 'DATA.hdf5']:
        (tmp_path / name).touch()
    assert dp.validate_processed_ddir(tmp_path) == 1


def test_validate_processed_ddir_accepts_legacy_npz_output(tmp_path):
    for name in ['probe_group', 'params.pkl', 'lfp_bp.npz', 'lfp_time.npy',
                 'lfp_fs.npy', 'ALL_DS', 'ALL_SWR']:
        (tmp_path / name).touch()
    assert dp.validate_processed_ddir(tmp_path) == 2


def test_validate_processed_ddir_rejects_incomplete_dir(tmp_path):
    (tmp_path / 'probe_group').touch()   # missing params.pkl and data files
    assert dp.validate_processed_ddir(tmp_path) == 0


def test_validate_processed_ddir_rejects_missing_dir(tmp_path):
    assert dp.validate_processed_ddir(tmp_path / 'nope') == 0
