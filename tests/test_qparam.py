"""Tests for qparam: parameter validation, repair, and .txt file I/O."""
import pytest

from toothy import qparam


###   VALIDATION   ###

def test_original_defaults_are_valid():
    is_valid, valid_ddict = qparam.validate_params(qparam.get_original_defaults())
    assert is_valid is True
    assert all(valid_ddict.values())


def test_every_default_has_a_label_and_description():
    info = qparam.get_param_info()
    for key in qparam.get_original_defaults():
        assert key in info, f'{key} is missing from PARAM_INFO'
        assert len(info[key]) == 2


def test_validate_params_flags_missing_key():
    ddict = qparam.get_original_defaults()
    del ddict['lfp_fs']
    is_valid, valid_ddict = qparam.validate_params(ddict)
    assert is_valid is False
    assert valid_ddict['lfp_fs'] is False


def test_validate_params_flags_scalar_where_range_expected():
    ddict = qparam.get_original_defaults()
    ddict['theta'] = 6.0  # must be a [low, high] pair
    assert qparam.validate_params(ddict)[1]['theta'] is False


def test_validate_params_flags_wrong_length_range():
    ddict = qparam.get_original_defaults()
    ddict['theta'] = [6.0, 10.0, 12.0]
    assert qparam.validate_params(ddict)[1]['theta'] is False


def test_validate_params_flags_string_where_number_expected():
    ddict = qparam.get_original_defaults()
    ddict['ds_height_thr'] = 'high'
    assert qparam.validate_params(ddict)[1]['ds_height_thr'] is False


@pytest.mark.parametrize('key, bad_value', [
    ('csd_method', 'bogus'),
    ('f_type', 'bogus'),
    ('clus_algo', 'bogus'),
    ('el_shape', 'bogus'),
    ('vaknin_el', 'maybe'),
])
def test_validate_params_flags_unknown_category(key, bad_value):
    ddict = qparam.get_original_defaults()
    ddict[key] = bad_value
    assert qparam.validate_params(ddict)[1][key] is False


###   REPAIR   ###

def test_fix_params_replaces_invalid_and_keeps_valid():
    defaults = qparam.get_original_defaults()
    ddict = dict(defaults)
    ddict['csd_method'] = 'bogus'   # invalid -> should be reset
    ddict['ds_wlen'] = 999.0        # valid (if unusual) -> should survive

    fixed = qparam.fix_params(ddict)
    assert fixed['csd_method'] == defaults['csd_method']
    assert fixed['ds_wlen'] == 999.0


def test_fix_params_fills_empty_dict_with_defaults():
    assert qparam.fix_params({}) == qparam.get_original_defaults()


def test_fix_params_drops_unrecognized_keys():
    ddict = dict(qparam.get_original_defaults())
    ddict['not_a_real_param'] = 1.0
    assert 'not_a_real_param' not in qparam.fix_params(ddict)


###   FILE I/O   ###

def test_param_txt_to_dict_parses_types_and_skips_comments(tmp_path):
    fpath = tmp_path / 'params.txt'
    fpath.write_text(
        '###  PARAMETERS  ###\n'
        '\n'
        'lfp_fs = 1000.0;\n'
        'trange = [0.0, -1.0];\n'
        'csd_method = standard;\n'
        'vaknin_el = True;\n'
        '# a comment line\n'
    )
    ddict = qparam.param_txt_to_dict(fpath)
    assert ddict == {
        'lfp_fs': 1000.0,
        'trange': [0.0, -1.0],
        'csd_method': 'standard',
        'vaknin_el': True,
    }


def test_param_file_roundtrip_preserves_defaults(tmp_path):
    fpath = tmp_path / 'default_params.txt'
    qparam.write_param_file(qparam.get_original_defaults(), fpath)

    ddict, invalid = qparam.read_param_file(fpath)
    assert invalid == []
    assert ddict == qparam.get_original_defaults()


def test_read_param_file_missing_file_returns_none():
    assert qparam.read_param_file('does_not_exist.txt') == (None, [])


def test_read_param_file_reports_invalid_params(tmp_path):
    ddict = qparam.get_original_defaults()
    ddict['csd_method'] = 'bogus'
    fpath = tmp_path / 'params.txt'
    qparam.write_param_file(ddict, fpath)

    # return_none=True -> caller gets nothing but the list of bad keys
    assert qparam.read_param_file(fpath, return_none=True) == (None, ['csd_method'])

    # return_none=False -> caller gets the loaded params AND the list of bad keys
    loaded, invalid = qparam.read_param_file(fpath, return_none=False)
    assert invalid == ['csd_method']
    assert loaded['csd_method'] == 'bogus'


def test_read_param_file_missing_entry_is_repairable(tmp_path):
    ddict = qparam.get_original_defaults()
    del ddict['ds_wlen']
    fpath = tmp_path / 'params.txt'
    qparam.write_param_file(ddict, fpath)

    loaded, invalid = qparam.read_param_file(fpath, return_none=False)
    assert invalid == ['ds_wlen']
    assert qparam.fix_params(loaded)['ds_wlen'] == qparam.get_original_defaults()['ds_wlen']
