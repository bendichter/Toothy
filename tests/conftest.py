"""Shared fixtures and headless setup for the Toothy test suite."""
import os
import sys
from pathlib import Path

# Toothy imports PyQt5 and matplotlib at module level; force both to run headless
# before any test module imports them.
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('MPLBACKEND', 'Agg')

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import pytest


LFP_FS = 1000.0


def gaussian_bump(t, center, amp, sigma=0.005):
    """ Single dentate-spike-like waveform: Gaussian of $amp mV at $center s """
    return amp * np.exp(-0.5 * ((t - center) / sigma) ** 2)


def ripple_burst(t, center, amp, freq=150.0, sigma=0.015):
    """ Sharp-wave-ripple-like waveform: $freq Hz sine under a Gaussian envelope """
    envelope = amp * np.exp(-0.5 * ((t - center) / sigma) ** 2)
    return envelope * np.sin(2 * np.pi * freq * (t - center))


@pytest.fixture
def lfp_time():
    """ 10 s of timestamps at 1 kHz """
    return np.arange(0, 10, 1 / LFP_FS)


@pytest.fixture
def ds_lfp(lfp_time):
    """ Flat LFP with three 1.0 mV dentate spikes at t = 1, 4, and 7 s """
    lfp = np.zeros(len(lfp_time), dtype='float64')
    for center in (1.0, 4.0, 7.0):
        lfp += gaussian_bump(lfp_time, center, amp=1.0)
    return lfp


@pytest.fixture
def swr_lfp():
    """ 5 s of LFP at 1 kHz with three 150 Hz ripple bursts at t = 1.5, 2.5, 3.5 s """
    t = np.arange(0, 5, 1 / LFP_FS)
    lfp = np.zeros(len(t), dtype='float64')
    for center in (1.5, 2.5, 3.5):
        lfp += ripple_burst(t, center, amp=1.0)
    return t, lfp


@pytest.fixture
def std_df():
    """ Per-channel frequency band power for an 8-channel probe """
    return pd.DataFrame({
        'theta'      : [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.0],
        'slow_gamma' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        'fast_gamma' : [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        'swr'        : [0.0, 0.0, 0.0, 0.0, 0.1, 1.0, 0.1, 0.9],
    }, index=np.arange(8))


@pytest.fixture
def event_df():
    """ Raw per-event dataframe indexed by channel: 2 events on ch 0, 1 on ch 3 """
    return pd.DataFrame({
        'idx' : [100, 200, 300],
        'amp' : [1.0, 3.0, 5.0],
        'time': [0.1, 0.2, 0.3],
    }, index=pd.Index([0, 0, 3], name=None))
