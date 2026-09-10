"""SNR computation, ported from ref_read_m_file.py's field-tested filter/window
convention on this exact dataset (line 80 + its filter chain), generalized to
a proper dB amplitude-ratio and computed independently per component.
"""

import numpy as np
from obspy import Trace, UTCDateTime

import config


def _filtered_trace(data, sample_rate):
    tr = Trace(data=np.asarray(data, dtype=np.float64))
    tr.stats.sampling_rate = sample_rate
    tr.detrend("linear")
    tr.filter("highpass", freq=config.SNR_HIGHPASS_FREQ, corners=config.SNR_HIGHPASS_CORNERS, zerophase=True)
    tr.filter("bandpass", freqmin=config.SNR_BANDPASS_FREQMIN, freqmax=config.SNR_BANDPASS_FREQMAX,
              corners=config.SNR_BANDPASS_CORNERS, zerophase=True)
    return tr


def _rms(data):
    if len(data) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(data))))


def compute_snr_db(data, sample_rate, ref_offset_sec, eps=1e-9):
    """data: 1-D array for a single channel of an already-extracted window.
    ref_offset_sec: seconds from window_start to ref_time (where signal/noise
    windows are anchored)."""
    tr = _filtered_trace(data, sample_rate)

    sig_start = ref_offset_sec + config.SNR_SIGNAL_WINDOW[0]
    sig_end = ref_offset_sec + config.SNR_SIGNAL_WINDOW[1]
    noise_start = ref_offset_sec + config.SNR_NOISE_WINDOW[0]
    noise_end = ref_offset_sec + config.SNR_NOISE_WINDOW[1]

    if noise_start < 0:
        return float("nan")

    sig_slice = tr.slice(tr.stats.starttime + sig_start, tr.stats.starttime + sig_end)
    noise_slice = tr.slice(tr.stats.starttime + noise_start, tr.stats.starttime + noise_end)

    rms_sig = _rms(sig_slice.data)
    rms_noise = max(_rms(noise_slice.data), eps)

    if rms_sig <= 0:
        return float("nan")

    return 20.0 * np.log10(rms_sig / rms_noise)
