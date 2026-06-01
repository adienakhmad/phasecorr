import numpy as np
import pytest

from phasecorr.phasecorr import acorr, xcorr

try:
    import obspy
    from phasecorr import phasecorr_seismic as pcs
    HAS_OBSPY = True
except ImportError:
    HAS_OBSPY = False

requires_obspy = pytest.mark.skipif(not HAS_OBSPY, reason="obspy is not installed")


def test_acorr():
    """Phase auto-correlation of a pure tone has a known closed form.

    For a cosine of period P the instantaneous phase is linear (phi[n] = w*n),
    so the phase difference over a lag k is the constant w*k. Schimmel (1999)
    eq. (5) then reduces to |cos(pi*k/P)| - |sin(pi*k/P)| at every lag.
    """
    period = 32
    n = np.arange(512)
    signal = np.cos(2 * np.pi * n / period)

    lags = np.arange(0, period + 1)
    expected = np.abs(np.cos(np.pi * lags / period)) - np.abs(np.sin(np.pi * lags / period))

    result = acorr(signal, lags=range(0, period + 1))

    np.testing.assert_allclose(result, expected, atol=1e-4)


def test_xcorr():
    """Phase cross-correlation of two tones of the same period has the same
    closed form as the auto-correlation.

    With both signals being cosines of period P, the phase difference at lag k
    is phi1[k+j] - phi2[j] = w*(k+j) - w*j = w*k, constant over the overlap, so
    Schimmel (1999) eq. (5) again reduces to |cos(pi*k/P)| - |sin(pi*k/P)|.
    """
    period = 32
    signal = 5.0 * np.cos(2 * np.pi * np.arange(512) / period)
    wavelet = 0.1 * np.cos(2 * np.pi * np.arange(128) / period)

    lags = np.arange(0, period + 1)
    expected = np.abs(np.cos(np.pi * lags / period)) - np.abs(np.sin(np.pi * lags / period))

    result = xcorr(signal, wavelet, lags=range(0, period + 1))

    np.testing.assert_allclose(result, expected, atol=1e-4)


def _tone_trace(amplitude, n, period):
    # sampling_rate = 1 Hz so tlags (seconds) map 1:1 to sample lags
    data = amplitude * np.cos(2 * np.pi * np.arange(n) / period)
    return obspy.Trace(data, header={"sampling_rate": 1.0})


@requires_obspy
def test_acorr_seismic():
    period = 32
    stream = obspy.Stream([_tone_trace(1.0, 512, period)])

    lags = np.arange(0, period + 1)
    expected = np.abs(np.cos(np.pi * lags / period)) - np.abs(np.sin(np.pi * lags / period))

    result = pcs.acorr(stream, tlags=(0, period + 1))[0]

    assert result.stats.network == "pac"
    np.testing.assert_allclose(result.data, expected, atol=1e-4)


@requires_obspy
def test_xcorr_seismic():
    period = 32
    signal = obspy.Stream([_tone_trace(5.0, 512, period)])
    wavelet = _tone_trace(0.1, 128, period)

    lags = np.arange(0, period + 1)
    expected = np.abs(np.cos(np.pi * lags / period)) - np.abs(np.sin(np.pi * lags / period))

    result = pcs.xcorr(signal, wavelet, tlags=(0, period + 1))[0]

    assert result.stats.network == "pcc"
    np.testing.assert_allclose(result.data, expected, atol=1e-4)
