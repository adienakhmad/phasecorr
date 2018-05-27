import numpy as _np
import scipy.fftpack as _scf
import scipy.signal as _scs
import warnings as _warning
import multiprocessing as _mp
import ctypes as _ctypes

__all__ = ['phasecorr']


def xcorr(signal1, signal2, **kwargs):
    kwargs['mode'] = 'pcc'
    phases1 = __instantphase(signal1, **kwargs)
    phases2 = __instantphase(signal2, **kwargs)

    lags = __default_lags_if_not_set(signal1, signal2, **kwargs)

    if kwargs.get('parallel'):
        return __phase_xcorr_parallel(phases1, phases2, lags, **kwargs)
    else:
        return __phase_xcorr(phases1, phases2, lags)


def acorr(signal1, **kwargs):
    kwargs['mode'] = 'pac'
    phase1 = __instantphase(signal1, **kwargs)

    lags = __default_lags_if_not_set(signal1, signal1, **kwargs)

    if kwargs.get('parallel'):
        return __phase_xcorr_parallel(phase1, phase1, lags, **kwargs)
    else:
        return __phase_xcorr(phase1, phase1, lags)


def __instantphase(real_signal, **kwargs):
    """
    Calculate the instantaneous phase for a given real signal
    :param real_signal:
    :return: real-valued array of phases
    """
    method = kwargs.get('analytic', 'fft')

    if method == 'fft':
        return _np.angle(_analytic_signal(real_signal))
    elif method == 'hilbert':
        return _np.angle(_scs.hilbert(real_signal))
    else:
        raise ValueError('Unknown analytic parameter', method, ' valid options: fft, hilbert')


def _analytic_signal(real_signal):
    """ Calculate analytic signal by removing
    redundant negative frequency content
    resulting from the Fourier transform.

    :param real_signal:
    :return: complex-valued analytic signal
    """
    n = real_signal.size
    nfft = __next_power_of_2(n)

    half = nfft // 2
    freq_domain = _scf.fft(real_signal, nfft)

    # multiply positive frequency component by two
    # leave the dc component untouched
    freq_domain[1:half] *= 2

    # zero the negative frequency component
    # leave the n/2 (Nyquist freq) component untouched
    freq_domain[half + 1:] = 0

    analytic_signal = _np.fft.ifft(freq_domain)

    # cast to complex64 to improve subsequent operation
    return analytic_signal[:n].astype(dtype=_np.complex64)


def __phase_xcorr(signal1_phases, signal2_phases, range_lags):
    pcc_signal = _np.empty(len(range_lags), dtype=_np.float32)

    pccset = pcc_signal.itemset

    for i in range(len(range_lags)):
        val = _phase_xcorr_at(signal1_phases, signal2_phases, range_lags[i])
        pccset(i, val)

    return pcc_signal


def __phase_xcorr_parallel(signal1_phases, signal2_phases, range_lags, **kwargs):
    mp_buffer1 = _mp.RawArray(_ctypes.c_float, signal1_phases.size)
    mp_signal1 = _np.frombuffer(mp_buffer1, dtype=_np.float32)
    _np.copyto(mp_signal1, signal1_phases)

    # if calculating auto-correlation, we only need one buffer
    if kwargs.get('mode') == 'pac':
        mp_wrapper = __PhasecorrMultiprocess(mp_buffer1, mp_buffer1, dtype=_np.float32)

    # if calculating cross-correlation, we need a second buffer
    else:
        mp_buffer2 = _mp.RawArray(_ctypes.c_float, signal2_phases.size)
        mp_signal2 = _np.frombuffer(mp_buffer2, dtype=_np.float32)
        _np.copyto(mp_signal2, signal2_phases)
        mp_wrapper = __PhasecorrMultiprocess(mp_buffer1, mp_buffer2, dtype=_np.float32)

    # degree of parallelism, number of child process to spawn
    dop = kwargs.get('processes', _mp.cpu_count())
    with _mp.Pool(processes=dop) as pool:
        result = pool.map(mp_wrapper, range_lags)

    pcc_signal = _np.array(result, dtype=_np.float32)

    return pcc_signal


def _phase_xcorr_at(signal1_phases, signal2_phases, sample_lag):
    """ Calculate number of overlapping sample
    between phase_source and phase ref
    and pass it into __pcc_at_lag

    :param signal1_phases:
    :param signal2_phases:
    :param sample_lag:
    :return:
    """

    # treat the index as 1-D line (x0, x1) coordinate
    source_x0 = 0
    source_x1 = signal1_phases.size
    ref_x0 = sample_lag
    ref_x1 = ref_x0 + signal2_phases.size

    # count overlapped samples
    # n = max(0, min(max1, max2) - max(min1, min2))
    n_overlapped = max(0, min(source_x1, ref_x1) - max(source_x0, ref_x0))

    # to count correlation at negative lag, just switch signal1 and signal2
    if sample_lag < 0:
        return __pcc_at_lag(signal2_phases, signal1_phases, -sample_lag, n_overlapped)

    return __pcc_at_lag(signal1_phases, signal2_phases, sample_lag, n_overlapped)


def __pcc_at_lag(signal1_phases, signal2_phases, sample_lag, n_overlap):
    """ Calculate phase cross correlation (pcc) from
    instantaneous phase at given sample lag

    Based on equation (5) from Schimmel, Martin. (1999).
    Phase cross-correlations: Design, comparisons, and applications.
    Bulletin of the Seismological Society of America. 89. 1366-378.

    :param signal1_phases: source signal phase
    :param signal2_phases: pilot wavelet phase
    :param sample_lag: sample lag to calculate pcc at
    :param n_overlap: number of overlapping sample
    :return: real-value scalar, correlation value at sample_lag
    """
    signal1 = signal1_phases[sample_lag:sample_lag + n_overlap]
    signal2 = signal2_phases[:n_overlap]

    if signal1.size != signal2.size:
        raise ValueError('Overlap error.\n'
                         'This function should NOT be used directly, '
                         'instead use acorr() or xcorr()')

    diff = _np.subtract(signal1, signal2) / 2
    _pcc = _np.sum(_np.abs(_np.cos(diff)) -
                   _np.abs(_np.sin(diff))) / n_overlap

    return _np.asscalar(_pcc)


# region helper functions

def __next_power_of_2(x):
    """ Calculate next power of 2 after x
    If x is a power of 2, then x is returned

    Using bit shift, works only on int type

    :param x: must be a positive integer
    :return:
    """
    return 1 << (x - 1).bit_length()


def __default_lags_if_not_set(ndarary1, ndarray2, **kwargs):
    lags = kwargs.get('lags')

    # default lags if not specified
    if lags is None:
        default_max = ndarary1.size if kwargs['mode'] == 'pac' \
            else (ndarary1.size - ndarray2.size) + 1

        if default_max == 1:
            _warning.warn('signal has same sizes.\nBy default, if no lag parameter is given, '
                          'no partial overlap will be calculated.\n'
                          'Since signal1 & signal2 have same sizes, '
                          'only lag at t=0 will be calculated.')
        lags = range(0, default_max)

    # check if the range object is valid
    if len(lags) == 0:
        raise ValueError('lags is not valid')

    return lags

# endregion


class __PhasecorrMultiprocess(object):
    """
    Helper class to run phasecorr in multiprocess
    This class is meant to be passed to multiprocessing pool.map()
    """
    def __init__(self, mp_buffer1, mp_buffer2, dtype):
        self.signal1 = _np.frombuffer(mp_buffer1, dtype=dtype)
        self.signal2 = _np.frombuffer(mp_buffer2, dtype=dtype)

    def __call__(self, sample_lag):
        return _phase_xcorr_at(self.signal1, self.signal2, sample_lag)
