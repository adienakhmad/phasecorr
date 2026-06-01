from obspy.core import trace as _tr
from obspy.core import stream as _st
import warnings as _warning
import phasecorr.phasecorr as phasecorr

__all__ = ['xcorr', 'acorr']


def xcorr(seismic_signal, wavelet, **kwargs):
    """
    Calculate phase cross correlation (pcc)
    between signal and wavelet
    foreach signal in seismic

    For this purpose wavelet is shifted in time and compared to
    corresponding portion in each signals

    :param seismic_signal: seismic stream, may contain multiple traces
    :param wavelet: seismic trace, or stream containing single trace
    :param kwargs:
    :return: cross-correlation as stream
    """

    # if seismic signal is a trace object, we pack it to a stream
    if isinstance(seismic_signal, _tr.Trace):
        sources = _st.Stream([seismic_signal])
    else:
        sources = seismic_signal

    if not isinstance(sources, _st.Stream):
        raise TypeError('seismic signal is not a Stream nor Trace object')

    # if wavelet is a stream, we take the first trace
    if isinstance(wavelet, _st.Stream):
        ref = wavelet[0]
        if wavelet.count() > 1:
            _warning.warn('wavelet contains multiple traces. Using the first trace as wavelet')
    else:
        ref = wavelet

    if not isinstance(ref, _tr.Trace):
        raise TypeError('wavelet is not a Stream nor Trace object')

    return _st.Stream([_xcorr_trace(tr, ref, **kwargs) for tr in sources])


def acorr(seismic_signal, **kwargs):
    """
    Calculate phase auto-correlation for each signal in seismic_stream

    :param seismic_signal:
    :param kwargs:
    :return:
    """
    # if seismic signal is a trace object, we pack it to a stream
    if isinstance(seismic_signal, _tr.Trace):
        sources = _st.Stream([seismic_signal])
    else:
        sources = seismic_signal

    if not isinstance(sources, _st.Stream):
        raise TypeError('seismic_stream is not a Stream nor Trace object')

    return _st.Stream([_acorr_trace(tr, **kwargs) for tr in sources])


def _xcorr_trace(signal1, signal2, **kwargs):
    """
    Calculate phase cross correlation (pcc)
    between signal1 and signal2

    For this purpose signal2 is shifted in time and compared to
    corresponding portion in signal1

    :type signal1: obspy.core.trace.Trace
    :type signal2: obspy.core.trace.Trace
    :param signal1: seismic trace
    :param signal2: seismic trace (wavelet) to correlate with
    :return: cross-correlation as trace
    """

    kwargs['mode'] = 'pcc'
    kwargs['lags'] = __default_lags_if_not_set(signal1, signal2, **kwargs)

    pcc_signal = phasecorr.xcorr(signal1.data, signal2.data, **kwargs)

    trace = _tr.Trace(data=pcc_signal)
    __writeheader(trace, signal1, **kwargs)

    return trace


def _acorr_trace(signal1, **kwargs):
    """
    Calculate phase auto correlation (pac)
    of signal1

    For this purpose a shifted copy in time of signal1 is compared to
    corresponding portion in signal1

    :type signal1: obspy.core.trace.Trace
    :param signal1: seismic trace
    :param kwargs:
    :return: auto-correlation as trace
    """
    kwargs['mode'] = 'pac'
    kwargs['lags'] = __default_lags_if_not_set(signal1, signal1, **kwargs)

    pac_signal = phasecorr.acorr(signal1.data, **kwargs)

    trace = _tr.Trace(data=pac_signal)

    __writeheader(trace, signal1, **kwargs)
    return trace


def __default_lags_if_not_set(trace1, trace2, **kwargs):

    if trace1.stats.sampling_rate != trace2.stats.sampling_rate:
        raise ValueError('sampling rate do not match between seismic and wavelet')

    tlags = kwargs.get('tlags')

    if tlags is not None:
        if not isinstance(tlags, tuple):
            raise TypeError('argument tlags must be a tuple of (tmin, tmax)')
        if len(tlags) != 2:
            raise ValueError('tmin and tmax must be specified.')
        nmin = int(tlags[0] * trace1.stats.sampling_rate)
        nmax = int(tlags[1] * trace1.stats.sampling_rate)
        lags = range(nmin, nmax, kwargs.get('step', 1))

    else:
        default = phasecorr.__default_lags_if_not_set(trace1.data, trace2.data, **kwargs)
        lags = kwargs.get('lag', default)

    return lags


def __writeheader(trace, source, **kwargs):
    lags = kwargs.get('lags')
    trace.stats.network = kwargs['mode']
    trace.stats.station = source.stats.station
    trace.stats.channel = source.stats.channel
    trace.stats.sampling_rate = source.stats.sampling_rate / lags.step
    trace.stats.starttime = \
        source.stats.starttime + (lags.start * trace.stats.delta)