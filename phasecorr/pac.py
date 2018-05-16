import numpy as _np
import obspy as _obspy
import multiprocessing as _mp
import scipy.signal as _scs
import ctypes

__all__ = ['PhaseAutocorr']


def _phase_autocorr_at(hilbert_function, sample_lag):
    """ Calculate phase auto-correlation from hilbert function array at given sample_lag
    :param hilbert_function: hilbert transform of a source signal
    :param sample_lag: number of lag (shift) in sample
    :return: phase auto-correlation of hilbert_function evaluated at sample_lag
    """
    # quick shortcut
    if sample_lag == 0:
        return 1

    length = hilbert_function.size - sample_lag
    signal1 = hilbert_function[sample_lag:]
    signal2 = hilbert_function[:length]
    diff = _np.subtract(signal1, signal2) / 2
    sum_cos_sin = _np.sum(_np.abs(_np.cos(diff)) -
                          _np.abs(_np.sin(diff))) / length
    return _np.asscalar(sum_cos_sin)


class PhaseAutocorr(object):
    """
    Phase Auto Correlation
    """

    def __init__(self, obspy_stream):
        self.stream = obspy_stream

    def calc(self, sample_lags=None):
        """
        Calculate phase autocorrelation at given sample lags.
        :param sample_lags: if None, calculate all from lag 0 up-to len(signal) - 1
        :return:
        """
        correlated_traces = [self.__calc_at_trace(trace, sample_lags)
                             for trace in self.stream]
        return _obspy.core.stream.Stream(traces=correlated_traces)

    # in Windows, this function is totally broken if called directly from interpreter
    # you should always guard this function call with if __name___ == '__main__'
    # since this uses module level global shared array, multiple asynchronous
    # of this call will definitely break

    def calc_parallel(self, sample_lags=None, dop=_mp.cpu_count()):
        """
        Calculate phase autocorrelation at given sample lags using multiprocess parallelism
        :param sample_lags: if None, calculate all from lag 0 up-to len(signal) - 1
        :param dop: degree of parallelism, number of child process to spawn
        :return:
        """
        correlated_traces = [self.__calc_at_trace_parallel(
            trace, sample_lags, dop) for trace in self.stream]
        return _obspy.core.stream.Stream(traces=correlated_traces)

    @staticmethod
    def __instant_phase(trace):
        return _np.unwrap(_np.angle(_scs.hilbert(trace.data)))

    @classmethod
    def __calc_at_trace(cls, trace, sample_lags):
        inst_phase = cls.__instant_phase(trace)
        sample_lags = range(trace.data.size) if sample_lags is None else sample_lags

        # efficient way to write per element to numpy array
        new_trace_data = _np.empty(len(sample_lags), dtype=_np.float32)
        iterator = _np.nditer(new_trace_data, flags=['c_index'], op_flags=['writeonly'])
        while not iterator.finished:
            iterator[0] = _phase_autocorr_at(
                inst_phase, sample_lags[iterator.index])
            iterator.iternext()

        out_trace = _obspy.core.trace.Trace(header=trace.stats)
        out_trace.data = new_trace_data
        return out_trace

    @classmethod
    def __calc_at_trace_parallel(cls, trace, sample_lags, number_of_processes):
        inst_phase = cls.__instant_phase(trace)
        sample_lags = range(trace.data.size) if sample_lags is None else sample_lags

        raw_arr = _mp.RawArray(ctypes.c_float, inst_phase.size)
        buffer_as_numpy = _np.frombuffer(raw_arr, dtype=_np.float32)
        _np.copyto(buffer_as_numpy, inst_phase)

        with _mp.Pool(processes=number_of_processes) as pool:
            result = pool.map(_PacParallel(raw_arr, dtype=_np.float32), sample_lags)

        out_trace = _obspy.core.trace.Trace(header=trace.stats)
        out_trace.data = _np.array(result)
        return out_trace


class _PacParallel(object):
    """
    Helper class to calculate pac in parallel, meant to be passed to pool.map
    """
    def __init__(self, mp_raw_array, dtype):
        self._numpy_arr = _np.frombuffer(mp_raw_array, dtype=dtype)

    def __call__(self, lag):
        return _phase_autocorr_at(self._numpy_arr, lag)
