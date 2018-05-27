## Phase Cross/Auto Correlation
Python implementation of amplitude-unbiased, phase-based correlation technique presented on

> Schimmel, Martin. (1999). [Phase cross-correlations: Design, comparisons, and applications](https://www.researchgate.net/publication/228578991_Phase_cross-correlations_Design_comparisons_and_applications). Bulletin of the Seismological Society of America. 89. 1366-378. 

## Quick Intro

There are two main module `phasecorr` and `phasecorr_seismic`. 
The former is written to work with regular 1-D numpy array, while the latter is a wrapper to simplify working with seismic files.

There are two function in either `phasecorr` and `phasecorr_seismic` namely `xcorr` for cross-corelation and `acorr` for auto-correlation.

Some switches are availble to be passed to either function to control its behavior.
- `lags = range(min, max, step)` : control which sample lag are calculated, use regular Python 3 `range` object 
- `analytic = string` : specify which method to calculate the analytic signals, valid options: `'fft'` and `'hilbert'`
- `parallel = boolean` : if `True` calculation will utilize Python multiprocessing library
- `processes = int` : control number of child process to run in if `parallel` is set to `True`


- `tlags = (tmin, tmax) ` : only for `phasecorr_seismic`. Tuple of tmin and tmax. Serve the same purpose as `lags` but use relative second instead. Require sampling rate information in the seismic files to be correct.
- `step` : only applicable if `tlags` is set, control the step between sample lag to be calculated.
## Dependency

1. numpy
2. scipy
3. [obspy](https://github.com/obspy/obspy) - only for `phasecorr_seismic`

## Example

Cross-corelation from numpy array
```python
import numpy as np
from phasecorr.phasecorr import xcorr

signal = np.zeros(15)
signal[5:8] = [0.5, 2, 0.5]

# signal
# [0.  0.  0.  0.  0.  0.5  2.0  0.5  0.  0.  0.  0.  0.  0.  0. ]

wavelet = np.array([0.25, 1, 0.25])

# calculate correlation at sample lag=0 until lag=10
pcc = xcorr(signal, wavelet, lags=range(0, 11))

# pcc (rounded to 2-decimal places for this demonstration only)
# [-0.   -0.   -0.    0.08  0.44  0.97  0.44  0.08 -0.   -0.   -0.  ]
```
Auto-correlation from numpy array
```python
import numpy as np
from phasecorr.phasecorr import acorr

# signal = [0. 0. 0. 0. 0. 0.5 2. 0.5 0. 0. 0. 0. 0. 0. 0. ]
signal = np.zeros(15)
signal[5:8] = [0.5, 2, 0.5]

# calculate correlation at sample lag= -5 until lag= +5
pac = acorr(signal, lags=range(-5, 6))

# pac (rounded to 2-decimal places for this demonstration only)
# [-0.2   0.09  0.33  0.54  0.72  1.    0.72  0.54  0.33  0.09 -0.2 ]
```

Cross-correlation from seismic files
```python
import obspy
# note that we use phasecorr_seismic module now
from phasecorr.phasecorr_seismic import xcorr

# use example data from obspy
st = obspy.read()
print(st)

# use the first trace as wavelet
wavelet = st[0]
print(wavelet)

# all traces in st will be correlated against the same wavelet
# for seismic files with correct header, time lags (second) can be used instead of sample lags
pcc = xcorr(st, wavelet, tlags=(0, 10)) # calculate from time lag= 0s to time lag= 10s

# the returned object is an obspy Stream
print(pcc)

# output (notice that this program change the network string)
# BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:32.990000Z | 100.0 Hz, 3000 samples
# BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:32.990000Z | 100.0 Hz, 3000 samples
# BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:32.990000Z | 100.0 Hz, 3000 samples
# BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:32.990000Z | 100.0 Hz, 3000 samples
#
# 3 Trace(s) in Stream:
# pcc.RJOB..EHZ | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:12.990000Z | 100.0 Hz, 1000 samples
# pcc.RJOB..EHN | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:12.990000Z | 100.0 Hz, 1000 samples
# pcc.RJOB..EHE | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:12.990000Z | 100.0 Hz, 1000 samples
```
Auto-correlation from seismic files
```python
import obspy
# note that we use phasecorr_seismic module now
from phasecorr.phasecorr_seismic import acorr

# use example data from obspy
st = obspy.read()
print(st)


# all traces in st will be correlated against itself
# for seismic files with correct header, time lags (second) can be used instead of sample lags
pac = acorr(st, tlags=(0, 10)) # calculate from time lag= 0s to time lag=10s

# the returned object is an obspy Stream
print(pac)

# output (notice that this program change the network string)
# 3 Trace(s) in Stream:
# BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:32.990000Z | 100.0 Hz, 3000 samples
# BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:32.990000Z | 100.0 Hz, 3000 samples
# BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:32.990000Z | 100.0 Hz, 3000 samples
#
# 3 Trace(s) in Stream:
# pac.RJOB..EHZ | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:12.990000Z | 100.0 Hz, 1000 samples
# pac.RJOB..EHN | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:12.990000Z | 100.0 Hz, 1000 samples
# pac.RJOB..EHE | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:12.990000Z | 100.0 Hz, 1000 samples
```
## Calculation in parallel

Due to the nature of the technique, calculating correlation for longer signal can take quite long time.
An optional switch `parallel=True` can be passed to utilize python multiprocessing library. 
To control the number of multiprocess to spawn, use `processes= no_of_process` switch

For Windows user, it is required to protect entry point of your program with `if __name__ == '__main__'` to prevent infinite spawning of child process.

Though not required, it's advisable for Linux user to use `if __name__ == '__main__'` appropriately, because it makes the intended division of work clearer.

```python
import obspy
from phasecorr.phasecorr_seismic import acorr

# required check for Windows user to run multiprocess
if __name__ == '__main__':
    # use example data from obspy
    st = obspy.read()
    print(st)

    # calculate from time lag= 0s to time lag=10s
    # all traces in st will be correlated against itself
    pac = acorr(st, tlags=(0, 10), parallel=True, processes=4)

    # the returned object is an obspy Stream
    print(pac)
    
 ```
