# ligotools/tests/test_utils.py
import numpy as np
from scipy.io import wavfile as wavread
from ligotools.utils import write_wavfile, reqshift

def test_write_wavfile_roundtrip(tmp_path):
    fs = 4096
    t = np.linspace(0, 1, fs, endpoint=False)
    x = 0.5 * np.sin(2 * np.pi * 440 * t)  # -1..1 scale (peak 0.5)

    out = tmp_path / "test.wav"
    write_wavfile(str(out), fs, x)

    rfs, y = wavread.read(str(out))
    # basic checks
    assert rfs == fs
    assert y.dtype == np.int16
    assert len(y) == len(x)
    # scaled to ~0.9 * max int16
    assert abs(int(y.max()) - int(0.9 * 32767)) <= 3  # small tolerance


def test_reqshift_moves_tone_frequency():
    sample_rate = 4096
    N = 4096
    t = np.arange(N) / sample_rate

    f0 = 100.0
    fshift = 200.0
    x = np.sin(2 * np.pi * f0 * t)

    y = reqshift(x, fshift=fshift, sample_rate=sample_rate)

    # dominant frequency before
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, 1 / sample_rate)
    f_peak_before = freqs[np.argmax(np.abs(X))]

    # dominant frequency after
    Y = np.fft.rfft(y)
    f_peak_after = freqs[np.argmax(np.abs(Y))]

    assert abs(f_peak_before - f0) <= 1.0  # within one FFT bin
    assert abs(f_peak_after - (f0 + fshift)) <= 1.0
