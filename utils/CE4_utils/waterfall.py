import os
import time
from pathlib import Path

import numpy as np

from .io_utils import infer_dt_from_2cl, infer_freq_axis_from_2cl, match_2cl_for_2c, read_ce4_2c


class CE4Waterfall:
    """Waterfall-like adapter for CE4 .2C dynamic spectra."""

    freq_axis = 2
    time_axis = 0
    beam_axis = 1

    def __init__(self, filename, f_start=None, f_stop=None, t_start=None, t_stop=None, load_data=True,
                 max_load=None, xml_path=None, freq_start_mhz=None, freq_stop_mhz=None, ascending=True):
        if filename is None:
            raise ValueError("A CE4 .2C filename must be supplied.")

        self.filename = str(filename)
        self.ext = Path(filename).suffix.lower()
        if self.ext != ".2c" or Path(filename).name.startswith("._"):
            raise ValueError(f"Expected a CE4 .2C file, got: {filename}")

        self.path_2cl = xml_path or match_2cl_for_2c(self.filename, os.path.dirname(self.filename))
        self.mm = read_ce4_2c(self.filename)
        self._spec = self.mm["spec"]
        self.file_shape = (self._spec.shape[0], 1, self._spec.shape[1])
        self.n_ints_in_file = self._spec.shape[0]
        self.n_channels_in_file = self._spec.shape[1]
        self.file_size_bytes = os.path.getsize(self.filename)

        full_freqs, self.band_meta = infer_freq_axis_from_2cl(self.path_2cl, self.n_channels_in_file)
        if freq_start_mhz is not None and freq_stop_mhz is not None:
            full_freqs = np.linspace(float(freq_start_mhz), float(freq_stop_mhz), self.n_channels_in_file,
                                     dtype=np.float64)
        if not ascending:
            full_freqs = full_freqs[::-1]
        self._full_freqs = full_freqs

        self._t0 = 0 if t_start is None else int(t_start)
        self._t1 = self.n_ints_in_file if t_stop is None else int(t_stop)
        self._t0 = max(0, min(self._t0, self.n_ints_in_file))
        self._t1 = max(self._t0, min(self._t1, self.n_ints_in_file))

        self._f0, self._f1 = self._freq_bounds_to_slice(f_start, f_stop, self._full_freqs)
        self.freqs = self._full_freqs[self._f0:self._f1]
        self.timestamps = np.arange(self._t0, self._t1, dtype=np.float64)
        self.selection_shape = (self._t1 - self._t0, 1, self._f1 - self._f0)

        tsamp = infer_dt_from_2cl(self.path_2cl, self.n_ints_in_file)
        foff = float(self.freqs[1] - self.freqs[0]) if self.freqs.size > 1 else 0.0
        fch1 = float(self.freqs[0]) if self.freqs.size else 0.0
        self.header = {
            "source_name": Path(self.filename).stem,
            "tsamp": tsamp,
            "nchans": int(self.selection_shape[self.freq_axis]),
            "fch1": fch1,
            "foff": foff,
            "data_type": 1,
            "rawdatafile": self.filename,
        }
        self.file_header = dict(self.header)
        self.data = None if not load_data else self._spec[self._t0:self._t1, np.newaxis, self._f0:self._f1]
        self.is_monotonic_inc = bool(np.all(np.diff(self.freqs) >= 0)) if self.freqs.size > 1 else True
        self.is_monotonic_dec = bool(np.all(np.diff(self.freqs) <= 0)) if self.freqs.size > 1 else True

    def __repr__(self):
        return f"CE4Waterfall data: {self.filename}"

    def get_freqs(self):
        return self.freqs.copy()

    def info(self):
        print("\n--- CE4 2C File Info ---")
        print(f"{'Filename':>24s} : {self.filename}")
        print(f"{'2CL label':>24s} : {self.path_2cl}")
        print(f"{'Num records in file':>24s} : {self.n_ints_in_file}")
        print(f"{'File shape':>24s} : {self.file_shape}")
        print(f"{'File size (bytes)':>24s} : {self.file_size_bytes}")
        print("--- Selection Info ---")
        print(f"{'Data selection shape':>24s} : {self.selection_shape}")
        print(f"{'Minimum freq (MHz)':>24s} : {float(np.nanmin(self.freqs)) if self.freqs.size else np.nan}")
        print(f"{'Maximum freq (MHz)':>24s} : {float(np.nanmax(self.freqs)) if self.freqs.size else np.nan}")
        print(f"{'Time resolution (s)':>24s} : {self.header['tsamp']}")
        if self.band_meta is not None:
            print(f"{'2CL bands':>24s} : {self.band_meta['raw']} {self.band_meta.get('unit') or ''}".rstrip())

    def grab_data(self, f_start=None, f_stop=None, t_start=None, t_stop=None, if_id=0, verbose=False, device="cpu"):
        start_total = time.time()
        t0 = 0 if t_start is None else int(t_start)
        t1 = self.selection_shape[self.time_axis] if t_stop is None else int(t_stop)
        t0 = max(0, min(t0, self.selection_shape[self.time_axis]))
        t1 = max(t0, min(t1, self.selection_shape[self.time_axis]))

        f0, f1 = self._freq_bounds_to_slice(f_start, f_stop, self.freqs)
        plot_f = self.freqs[f0:f1]
        raw = self._spec[self._t0 + t0:self._t0 + t1, self._f0 + f0:self._f0 + f1]
        plot_data = np.asarray(raw, dtype=np.float32)

        if verbose:
            print(f"CE4Waterfall.grab_data: {time.time() - start_total:.3f} seconds")
        return plot_f, plot_data

    def plot_waterfall(self, f_start=None, f_stop=None, t_start=None, t_stop=None, **kwargs):
        from matplotlib import pyplot as plt

        plot_f, plot_data = self.grab_data(f_start=f_start, f_stop=f_stop, t_start=t_start, t_stop=t_stop)
        if plot_data.size == 0:
            raise ValueError("No CE4 data selected for plotting.")

        vmin, vmax = np.nanpercentile(plot_data, [1, 99])
        if vmin == vmax:
            vmin, vmax = None, None

        plt.imshow(
            plot_data,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=[float(plot_f[0]), float(plot_f[-1]), 0, plot_data.shape[0] - 1],
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )
        plt.title(f"CE4 2C Waterfall: {Path(self.filename).name}")
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Record index")
        plt.colorbar(label="Science data")

    @staticmethod
    def _freq_bounds_to_slice(f_start, f_stop, freqs):
        if freqs.size == 0:
            return 0, 0
        if f_start is None and f_stop is None:
            return 0, freqs.size

        lo = float(np.nanmin(freqs)) if f_start is None else float(f_start)
        hi = float(np.nanmax(freqs)) if f_stop is None else float(f_stop)
        lo, hi = sorted([lo, hi])
        mask = (freqs >= lo) & (freqs <= hi)
        if not np.any(mask):
            raise ValueError(f"No frequency channels found in range [{lo}, {hi}] MHz")
        idx = np.where(mask)[0]
        return int(idx[0]), int(idx[-1]) + 1
