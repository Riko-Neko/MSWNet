import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import Settings


@dataclass
class PeakPoint:
    t: int
    f: float
    v: float
    s: float


@dataclass
class TrackLine:
    t_start: float
    f_start: float
    t_end: float
    f_end: float
    slope: float
    intercept: float
    t_min: int
    t_max: int
    n_points: int
    coverage: float
    gap_count: int
    rmse: float
    peak_mean: float
    peak_sum: float
    score: float
    points_t: np.ndarray
    points_f: np.ndarray
    points_w: np.ndarray


@dataclass
class _TrackState:
    points_t: list[int] = field(default_factory=list)
    points_f: list[float] = field(default_factory=list)
    points_w: list[float] = field(default_factory=list)
    last_t: int = 0
    last_f: float = 0.0
    slope: float = 0.0
    miss_count: int = 0
    gap_count: int = 0

    def append(self, peak: PeakPoint):
        self.points_t.append(int(peak.t))
        self.points_f.append(float(peak.f))
        self.points_w.append(float(peak.v))
        if len(self.points_t) >= 2:
            dt = self.points_t[-1] - self.points_t[-2]
            if dt > 0:
                self.slope = (self.points_f[-1] - self.points_f[-2]) / dt
        self.last_t = int(peak.t)
        self.last_f = float(peak.f)
        self.miss_count = 0


class TrackLineDetector:
    """
    Lightweight line tracker for denoised dynamic spectra.

    Input:
        patch: (T, F) numpy array or torch tensor

    Output:
        list[TrackLine]
    """

    def __init__(self, peak=3.5, peak_dist=2, center=1, topk=3, link=4.0, gap=2, resid=2.0, min_len=6,
                 min_cover=0.06, max_rmse=2.0, line_dist=3.0, line_iou=0.5, eps=1e-8):
        self.peak = float(peak)
        self.peak_dist = int(peak_dist)
        self.center = int(center)
        self.topk = int(topk)
        self.link = float(link)
        self.gap = int(gap)
        self.resid = float(resid)
        self.min_len = int(min_len)
        self.min_cover = float(min_cover)
        self.max_rmse = float(max_rmse)
        self.line_dist = float(line_dist)
        self.line_iou = float(line_iou)
        self.eps = float(eps)

    def __call__(self, patch):
        return self.detect(patch)

    def detect(self, patch):
        x = self._to_numpy(patch)
        if x.ndim != 2:
            raise ValueError(f"TrackLineDetector expects a 2D patch, got shape {tuple(x.shape)}")
        T, F = x.shape
        if T < 2 or F < 2:
            return []
        debug_stats = {
            "reject_short": 0,
            "reject_fit": 0,
            "reject_refit": 0,
            "reject_filter": 0,
            "accept": 0,
            "nms_removed": 0,
        }

        z, norm_stats = self._normalize(x)
        if Settings.DEBUG:
            print(
                f"[\033[36mDebug\033[0m] TrackLine: shape={x.shape}, std={float(np.std(x)):.4f}, "
                f"med={float(norm_stats['med']):.4f}, mad={float(norm_stats['mad']):.4f}, "
                f"scale={float(norm_stats['scale']):.4f}, zmax={float(norm_stats['zmax']):.4f}"
            )
        peaks_by_row = [self._extract_row_peaks(z[t], t) for t in range(T)]
        rows_with_peaks = sum(1 for peaks in peaks_by_row if peaks)
        total_peaks = sum(len(peaks) for peaks in peaks_by_row)
        states = self._link_tracks(peaks_by_row)

        tracks = []
        for state in states:
            track = self._finalize_track(state, T, F, debug_stats)
            if track is not None:
                tracks.append(track)

        tracks.sort(key=lambda item: item.score, reverse=True)
        kept = self._line_nms(tracks, debug_stats)
        if Settings.DEBUG:
            print(
                f"[\033[36mDebug\033[0m] TrackLine: peak_rows={rows_with_peaks}/{T}, total_peaks={total_peaks}, "
                f"states={len(states)}, accept={debug_stats['accept']}, reject_short={debug_stats['reject_short']}, "
                f"reject_fit={debug_stats['reject_fit']}, reject_refit={debug_stats['reject_refit']}, "
                f"reject_filter={debug_stats['reject_filter']}, nms_removed={debug_stats['nms_removed']}, kept={len(kept)}"
            )
        return kept

    def _to_numpy(self, patch):
        if isinstance(patch, torch.Tensor):
            patch = patch.detach().float().cpu().numpy()
        x = np.asarray(patch, dtype=np.float32)
        if x.ndim == 4:
            x = x.squeeze(0).squeeze(0)
        elif x.ndim == 3:
            x = x.squeeze(0)
        return x

    def _normalize(self, patch):
        x = np.maximum(patch, 0.0).astype(np.float32)
        pos = x[x > 0]

        if pos.size == 0:
            return np.zeros_like(x, dtype=np.float32), {"med": 0.0, "mad": 0.0, "scale": 0.0, "zmax": 0.0}

        med = float(np.median(pos))
        mad = float(np.median(np.abs(pos - med)))
        scale = 1.4826 * mad

        if not np.isfinite(scale) or scale <= self.eps:
            scale = float(np.std(pos))
        if not np.isfinite(scale) or scale <= self.eps:
            q50, q95 = np.percentile(pos, [50, 95])
            scale = float(q95 - q50)
        if not np.isfinite(scale) or scale <= self.eps:
            scale = float(np.max(pos))
        if not np.isfinite(scale) or scale <= self.eps:
            return np.zeros_like(x, dtype=np.float32), {"med": med, "mad": mad, "scale": 0.0, "zmax": 0.0}

        z = x / (scale + self.eps)
        return z, {"med": med, "mad": mad, "scale": scale, "zmax": float(np.max(z))}

    def _extract_row_peaks(self, row, t):
        mask = row >= self.peak
        if not np.any(mask):
            return []

        edges = np.diff(mask.astype(np.int8))
        starts = np.where(edges == 1)[0] + 1
        stops = np.where(edges == -1)[0]
        if mask[0]:
            starts = np.r_[0, starts]
        if mask[-1]:
            stops = np.r_[stops, len(row) - 1]

        candidates = []
        for start, stop in zip(starts.tolist(), stops.tolist()):
            lo = max(0, int(start) - self.center)
            hi = min(len(row) - 1, int(stop) + self.center)
            xs = np.arange(lo, hi + 1, dtype=np.float32)
            ws = np.clip(row[lo:hi + 1], 0.0, None).astype(np.float32)
            if ws.sum() > self.eps:
                f_center = float((xs * ws).sum() / (ws.sum() + self.eps))
                local_sum = float(ws.sum())
                peak_val = float(ws.max())
            else:
                center_idx = 0.5 * (lo + hi)
                f_center = float(center_idx)
                local_sum = 0.0
                peak_val = 0.0
            candidates.append(PeakPoint(t=t, f=f_center, v=peak_val, s=local_sum))

        candidates.sort(key=lambda item: item.v, reverse=True)
        out = []
        min_dist = max(1, self.peak_dist)
        for item in candidates:
            if all(abs(item.f - prev.f) >= min_dist for prev in out):
                out.append(item)
            if len(out) >= max(1, self.topk):
                break

        out.sort(key=lambda item: item.f)
        return out

    def _link_tracks(self, peaks_by_row):
        active = []
        finished = []

        for t, peaks in enumerate(peaks_by_row):
            if not active:
                for peak in peaks:
                    state = _TrackState()
                    state.append(peak)
                    active.append(state)
                continue

            matched_tracks = set()
            matched_peaks = set()
            if active and peaks:
                big = 1e6
                cost = np.full((len(active), len(peaks)), big, dtype=np.float32)
                for i, state in enumerate(active):
                    f_pred = state.last_f + state.slope
                    for j, peak in enumerate(peaks):
                        dist = abs(peak.f - f_pred)
                        if dist <= self.link:
                            cost[i, j] = dist

                rows, cols = linear_sum_assignment(cost)
                for i, j in zip(rows.tolist(), cols.tolist()):
                    if cost[i, j] >= big:
                        continue
                    active[i].append(peaks[j])
                    matched_tracks.add(i)
                    matched_peaks.add(j)

            next_active = []
            for i, state in enumerate(active):
                if i in matched_tracks:
                    next_active.append(state)
                    continue
                state.miss_count += 1
                state.gap_count += 1
                if state.miss_count <= self.gap:
                    next_active.append(state)
                else:
                    finished.append(state)

            for j, peak in enumerate(peaks):
                if j in matched_peaks:
                    continue
                state = _TrackState()
                state.append(peak)
                next_active.append(state)

            active = next_active

        finished.extend(active)
        return finished

    def _finalize_track(self, state, T, F, debug_stats):
        if len(state.points_t) < max(2, self.min_len):
            if Settings.DEBUG:
                debug_stats["reject_short"] += 1
            return None

        t = np.asarray(state.points_t, dtype=np.float32)
        f = np.asarray(state.points_f, dtype=np.float32)
        w = np.asarray(state.points_w, dtype=np.float32)
        if not np.isfinite(w).all() or w.sum() <= self.eps:
            w = np.ones_like(t, dtype=np.float32)

        fit = self._fit_line(t, f, w)
        if fit is None:
            if Settings.DEBUG:
                debug_stats["reject_fit"] += 1
            return None
        slope, intercept = fit
        residual = f - (slope * t + intercept)
        keep = np.abs(residual) <= self.resid
        if keep.sum() >= 2 and keep.sum() < len(t):
            t_fit = t[keep]
            f_fit = f[keep]
            w_fit = w[keep]
            fit = self._fit_line(t_fit, f_fit, w_fit)
            if fit is None:
                if Settings.DEBUG:
                    debug_stats["reject_refit"] += 1
                return None
            slope, intercept = fit
        else:
            t_fit, f_fit, w_fit = t, f, w

        residual = f_fit - (slope * t_fit + intercept)
        rmse = float(np.sqrt(np.average(residual ** 2, weights=w_fit + self.eps)))
        n_points = int(len(t_fit))
        coverage = float(n_points / max(T, 1))
        if n_points < self.min_len or coverage < self.min_cover or rmse > self.max_rmse:
            if Settings.DEBUG:
                debug_stats["reject_filter"] += 1
            return None

        peak_mean = float(w_fit.mean())
        peak_sum = float(w_fit.sum())
        score = peak_mean * coverage / (1.0 + rmse) / (1.0 + state.gap_count)
        clipped = self._clip_line_segment(float(slope), float(intercept), T, F)
        if clipped is None:
            if Settings.DEBUG:
                debug_stats["reject_fit"] += 1
            return None
        t_start, f_start, t_end, f_end = clipped
        if Settings.DEBUG:
            debug_stats["accept"] += 1

        return TrackLine(
            t_start=t_start,
            f_start=f_start,
            t_end=t_end,
            f_end=f_end,
            slope=float(slope),
            intercept=float(intercept),
            t_min=int(t_fit.min()),
            t_max=int(t_fit.max()),
            n_points=n_points,
            coverage=coverage,
            gap_count=int(state.gap_count),
            rmse=rmse,
            peak_mean=peak_mean,
            peak_sum=peak_sum,
            score=float(score),
            points_t=t_fit.copy(),
            points_f=f_fit.copy(),
            points_w=w_fit.copy(),
        )

    def _clip_line_segment(self, slope, intercept, T, F):
        t_max = float(T - 1)
        f_max = float(F - 1)
        points = []

        def add_point(t_val, f_val):
            if not (np.isfinite(t_val) and np.isfinite(f_val)):
                return
            if t_val < -self.eps or t_val > t_max + self.eps:
                return
            if f_val < -self.eps or f_val > f_max + self.eps:
                return
            t_val = float(np.clip(t_val, 0.0, t_max))
            f_val = float(np.clip(f_val, 0.0, f_max))
            for prev_t, prev_f in points:
                if abs(prev_t - t_val) <= 1e-6 and abs(prev_f - f_val) <= 1e-6:
                    return
            points.append((t_val, f_val))

        add_point(0.0, intercept)
        add_point(t_max, slope * t_max + intercept)
        if abs(slope) > self.eps:
            add_point((-intercept) / slope, 0.0)
            add_point((f_max - intercept) / slope, f_max)

        if len(points) < 2:
            return None

        points.sort(key=lambda item: (item[0], item[1]))
        t_start, f_start = points[0]
        t_end, f_end = points[-1]
        return t_start, f_start, t_end, f_end

    def _fit_line(self, t, f, w):
        if len(t) < 2:
            return None
        sw = np.sqrt(np.clip(w, self.eps, None))
        A = np.stack([t, np.ones_like(t)], axis=1)
        Aw = A * sw[:, None]
        yw = f * sw
        try:
            coef, _, _, _ = np.linalg.lstsq(Aw, yw, rcond=None)
        except np.linalg.LinAlgError:
            return None
        return float(coef[0]), float(coef[1])

    def _line_nms(self, tracks, debug_stats):
        kept = []
        for track in tracks:
            duplicate = False
            for prev in kept:
                close_ends = (abs(track.f_start - prev.f_start) <= self.line_dist and
                              abs(track.f_end - prev.f_end) <= self.line_dist)
                curr_low = min(track.f_start, track.f_end)
                curr_high = max(track.f_start, track.f_end)
                prev_low = min(prev.f_start, prev.f_end)
                prev_high = max(prev.f_start, prev.f_end)
                inter = max(0.0, min(curr_high, prev_high) - max(curr_low, prev_low))
                curr_len = max(curr_high - curr_low, 0.0)
                prev_len = max(prev_high - prev_low, 0.0)
                union = curr_len + prev_len - inter + self.eps
                iou = inter / union
                high_iou = iou > self.line_iou
                if close_ends or high_iou:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(track)
            elif Settings.DEBUG:
                debug_stats["nms_removed"] += 1
        return kept


__all__ = ["PeakPoint", "TrackLine", "TrackLineDetector"]


if __name__ == "__main__":
    import os
    import sys
    from pathlib import Path

    from matplotlib import pyplot as plt

    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from gen.SETIdataset import DynamicSpectrumDataset

    tchans = 256
    fchans = 256
    df = 7.5
    dt = 1.0
    drift_min = -4.0
    drift_max = 4.0
    drift_min_abs = 0.0
    dataset = DynamicSpectrumDataset(mode='detection', tchans=tchans, fchans=fchans, df=df, dt=dt, fch1=None,
                                     ascending=True, drift_min=drift_min, drift_max=drift_max,
                                     drift_min_abs=drift_min_abs, snr_min=50.0, snr_max=60.0, width_min=10,
                                     width_max=15, num_signals=(1, 1), noise_std_min=0.025, noise_std_max=0.05)

    noisy_spec, clean_spec, gt_boxes = dataset[0]

    def add_gaussian_noise(spec, noise_level=0.1):
        noise = np.random.normal(loc=0.0, scale=noise_level, size=spec.shape)
        return spec + noise

    patch = add_gaussian_noise(clean_spec.squeeze(), noise_level=0.1)
    detector = TrackLineDetector()
    tracks = detector.detect(patch)

    out_dir = ROOT / "pipeline" / "log" / "trackline_test"
    os.makedirs(out_dir, exist_ok=True)
    out_path = Path(out_dir) / "trackline_test.png"

    print(f"[\033[32mInfo\033[0m] tracks={len(tracks)}")

    fig, ax = plt.subplots(figsize=(15, 3))
    vmin, vmax = np.nanpercentile(patch, [1, 99])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = None, None
    ax.imshow(patch, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    for i, track in enumerate(tracks):
        ax.plot([track.f_start, track.f_end], [track.t_start, track.t_end], "r--",
                linewidth=1.5, alpha=0.8, label="TrackLine fit" if i == 0 else None)
    if tracks:
        ax.legend()
    ax.set_title("TrackLine test")
    ax.set_xlabel("Frequency Channel")
    ax.set_ylabel("Time Channel")
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    print(f"[\033[32mInfo\033[0m] plot={out_path}")
