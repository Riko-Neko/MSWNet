#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CE4 visual checker.

Inputs:
- One merged CE4 CSV containing at least:
  freq_start, freq_end, time_start, time_end
  and preferably ce4_time_id.

- The CE4 observation path from the active Settings.CONFIG config.

Plots:
- One CE4 dynamic spectrum per CSV row.
- Frequency window pads left/right by 20% of the candidate frequency width.
- Time window pads top/bottom by 5% of the candidate time width.

Outputs:
./data_process/post_process/visual_val/visual/vis/CE4_<current_timestamp>/
  event_*.png
  event_meta.csv
"""

import argparse
import re
import sys
from math import ceil, floor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.configs import load_config  # noqa: E402
from config.settings import Settings  # noqa: E402
from utils.CE4_utils import CE4Waterfall  # noqa: E402

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None


DEFAULT_CSV_DIR = ROOT / "data_process/post_process/filter_workflow/others/CE4"
DEFAULT_OUTPUT_ROOT = ROOT / "data_process/post_process/visual_val/visual/vis"
CE4_TIME_ID_RE = re.compile(r"_(\d{14})_(\d{14})_(\d{4}_[A-Za-z])$")
F_PAD_FRAC = 0.20
T_PAD_FRAC = 0.05
DEFAULT_MIN_FCHANS = 64
DEFAULT_MIN_TCHANS = 64
DEFAULT_DPI = 300
DEFAULT_FMT = "png"
VMIN_PCT = 2.0
VMAX_PCT = 98.0

if Settings.PROD:
    import matplotlib as mpl

    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["font.size"] = 20
    mpl.rcParams["font.weight"] = "semibold"
    mpl.rcParams["axes.titleweight"] = "bold"
    mpl.rcParams["axes.labelweight"] = "bold"


def progress(iterable, total: int, desc: str):
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, unit="evt")

    def _gen():
        k = 0
        for item in iterable:
            k += 1
            if k == 1 or k % 5 == 0 or k == total:
                endc = "\n" if k == total else "\r"
                print(f"[\033[32mInfo\033[0m] {desc}: {k}/{total}", end=endc, flush=True)
            yield item

    return _gen()


def latest_ce4_csv(csv_dir: Path) -> Path:
    matches = sorted(csv_dir.glob("CE4_*.csv"), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No CE4_*.csv found in: {csv_dir}")
    return matches[-1].resolve()


def make_timestamp_dir(out_root: Path) -> Path:
    out_dir = out_root / f"CE4_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def collect_ce4_files(obs_file_path, obs_suffixes: List[str]) -> List[Path]:
    suffixes = {s.lower() for s in obs_suffixes}
    paths = obs_file_path if isinstance(obs_file_path, (list, tuple)) else [obs_file_path]

    files: List[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        if path.is_file() and path.suffix.lower() in suffixes and not path.name.startswith("._"):
            files.append(path.resolve())
        elif path.is_dir():
            files.extend(
                p.resolve()
                for p in path.rglob("*")
                if p.is_file() and p.suffix.lower() in suffixes and not p.name.startswith("._")
            )
    return sorted(set(files))


def parse_ce4_time_id(path: Path) -> Optional[str]:
    match = CE4_TIME_ID_RE.search(path.stem)
    if match is None:
        return None
    return f"{match.group(1)}_{match.group(2)}_{match.group(3)}"


def build_file_index(files: List[Path]) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for path in files:
        time_id = parse_ce4_time_id(path)
        if time_id is not None and time_id not in index:
            index[time_id] = path
    return index


def ensure_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def padded_bounds(row: pd.Series) -> Tuple[float, float, float, float]:
    f0 = float(row["freq_start"])
    f1 = float(row["freq_end"])
    t0 = float(row["time_start"])
    t1 = float(row["time_end"])

    f_lo, f_hi = sorted([f0, f1])
    t_lo, t_hi = sorted([t0, t1])
    f_width = max(f_hi - f_lo, 1e-9)
    t_width = max(t_hi - t_lo, 1e-9)

    return (
        f_lo - F_PAD_FRAC * f_width,
        f_hi + F_PAD_FRAC * f_width,
        max(0.0, t_lo - T_PAD_FRAC * t_width),
        max(0.0, t_hi + T_PAD_FRAC * t_width),
    )


def expand_freq_bounds_to_min_channels(
        f_start: float,
        f_stop: float,
        f_center: float,
        freqs: np.ndarray,
        min_fchans: int,
) -> Tuple[float, float]:
    if min_fchans <= 0 or freqs.size == 0:
        return f_start, f_stop

    lo, hi = sorted([float(f_start), float(f_stop)])
    current = int(np.count_nonzero((freqs >= lo) & (freqs <= hi)))
    if current >= min_fchans:
        return lo, hi

    window = min(int(min_fchans), int(freqs.size))
    center_idx = int(np.argmin(np.abs(freqs - float(f_center))))
    start = center_idx - window // 2
    stop = start + window
    if start < 0:
        start = 0
        stop = window
    if stop > freqs.size:
        stop = freqs.size
        start = max(0, stop - window)

    selected = freqs[start:stop]
    return float(np.nanmin(selected)), float(np.nanmax(selected))


def expand_time_indices_to_min_records(
        t_start_s: float,
        t_stop_s: float,
        t_center_s: float,
        tsamp: float,
        n_records: int,
        min_tchans: int,
) -> Tuple[int, int]:
    if n_records <= 0:
        return 0, 0

    tsamp = float(tsamp) or 1.0
    t0_idx = max(0, int(floor(float(t_start_s) / tsamp)))
    t1_idx = max(t0_idx + 1, int(ceil(float(t_stop_s) / tsamp)))
    t0_idx = max(0, min(t0_idx, n_records))
    t1_idx = max(t0_idx, min(t1_idx, n_records))
    if min_tchans <= 0 or (t1_idx - t0_idx) >= min_tchans:
        return t0_idx, t1_idx

    window = min(int(min_tchans), int(n_records))
    center_idx = int(round(float(t_center_s) / tsamp))
    center_idx = max(0, min(center_idx, n_records - 1))
    start = center_idx - window // 2
    stop = start + window
    if start < 0:
        start = 0
        stop = window
    if stop > n_records:
        stop = n_records
        start = max(0, stop - window)
    return start, stop


def robust_vmin_vmax(arr: np.ndarray) -> Tuple[float, float]:
    vals = np.asarray(arr)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0

    vmin = float(np.percentile(vals, VMIN_PCT))
    vmax = float(np.percentile(vals, VMAX_PCT))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        if vmax <= vmin:
            vmax = vmin + 1.0
    return vmin, vmax


def format_metric_block(row: pd.Series) -> str:
    specs = [
        ("SNR", "SNR", ".2f", ""),
        ("gSNR", "gSNR", ".2f", ""),
        ("confidence", "confidence", ".3f", ""),
        ("DriftRate", "DriftRate", ".4f", " Hz/s"),
    ]
    lines = []
    for label, col, fmt, unit in specs:
        if col not in row.index or pd.isna(row[col]):
            continue
        try:
            text = f"{float(row[col]):{fmt}}"
        except Exception:
            text = str(row[col])
        lines.append(f"{label}: {text}{unit}")
    return "\n".join(lines)


def plot_ce4_event(
        arr_tf: np.ndarray,
        freqs: np.ndarray,
        time_window: Tuple[float, float],
        event_freq: Tuple[float, float],
        event_time: Tuple[float, float],
        title: str,
        save_path: Path,
        dpi: int,
        fmt: str,
        metrics_text: str = "",
):
    if arr_tf.size == 0 or freqs.size == 0:
        raise ValueError("Empty CE4 data slice.")

    f_start = float(freqs[0])
    f_stop = float(freqs[-1])
    t_start, t_stop = time_window
    vmin, vmax = robust_vmin_vmax(arr_tf)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    im = ax.imshow(
        arr_tf,
        aspect="auto",
        origin="lower",
        extent=[f_start, f_stop, t_start, t_stop],
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )

    ef0, ef1 = sorted(event_freq)
    et0, et1 = sorted(event_time)
    ax.plot([ef0, ef1, ef1, ef0, ef0], [et0, et0, et1, et1, et0], color="red", linewidth=1.2)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Time (s)")
    if metrics_text:
        ax.text(
            0.015,
            0.985,
            metrics_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            color="white",
            fontsize=12,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "black", "alpha": 0.55, "edgecolor": "none"},
        )
    if not Settings.PROD:
        ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Intensity (arb.)")
    fig.tight_layout()
    fig.savefig(save_path.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def safe_str(value) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def main():
    ap = argparse.ArgumentParser(description="Render CE4 visual checks around merged candidate CSV events.")
    ap.add_argument("--target_csv", type=str, default=None, help="Merged CE4 CSV. Default: latest CE4_*.csv.")
    ap.add_argument("--out_root", type=str, default=str(DEFAULT_OUTPUT_ROOT), help="Output root directory.")
    ap.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="Output image DPI.")
    ap.add_argument("--fmt", type=str, default=DEFAULT_FMT, choices=["png", "jpg", "jpeg", "pdf"], help="Image format.")
    ap.add_argument("--min_fchans", type=int, default=DEFAULT_MIN_FCHANS, help="Minimum plotted frequency channels.")
    ap.add_argument("--min_tchans", type=int, default=DEFAULT_MIN_TCHANS, help="Minimum plotted time channels.")
    ap.add_argument("--max_events", type=int, default=0, help="If >0, only process first N events.")
    args = ap.parse_args()

    config = load_config()
    target_csv = Path(args.target_csv).expanduser().resolve() if args.target_csv else latest_ce4_csv(DEFAULT_CSV_DIR)
    out_root = Path(args.out_root).expanduser().resolve()
    obs_file_path = config["obs_file_path"]
    obs_suffixes = config.get("obs_suffixes", [".2C"])

    if not target_csv.exists():
        print(f"[\033[31mError\033[0m] Target CSV not found: {target_csv}")
        sys.exit(1)

    ce4_files = collect_ce4_files(obs_file_path, obs_suffixes)
    if not ce4_files:
        print(f"[\033[31mError\033[0m] No CE4 files found in obs_file_path: {obs_file_path}")
        sys.exit(1)
    file_index = build_file_index(ce4_files)

    df = pd.read_csv(target_csv)
    required = ["freq_start", "freq_end", "time_start", "time_end"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"[\033[31mError\033[0m] Missing required columns: {missing}")
        print(f"[\033[31mError\033[0m] Present columns: {list(df.columns)}")
        sys.exit(1)
    if "ce4_time_id" not in df.columns:
        print("[\033[31mError\033[0m] Missing required CE4 matching column: ce4_time_id")
        sys.exit(1)

    df = ensure_numeric(df, required)
    before = len(df)
    df = df.dropna(subset=required + ["ce4_time_id"]).copy()
    dropped = before - len(df)
    if dropped:
        print(f"[\033[32mInfo\033[0m] Dropped {dropped} rows due to missing required values.")
    if args.max_events and args.max_events > 0:
        df = df.head(args.max_events).copy()

    out_dir = make_timestamp_dir(out_root)
    meta_rows: List[Dict] = []

    print("\n==============================")
    print("[\033[32mInfo\033[0m] CE4 VISUAL CHECK")
    print("==============================")
    print(f"[\033[32mInfo\033[0m] Config     : {Settings.CONFIG}")
    print(f"[\033[32mInfo\033[0m] Target CSV : {target_csv}")
    print(f"[\033[32mInfo\033[0m] CE4 files  : {len(ce4_files)}")
    print(f"[\033[32mInfo\033[0m] Output dir : {out_dir}")
    print(f"[\033[32mInfo\033[0m] Format/DPI : {args.fmt}/{args.dpi}")

    for idx, row in progress(df.reset_index(drop=True).iterrows(), total=len(df), desc="Events"):
        ce4_time_id = str(row["ce4_time_id"])
        ce4_file = file_index.get(ce4_time_id)
        meta = {
            "event_idx": idx,
            "ce4_time_id": ce4_time_id,
            "ce4_file": str(ce4_file) if ce4_file else "",
            "rendered": "0",
            "csv_source": str(target_csv),
            "freq_start": safe_str(row["freq_start"]),
            "freq_end": safe_str(row["freq_end"]),
            "time_start": safe_str(row["time_start"]),
            "time_end": safe_str(row["time_end"]),
            "ce4_time": safe_str(row["ce4_time"]) if "ce4_time" in df.columns else "",
            "DriftRate": safe_str(row["DriftRate"]) if "DriftRate" in df.columns else "",
            "SNR": safe_str(row["SNR"]) if "SNR" in df.columns else "",
            "gSNR": safe_str(row["gSNR"]) if "gSNR" in df.columns else "",
            "class_id": safe_str(row["class_id"]) if "class_id" in df.columns else "",
            "confidence": safe_str(row["confidence"]) if "confidence" in df.columns else "",
        }

        if ce4_file is None:
            print(f"[\033[33mWarn\033[0m] No CE4 file matched ce4_time_id={ce4_time_id}. Skipped.")
            meta_rows.append(meta)
            continue

        try:
            f0, f1, t0_s, t1_s = padded_bounds(row)
            base_wf = CE4Waterfall(ce4_file, load_data=False)
            tsamp = float(base_wf.header.get("tsamp", 1.0)) or 1.0
            f_center = 0.5 * (float(row["freq_start"]) + float(row["freq_end"]))
            t_center_s = 0.5 * (float(row["time_start"]) + float(row["time_end"]))
            f0, f1 = expand_freq_bounds_to_min_channels(f0, f1, f_center, base_wf.get_freqs(), args.min_fchans)
            t0_idx, t1_idx = expand_time_indices_to_min_records(
                t0_s,
                t1_s,
                t_center_s,
                tsamp,
                base_wf.n_ints_in_file,
                args.min_tchans,
            )

            wf = CE4Waterfall(ce4_file, f_start=f0, f_stop=f1, t_start=t0_idx, t_stop=t1_idx, load_data=False)
            freqs, data = wf.grab_data()
            time_window = (t0_idx * tsamp, (t0_idx + data.shape[0]) * tsamp)

            save_base = (
                f"event_{idx:06d}_{ce4_time_id}_"
                f"f{float(row['freq_start']):.6f}-{float(row['freq_end']):.6f}_"
                f"t{float(row['time_start']):.2f}-{float(row['time_end']):.2f}"
            )
            title = f"CE4 | evt={idx} | {ce4_time_id}"
            save_path = out_dir / save_base
            plot_ce4_event(
                arr_tf=data,
                freqs=freqs,
                time_window=time_window,
                event_freq=(float(row["freq_start"]), float(row["freq_end"])),
                event_time=(float(row["time_start"]), float(row["time_end"])),
                title=title,
                save_path=save_path,
                dpi=args.dpi,
                fmt=args.fmt,
                metrics_text=format_metric_block(row),
            )

            meta.update({
                "rendered": "1",
                "plot_file": str(save_path.with_suffix(f".{args.fmt}")),
                "plot_freq_start": f0,
                "plot_freq_end": f1,
                "plot_time_start": time_window[0],
                "plot_time_end": time_window[1],
                "t_start_idx": t0_idx,
                "t_stop_idx": t0_idx + data.shape[0],
                "plot_fchans": data.shape[1],
                "plot_tchans": data.shape[0],
            })
        except Exception as e:
            print(f"[\033[33mWarn\033[0m] Failed event_idx={idx}, ce4_time_id={ce4_time_id}: {e}")
            meta["error"] = str(e)

        meta_rows.append(meta)

    meta_path = out_dir / "event_meta.csv"
    pd.DataFrame(meta_rows).to_csv(meta_path, index=False)
    print(f"\n[\033[32mInfo\033[0m] Metadata written: {meta_path}")
    print("[\033[32mInfo\033[0m] Done!")


if __name__ == "__main__":
    main()
