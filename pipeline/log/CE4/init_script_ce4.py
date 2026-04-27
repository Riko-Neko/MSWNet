import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[2]
DEFAULT_DST_DIR = ROOT_DIR / "data_process/post_process/filter_workflow/others/CE4"
CE4_TIME_RE = re.compile(r"_(\d{14})_(\d{14})_(\d{4}_[A-Za-z])$")


def parse_ce4_time_match(path: Path):
    match = CE4_TIME_RE.search(path.parent.name)
    if match is None:
        raise ValueError(f"Cannot parse CE4 time from parent directory: {path.parent.name}")
    return match


def parse_ce4_start_time(path: Path):
    match = parse_ce4_time_match(path)
    return datetime.strptime(match.group(1), "%Y%m%d%H%M%S")


def parse_ce4_time_id(path: Path):
    match = parse_ce4_time_match(path)
    return f"{match.group(1)}_{match.group(2)}_{match.group(3)}"


def merge_ce4_csvs(src_dir=None, dst_dir=None):
    src = SCRIPT_DIR if src_dir is None else Path(src_dir).resolve()
    dst = DEFAULT_DST_DIR if dst_dir is None else Path(dst_dir).resolve()
    dst.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_files = sorted(
        p for p in src.rglob("*.csv")
        if p.resolve().parent != dst and not p.name.startswith("CE4_")
    )
    if not csv_files:
        print(f"[\033[33mWarn\033[0m] No CSV files found in: {src}")
        return None

    frames = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        ce4_start = parse_ce4_start_time(csv_path)
        if "time_start" in df.columns:
            offsets = pd.to_numeric(df["time_start"], errors="coerce").fillna(0.0)
        else:
            offsets = pd.Series([0.0] * len(df), index=df.index)

        ce4_times = offsets.map(lambda seconds: ce4_start + timedelta(seconds=float(seconds)))
        df["ce4_time"] = ce4_times.map(lambda ts: ts.isoformat(timespec="microseconds"))
        df["ce4_time_id"] = parse_ce4_time_id(csv_path)
        df["_ce4_sort_time"] = ce4_times
        frames.append(df)

    if not frames:
        print(f"[\033[33mWarn\033[0m] CSV files found, but all are empty: {src}")
        return None

    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = merged.sort_values("_ce4_sort_time", kind="mergesort").drop(columns=["_ce4_sort_time"])

    out_file = dst / f"CE4_{timestamp}.csv"
    merged.to_csv(out_file, index=False)
    print(f"[\033[32mInfo\033[0m] Done. Merged {len(csv_files)} CSVs into: {out_file}")
    return out_file


if __name__ == "__main__":
    current_dir = SCRIPT_DIR
    dest_dir = DEFAULT_DST_DIR
    csv_files = sorted(current_dir.rglob("*.csv"))

    print(f"[\033[32mInfo\033[0m] Current dir:", current_dir)
    print(f"[\033[32mInfo\033[0m] CSV count:", len(csv_files))
    print("Destination dir:", dest_dir)
    input("Continue? (press any key)")
    merge_ce4_csvs(current_dir, dest_dir)
