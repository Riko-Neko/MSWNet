import glob
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import numpy as np


CE4_RECORD_LEN = 8287
CE4_NCHANS = 2048
PDS_NS = {"pds": "http://pds.nasa.gov/pds4/pds/v1"}


def read_ce4_2c(path, nrec=None):
    """
    Return a memmap structured array for CE4 .2C files.
    mm["spec"] has shape (Nrec, 2048).
    """
    dt = np.dtype([
        ("frame_id", ("u1", 4)),
        ("version", "u1"),
        ("work_param", ("u1", 71)),
        ("solar_el", ">f4"),
        ("solar_az", ">f4"),
        ("cancel_period", ">u4"),
        ("cancel_num", ">u2"),
        ("accum_times", ">u2"),
        ("rec_len", ">u2"),
        ("spec", (">f4", CE4_NCHANS)),
        ("quality", "u1"),
    ], align=False)

    if dt.itemsize != CE4_RECORD_LEN:
        raise ValueError(f"dtype itemsize={dt.itemsize} != record_length={CE4_RECORD_LEN}")

    mm = np.memmap(path, dtype=dt, mode="r")
    if nrec is not None:
        mm = mm[:nrec]
    return mm


def match_2cl_for_2c(path_2c, data_dir=None):
    data_dir = os.path.dirname(path_2c) if data_dir is None else data_dir
    stem = os.path.splitext(os.path.basename(path_2c))[0]

    cands = glob.glob(os.path.join(data_dir, stem + "*.2CL")) + glob.glob(os.path.join(data_dir, stem + "*.xml"))
    cands = [x for x in cands if not os.path.basename(x).startswith("._")]
    if cands:
        return sorted(cands, key=lambda x: len(os.path.basename(x)))[0]

    all_labels = glob.glob(os.path.join(data_dir, "*.2CL")) + glob.glob(os.path.join(data_dir, "*.xml"))
    all_labels = [x for x in all_labels if not os.path.basename(x).startswith("._")]
    if not all_labels:
        return None

    toks_a = set(stem.split("_"))
    return max(all_labels, key=lambda x: len(toks_a & set(os.path.splitext(os.path.basename(x))[0].split("_"))))


def parse_ce4_time_from_2cl(path_2cl):
    root = ET.parse(path_2cl).getroot()
    start = root.find(".//pds:Time_Coordinates/pds:start_date_time", PDS_NS).text.strip()
    stop = root.find(".//pds:Time_Coordinates/pds:stop_date_time", PDS_NS).text.strip()

    def parse_z(ts):
        if ts.endswith("Z"):
            ts = ts[:-1]
        return datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)

    return parse_z(start), parse_z(stop)


def build_time_axis_from_2cl(path_2cl, nrec):
    start, stop = parse_ce4_time_from_2cl(path_2cl)
    total_sec = (stop - start).total_seconds()
    dt = total_sec / float(nrec) if nrec > 0 else 1.0

    start64 = np.datetime64(start.replace(tzinfo=None))
    step_ns = int(round(dt * 1e9))
    time_axis = start64 + np.arange(nrec, dtype=np.int64) * np.timedelta64(step_ns, "ns")
    return time_axis, float(dt)


def infer_dt_from_2cl(path_2cl, nrec):
    if path_2cl is None:
        return 1.0
    try:
        _, dt = build_time_axis_from_2cl(path_2cl, nrec)
        return float(dt) if dt > 0 else 1.0
    except Exception:
        return 1.0


def infer_dt_fs_from_2cl(path_2cl, nrec):
    dt = infer_dt_from_2cl(path_2cl, nrec)
    return dt, 1.0 / dt if dt > 0 else 1.0


def infer_freq_axis_from_2cl(path_2cl, nchans):
    if path_2cl is None:
        return np.arange(nchans, dtype=np.float64), None

    try:
        root = ET.parse(path_2cl).getroot()
        bands = root.find(".//pds:Instrument_Parm/pds:bands", PDS_NS)
        if bands is None or bands.text is None:
            return np.arange(nchans, dtype=np.float64), None

        raw = bands.text.strip()
        unit = (bands.attrib.get("unit") or "").strip().lower()
        unit_factor = {
            "hz": 1e-6,
            "khz": 1e-3,
            "mhz": 1.0,
            "ghz": 1e3,
        }.get(unit, 1.0)

        if "-" not in raw:
            return np.arange(nchans, dtype=np.float64), {"raw": raw, "unit": unit}

        lo, hi = [float(x.strip()) * unit_factor for x in raw.split("-", 1)]
        return np.linspace(lo, hi, nchans, dtype=np.float64), {"raw": raw, "unit": unit}
    except Exception:
        return np.arange(nchans, dtype=np.float64), None
