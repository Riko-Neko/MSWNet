#!/usr/bin/env python3
import sys
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.CE4_utils import CE4Waterfall  # noqa: E402


def print_info(wf: CE4Waterfall):
    print("=== CE4Waterfall.info() Output ===")
    try:
        wf.info()
    except Exception as e:
        print(f"Error calling wf.info(): {e}")

    print("\n=== wf.header dict ===")
    hdr = getattr(wf, "header", None)
    if hdr is None:
        print("No wf.header attribute found.")
    else:
        for k, v in hdr.items():
            print(f"{k}: {v}")


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 6:
        print("Usage: python CE4_2C_checker.py <ce4_file.2C> [f_start] [f_stop] [t_start] [t_stop]")
        print("  f_start, f_stop: optional floats, frequency range in MHz")
        print("  t_start, t_stop: optional integers, CE4 record-index range")
        sys.exit(1)

    fname = sys.argv[1]
    f_start = float(sys.argv[2]) if len(sys.argv) >= 3 else None
    f_stop = float(sys.argv[3]) if len(sys.argv) >= 4 else None
    t_start = int(sys.argv[4]) if len(sys.argv) >= 5 else None
    t_stop = int(sys.argv[5]) if len(sys.argv) == 6 else None

    try:
        wf = CE4Waterfall(fname, f_start=f_start, f_stop=f_stop, t_start=t_start, t_stop=t_stop)
    except Exception as e:
        print(f"Error opening file {fname}: {e}")
        sys.exit(1)

    print_info(wf)
    try:
        plt.figure(figsize=(12, 6))
        wf.plot_waterfall()
        plt.show()
    except Exception as e:
        print(f"Error plotting waterfall for {fname}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
CE4 2C examples:
python data/CE4_2C_checker.py data/CE4/CE4_GRAS_LFRS-TR_SCI_P_20190103162200_20190103174300_0001_B.2C
python data/CE4_2C_checker.py data/CE4/CE4_GRAS_LFRS-TR_SCI_P_20190103162200_20190103174300_0001_B.2C 0.1 2.0
python data/CE4_2C_checker.py data/CE4/CE4_GRAS_LFRS-TR_SCI_P_20190103162200_20190103174300_0001_B.2C 0.1 2.0 0 256
python data/CE4_2C_checker.py data/CE4/CE4_GRAS_LFRS-TR_SCI_P_20190103162200_20190103174300_0001_B.2C 1.0 40.0
python data/CE4_2C_checker.py data/CE4/CE4_GRAS_LFRS-TR_SCI_P_20190506000000_20190506235959_0124_A.2C
python data/CE4_2C_checker.py data/CE4/CE4_GRAS_LFRS-TR_SCI_P_20190506000000_20190506235959_0124_A.2C 0.1 2.0
python data/CE4_2C_checker.py data/CE4/CE4_GRAS_LFRS-TR_SCI_P_20190506000000_20190506235959_0124_A.2C 1.0 40.0
python data/CE4_2C_checker.py data/CE4/CE4_GRAS_LFRS-TR_SCI_P_20190902000000_20190902235959_0243_A.2C
python data/CE4_2C_checker.py data/CE4/CE4_GRAS_LFRS-TR_SCI_P_20190902000000_20190902235959_0243_A.2C 0.1 2.0
python data/CE4_2C_checker.py data/CE4/CE4_GRAS_LFRS-TR_SCI_P_20190902000000_20190902235959_0243_A.2C 1.0 40.0

events:
python data/CE4_2C_checker.py data/CE4/CE4_GRAS_LFRS-TR_SCI_P_20211205160000_20211206040000_0211_B.2C 14.0 20.0 15170 15320
python data/CE4_2C_checker.py data/CE4/CE4_GRAS_LFRS-TR_SCI_P_20211231023000_20211231143000_0213_B.2C 14.0 20.0 21073 21219
python data/CE4_2C_checker.py data/CE4/CE4_GRAS_LFRS-TR_SCI_P_20190830160000_20190831040000_0056_B.2C 4.0 11.0 30708 30836

Paired 2CL label in data/CE4 is auto-matched:
data/CE4/CE4_GRAS_LFRS-TR_SCI_P_20190103162200_20190103174300_0001_B.2CL
data/CE4/CE4_GRAS_LFRS-TR_SCI_P_20190506000000_20190506235959_0124_A.2CL
data/CE4/CE4_GRAS_LFRS-TR_SCI_P_20190902000000_20190902235959_0243_A.2CL
"""
