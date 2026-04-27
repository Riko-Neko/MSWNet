import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PyQt5.QtWidgets import QApplication
from torch.utils.data import DataLoader

from config.configs import load_config
from config.settings import Settings
from gen.SETIdataset import DynamicSpectrumDataset
from model.DetMSWNet import MSWNet as DetMSWNet
from model.MSWNet import MSWNet as DenoMSWNet
from model.UNet import UNet
from model.utils.TrackLine import TrackLineDetector
from pipeline.patch_engine import SETIWaterFullDataset
from pipeline.pipeline_processor import SETIPipelineProcessor
from pipeline.renderer import SETIWaterfallRenderer
from utils.pred_core import pred

_config = load_config()

pmode = _config["pmode"]
patch_t = _config["patch_t"]
patch_f = _config["patch_f"]
overlap_pct = _config["overlap_pct"]
t_adaptive = _config["t_adaptive"]
adaptive_scale = _config["adaptive_scale"]
tchans = _config["tchans"]
fchans = _config["fchans"]
df = _config["df"]
dt = _config["dt"]
fch1 = _config["fch1"]
ascending = _config["ascending"]
drift_min = _config["drift_min"]
drift_max = _config["drift_max"]
drift_min_abs = _config["drift_min_abs"]
snr_min = _config["snr_min"]
snr_max = _config["snr_max"]
width_min = _config["width_min"]
width_max = _config["width_max"]
num_signals = _config["num_signals"]
noise_std_min = _config["noise_std_min"]
noise_std_max = _config["noise_std_max"]
noise_mean_min = _config["noise_mean_min"]
noise_mean_max = _config["noise_mean_max"]
nosie_type = _config["nosie_type"]
rfi_enhance = _config["rfi_enhance"]
use_fil = _config["use_fil"]
fil_folder = _config["fil_folder"]
background_fil = _config["background_fil"]

ignore_polarization = _config["ignore_polarization"]
stokes_mode = _config["stokes_mode"]
XX_dir = _config["XX_dir"]
YY_dir = _config["YY_dir"]
Beam = _config["Beam"]

obs_file_path = _config["obs_file_path"]
obs_suffixes = _config["obs_suffixes"]

RAW = _config["RAW"]
batch_size = _config["batch_size"]
num_workers = _config["num_workers"]
pred_dir = _config["pred_dir"]
pred_steps = _config["pred_steps"]
dwtnet_ckpt = _config["dwtnet_ckpt"]
unet_ckpt = _config["unet_ckpt"]
P = _config["P"]

nms_kargs = _config["nms_kargs"]

drift = _config["drift"]
snr_threshold = _config["snr_threshold"]
pad_fraction = _config["pad_fraction"]
fsnr_args = _config["fsnr_args"]
dedrift_args = _config["dedrift_args"]

dwtnet_args = _config["dwtnet_args"]
unet_args = _config["unet_args"]
detect_backend = _config["detect_backend"]
detector_args = _config["detector_args"]
trackline_args = _config["trackline_args"]

del _config


def main(mode=None, ui=False, obs=False, verbose=False, device=None, *args):
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    global batch_size, ignore_polarization, t_adaptive, obs_suffixes
    if batch_size != 1:
        print(f"[\033[31mSevere Warn\033[0m] !!!! Batch size is fixed to 1 for now, cannot use batch_size > 1 !!!!")
        batch_size = 1
    if Settings.WORKFLOW == "CE4":
        print(
            "[\033[33mWarn\033[0m] CE4 workflow enabled: using CE4Waterfall adapter as a reuse path; forcing ignore_polarization=False, t_adaptive=False, and obs_suffixes=['.2c']."
        )
        ignore_polarization = False
        obs_suffixes = [".2c"]

    # Set device
    def check_device(dev):
        try:
            if dev.type == "cuda":
                return torch.cuda.is_available()
            elif dev.type == "mps":
                return torch.backends.mps.is_available() and torch.backends.mps.is_built()
            elif dev.type == "cpu":
                return True
            else:
                return False
        except Exception:
            return False

    def match_polarization_files(files, M_list=None):
        """
        Match polarization files only for selected beams M_list.
        Example M_list = [1,2,3] -> match M01, M02, M03

        Args:
            files: list of Path objects
            M_list: list of integers for beam selection

        Returns:
            matched_groups (sorted), unmatched (sorted)
        """
        from collections import defaultdict

        groups = defaultdict(list)
        unmatched = []

        if M_list is not None:
            # Convert [8,10] -> ["M08","M10"]
            allowed_M = [f"M{m:02d}" for m in M_list]
            allowed_M_set = set(allowed_M)
            # Priority map: lower index = higher priority
            M_priority = {m: i for i, m in enumerate(allowed_M)}
            print(f"[\033[32mInfo\033[0m] Selected beams (ordered): {allowed_M}")
        else:
            allowed_M = None
            allowed_M_set = None
            M_priority = None
            print("[\033[32mInfo\033[0m] No beam filtering applied.")

        for file_path in files:
            stem = file_path.stem

            if "_pol" not in stem:
                unmatched.append(str(file_path))
                continue

            try:
                parts = stem.split("_")
                beam_name = next(p for p in parts if p.startswith("M") and len(p) == 3)
            except StopIteration:
                unmatched.append(str(file_path))
                continue

            if allowed_M_set is not None and beam_name not in allowed_M_set:
                continue

            base = stem.split("_pol")[0]
            groups[(beam_name, base)].append(str(file_path))

        matched_groups = []
        for (beam_name, base), group in groups.items():
            if len(group) > 1:
                matched_groups.append((beam_name, group))
            else:
                unmatched.extend(group)

        if M_priority is not None:
            matched_groups = sorted(matched_groups, key=lambda x: M_priority[x[0]])
        else:
            matched_groups = sorted(matched_groups, key=lambda x: int(x[0][1:]))

        matched_groups = [g for (_, g) in matched_groups]

        return matched_groups, sorted(unmatched)

    def load_model(model_class, checkpoint_path, **kwargs):
        model = model_class(**kwargs).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()
        return model

    if device is not None:
        try:
            device = torch.device(device)
            if not check_device(device):
                print(f"[\033[33mWarn\033[0m] Device '{device}' is not available. Fallback to default...")
                device = None
        except Exception as e:
            print(f"[\033[33mWarn\033[0m] Invalid device argument ({device}): {e}. Fallback to default...")
            device = None

    if device is None:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    print(f"\n[\033[32mInfo\033[0m] Using device: {device}")

    active_detect_backend = detect_backend
    if pmode != "detection":
        active_detect_backend = "regressor"

    def get_msw_model_spec(backend):
        if backend == "trackline":
            return DenoMSWNet, dict(dwtnet_args)
        return DetMSWNet, {**dwtnet_args, **detector_args}

    trackline_detector = None
    if active_detect_backend == "trackline":
        trackline_detector = TrackLineDetector(**trackline_args, line_iou=nms_kargs["iou_thresh"])

    # Create datasets based on mode and obs flag
    if obs and mode != "pipeline":
        if ignore_polarization:
            if not isinstance(obs_file_path, list):
                raise ValueError(
                    "In observation mode ignoring polarization, observation data should be a list of [pol1_dir, pol2_dir, ...].")
            else:
                matched, _ = match_polarization_files(
                    sorted([f for f in Path(obs_file_path[0]).iterdir() if f.suffix.lower() in obs_suffixes]) + sorted(
                        [f for f in Path(obs_file_path[1]).iterdir() if f.suffix.lower() in obs_suffixes]), M_list=Beam)
                obs_file_1st = matched[0]
        else:
            if isinstance(obs_file_path, list):
                raise ValueError("In non-pipeline mode, observation data path should be a file, not a list.")
            else:
                if Path(obs_file_path).is_dir():
                    raise ValueError("In non-pipeline mode, observation data path should be a file, not a directory.")
                obs_file_1st = obs_file_path
        # Use pipeline dataset for obs mode
        print("[\033[32mInfo\033[0m] Using observation data from:", obs_file_1st)
        dataset = SETIWaterFullDataset(file_path=obs_file_1st, patch_t=patch_t, patch_f=patch_f,
                                       overlap_pct=overlap_pct, ignore_polarization=ignore_polarization,
                                       stokes_mode=stokes_mode, t_adaptive=t_adaptive,
                                       adaptive_scale=adaptive_scale)
        pred_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    else:

        pred_dataset = DynamicSpectrumDataset(mode=pmode, tchans=tchans, fchans=fchans, df=df, dt=dt, fch1=fch1,
                                              ascending=ascending, drift_min=drift_min, drift_max=drift_max,
                                              drift_min_abs=drift_min_abs, snr_min=snr_min, snr_max=snr_max,
                                              width_min=width_min, width_max=width_max, num_signals=num_signals,
                                              noise_std_min=noise_std_min, noise_std_max=noise_std_max,
                                              noise_mean_min=noise_mean_min, noise_mean_max=noise_mean_max,
                                              noise_type=nosie_type, rfi_enhance=rfi_enhance, use_fil=use_fil,
                                              background_fil=background_fil)
        pred_dataloader = DataLoader(pred_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    if mode == "dbl":
        global pred_dir
        pred_dir = Path(pred_dir) / "dbl"
        print("[\033[32mInfo\033[0m] Running dual-model comparison mode")
        # Load both models
        msw_model_class, msw_model_kwargs = get_msw_model_spec(active_detect_backend)
        dwtnet = load_model(msw_model_class, dwtnet_ckpt, **msw_model_kwargs)
        unet = load_model(UNet, unet_ckpt)
        # Process the same samples with both models
        for idx, batch in enumerate(pred_dataloader):
            if idx >= pred_steps:
                break
            print(f"[\033[32mInfo\033[0m] Processing sample {idx + 1}/{pred_steps}")
            print("[\033[32mInfo\033[0m] Running MSWNet inference...")
            pred(dwtnet, data_mode='dbl', mode=pmode, data=batch, idx=idx, save_dir=pred_dir, device=device,
                 save_npy=False, plot=True, detect_backend=active_detect_backend,
                 trackline_detector=trackline_detector, group_nms=nms_kargs, group_fsnr=fsnr_args,
                 group_dedrift=dedrift_args)
            print("[\033[32mInfo\033[0m] Running UNet inference...")
            pred(unet, data_mode='dbl', mode=pmode, data=batch, idx=idx, save_dir=pred_dir, device=device,
                 save_npy=False, plot=True, detect_backend=active_detect_backend,
                 trackline_detector=trackline_detector, group_nms=nms_kargs, group_fsnr=fsnr_args,
                 group_dedrift=dedrift_args)


    elif mode == "pipeline":
        print("[\033[32mInfo\033[0m] Running pipeline processing mode")
        app = None
        if ui:
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)

        all_files = []
        if ignore_polarization:
            # When True, handle polarization matching
            if isinstance(obs_file_path, str) and Path(obs_file_path).is_file():
                print(
                    f"[\033[31mError\033[0m] When ignoring polarization, observation data cannot be a single file: {obs_file_path}")
                sys.exit(1)  # Or raise error

            print("[\033[32mInfo\033[0m] Ignoring polarization: matching files for intensity stacking")
            if isinstance(obs_file_path, list) and len(obs_file_path) == 2:

                # [XX_dir, YY_dir]
                xx_dir = Path(obs_file_path[0])
                yy_dir = Path(obs_file_path[1])
                if not (xx_dir.is_dir() and yy_dir.is_dir()):
                    print(
                        f"[\033[31mError\033[0m] Both elements in observation data path must be directories when ignoring polarization: {obs_file_path}")
                    sys.exit(1)

                xx_files = sorted([f for f in xx_dir.iterdir() if f.suffix.lower() in obs_suffixes])
                yy_files = sorted([f for f in yy_dir.iterdir() if f.suffix.lower() in obs_suffixes])

                # Match by base name, assuming xx_files have _pol1, yy have _pol2
                all_files = xx_files + yy_files

            elif isinstance(obs_file_path, str) and Path(obs_file_path).is_dir():
                # Single directory, collect all files
                obs_path = Path(obs_file_path)
                all_files = sorted([f for f in obs_path.iterdir() if f.suffix.lower() in obs_suffixes])
            else:
                print(
                    f"[\033[31mError\033[0m] Invalid observation data path format: {obs_file_path}")
                sys.exit(1)

            if not all_files:
                print(f"[\033[31mError\033[0m] No .fil or .h5 files found in provided paths: {obs_file_path}")
                sys.exit(1)

            file_groups, unmatched = match_polarization_files(all_files, M_list=Beam)
            if unmatched:
                print(
                    f"[\033[33mWarning\033[0m] Unmatched files (not paired or no _pol* pattern): {', '.join(unmatched)}")
            # file_list is now list of lists (groups)
            file_list = file_groups  # Only process matched groups

        else:
            if isinstance(obs_file_path, list):
                print(
                    f"[\033[31mError\033[0m] When ignoring polarization, observation data path must be a file or directory, not a list: {obs_file_path}")
                sys.exit(1)
            obs_path = Path(obs_file_path)
            if obs_path.is_dir():
                file_list = sorted([f for f in obs_path.iterdir() if f.suffix.lower() in obs_suffixes])
                if not file_list:
                    print(f"[\033[31mError\033[0m] No .fil or .h5 files found in directory: {obs_path}")
                    sys.exit(1)
            else:
                file_list = [obs_path]

        for idx, f in enumerate(file_list):
            # f could be Path or list[str]
            if isinstance(f, list):
                print(f"[\033[32mInfo\033[0m] Processing polarization group: {', '.join([Path(p).name for p in f])}")
                file_path_for_dataset = f  # list[str]
                f_log_dir = Path(f[0]).stem.split('_pol')[0]  # Use base name for log dir
            else:
                print(f"[\033[32mInfo\033[0m] Processing file: {f}")
                file_path_for_dataset = str(f)
                f_log_dir = f.stem
            if Settings.WORKFLOW == "CE4":
                f_log_dir = Path("CE4") / f_log_dir
            dataset = SETIWaterFullDataset(file_path=file_path_for_dataset, patch_t=patch_t, patch_f=patch_f,
                                           overlap_pct=overlap_pct, device=device,
                                           ignore_polarization=ignore_polarization, stokes_mode=stokes_mode,
                                           t_adaptive=t_adaptive, adaptive_scale=adaptive_scale)

            # Load model
            msw_model_class, msw_model_kwargs = get_msw_model_spec(active_detect_backend)
            model = load_model(msw_model_class, dwtnet_ckpt, **msw_model_kwargs)

            if ui:
                if RAW:
                    print("[\033[33mWarn\033[0m] UI mode cannot be used with RAW output, using original config...")
                renderer = SETIWaterfallRenderer(dataset, model, device, mode=pmode, log_dir=f_log_dir, drift=drift,
                                                 snr_threshold=snr_threshold, min_abs_drift=drift_min_abs,
                                                 verbose=verbose, detect_backend=active_detect_backend,
                                                 trackline_detector=trackline_detector, **nms_kargs, **fsnr_args)
                renderer.setWindowTitle(f"SETI Waterfall Data Processor - {f_log_dir}")
                renderer.show()
                if idx == len(file_list) - 1:
                    sys.exit(app.exec_())
                else:
                    app.exec_()

            else:
                print("[\033[32mInfo\033[0m] Running in no-UI mode, logging only")
                if RAW:
                    print(
                        "[\033[33mWarn\033[0m] You are logging raw data, which may be extremely large. Make sure you have enough space.")
                processor = SETIPipelineProcessor(dataset, model, device, mode=pmode, log_dir=f_log_dir,
                                                  raw_output=RAW, drift=drift, snr_threshold=snr_threshold,
                                                  pad_fraction=pad_fraction, min_abs_drift=drift_min_abs,
                                                  verbose=verbose, detect_backend=active_detect_backend,
                                                  trackline_detector=trackline_detector, **nms_kargs, **fsnr_args)
                processor.process_all_patches()

    else:
        print("[\033[32mInfo\033[0m] Running single-model mode")
        execute0, execute1 = args
        # --- 推理 MSWNet ---
        if execute0:
            print("[\033[32mInfo\033[0m] Running MSWNet inference...")
            msw_model_class, msw_model_kwargs = get_msw_model_spec(active_detect_backend)
            dwtnet = load_model(msw_model_class, dwtnet_ckpt, **msw_model_kwargs)
            pred(dwtnet, mode=pmode, data=pred_dataloader, save_dir=pred_dir, device=device, max_steps=pred_steps,
                 save_npy=False, plot=True, detect_backend=active_detect_backend,
                 trackline_detector=trackline_detector, group_nms=nms_kargs, group_fsnr=fsnr_args,
                 group_dedrift=dedrift_args)
        # --- 推理 UNet ---
        if execute1:
            print("[\033[32mInfo\033[0m] Running UNet inference...")
            unet = load_model(UNet, unet_ckpt, **unet_args)
            pred(unet, mode=pmode, data=pred_dataloader, save_dir=pred_dir, device=device, max_steps=pred_steps,
                 save_npy=False, plot=True, detect_backend=active_detect_backend,
                 trackline_detector=trackline_detector, group_nms=nms_kargs, group_fsnr=fsnr_args,
                 group_dedrift=dedrift_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select the run mode")
    parser.add_argument("--mode",
                        type=str,
                        choices=["dbl", "pipeline"],
                        help="Run mode: no argument for single-model pred, 'dbl' for dual-model comparison, 'pipeline' for pipeline processing")
    parser.add_argument("--ui",
                        action="store_true",
                        default=False,
                        help="Run pipeline in UI mode")
    parser.add_argument("--obs",
                        action="store_true",
                        default=False,
                        help="Use observation data file for default and dbl modes")
    parser.add_argument("--verbose",
                        action="store_true",
                        default=False,
                        help="Use verbose output for pipeline mode")
    parser.add_argument("-d", "--device",
                        type=str,
                        default=None,
                        help="Device to use for inference (e.g. 'cuda:0', 'cpu', 'mps')")
    args = parser.parse_args()

    if args.mode is None:
        main(None, args.ui, args.obs, args.verbose, args.device, True, False)
    elif args.mode == "dbl":
        main("dbl", args.ui, args.obs, args.verbose, args.device)
    elif args.mode == "pipeline":
        main("pipeline", args.ui, args.obs, args.verbose, args.device)
