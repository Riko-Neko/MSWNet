from copy import deepcopy
from pathlib import Path

from config.settings import Settings

CONFIGS = {
    "default": {
        # Data config
        "tchans": 116,
        "fchans": 256,
        "df": 7.450580597,
        "dt": 10.200547328,
        "fch1": None,
        "ascending": True,
        "drift_min": -4.0,
        "drift_max": 4.0,
        "drift_min_abs": 7.450580597 // (116 * 10.200547328),
        "snr_min": 15,
        "snr_max": 35,
        "width_min": 10,
        "width_max": 30,
        "num_signals": (0, 1),
        "noise_std_min": 0.025,
        "noise_std_max": 0.05,
        "noise_mean_min": 2,
        "noise_mean_max": 3,
        "nosie_type": "chi2",
        "rfi_enhance": False,
        "use_fil": True,
        "fil_folder": Path("./data/33exoplanets"),
        "background_fil": list(Path("./data/33exoplanets").rglob("*.fil")),

        # Training config
        "num_epochs": 1000,
        "steps_per_epoch": 200,
        "valid_interval": 1,
        "valid_steps": 50,
        "log_interval": 50,
        "lr": 0.001,
        "weight_decay": 1.e-9,
        "T_max": 30,
        "eta_min": 1.0e-15,
        "force_save_best": True,
        "freeze_backbone": True,
        "force_reconstruct": False,
        "mismatch_load": True,
        "checkpoint_dir": "./checkpoints/mswunet/bin256",
        "N": 5,

        # Model config
        "dim": 64,
        "levels": [2, 4, 8, 16],
        "feat_channels": 64,
        "dwtnet_args": {
            "in_chans": 1,
            "dim": 64,
            "levels": [2, 4, 8, 16],
            "wavelet_name": "db4",
            "extension_mode": "periodization",
        },
        "detector_args": {
            "fchans": 256,
            "N": 5,
            "num_classes": 2,
            "feat_channels": 64,
            "dropout": 0.005,
        },
        "detect_backend": "regressor",
        "regress_loss_args": {
            "lambda_denoise": 0.,
            "loss_type": "mse",
            "lambda_learnable": False,
            "regression_loss_kwargs": {
                "num_classes": 2,
                "N": 5,
                "w_loc": 1.5,
                "w_class": 0.1,
                "w_conf": 0.5,
                "eps": 1e-8,
            },
        },
        "trackline_args": {
            "peak": 10,
            "peak_dist": 3,
            "center": 5,
            "topk": 3,
            "link": 5.0,
            "gap": 100,
            "resid": 2.0,
            "min_len": 30,
            "min_cover": 0.03,
            "max_rmse": 2.5,
            "line_dist": 3.0,
        },
        "mask_loss_args": {
            "alpha": 1.0,
            "beta": 0.,
            "gamma": 0.,
            "delta": 0.,
            "momentum": 0.99,
            "fixed_g_d": True,
        },
        "unet_args": {},

        # Prediction config
        "patch_t": 116,
        "patch_f": 256,
        "overlap_pct": 0.02,
        "t_adaptive": True,
        "adaptive_scale": None,
        "pmode": "detection",
        "RAW": False,
        "batch_size": 1,
        "num_workers": 0,
        "pred_dir": "./pred_results",
        "pred_steps": 9999999,
        "dwtnet_ckpt": Path("./checkpoints/mswunet/bin256") / "final.pth",
        "unet_ckpt": Path("./checkpoints/unet") / "best_model.pth",
        "P": 2,

        # Polarization config
        "ignore_polarization": True,
        "stokes_mode": "I",
        "XX_dir": "./data/33exoplanets/xx/",
        "YY_dir": "./data/33exoplanets/yy/",
        "Beam": None,

        # Observation data
        "obs_file_path": ["./data/33exoplanets/xx/", "./data/33exoplanets/yy/"],
        "obs_suffixes": [".fil", ".h5"],

        # NMS config
        "nms_kargs": {
            "iou_thresh": 0.5,
            "score_thresh": 0.0,
        },

        # Hits config
        "drift": [-4.0, 4.0],
        "snr_threshold": 5.0,
        "pad_fraction": 0.5,
        "fsnr_args": {
            "fsnr_threshold": 300,
            "top_fraction": 0.001,
            "min_pixels": 50,
        },
        "dedrift_args": {
            "df_hz": 7.450580597,
            "dt_s": 10.200547328,
            "guard_bins": 3,
        },
    },

    "quick_start": {
        # synthetic path
        "use_fil": False,
        "background_fil": [],
        "checkpoint_dir": "./checkpoints/<path-to-your-output-checkpoints>",

        # local observation quick-start path
        "dwtnet_ckpt": Path("./checkpoints/<path-to-your-mswnet-weights>") / "mswnet-bin256-final.pth",
        "unet_ckpt": Path("./checkpoints/<path-to-your-unet-weights>") / "unet-final.pth",
        "pred_steps": 9999999,
        "adaptive_scale": None,
        "obs_suffixes": [".fil", ".h5"],
        "obs_file_path": ["./data/<path-to-your-data>/xx/", "./data/<path-to-your-data>/yy/"],
        "Beam": None,
        "YY_dir": "./data/<path-to-your-data>/yy/",
        "XX_dir": "./data/<path-to-your-data>/xx/",
        "stokes_mode": "I",
        "ignore_polarization": True,
    },

    "33exoplanets": {
        # pipeline
        "dwtnet_ckpt": Path("./checkpoints/mswunet/bin256") / "final.pth",
        "unet_ckpt": Path("./checkpoints/unet") / "best_model.pth",
        "pred_steps": 9999999,
        "adaptive_scale": None,
        "obs_suffixes": [".fil", ".h5"],
        "obs_file_path": ["/data/Raid0/obs_data/33exoplanets/xx/", "/data/Raid0/obs_data/33exoplanets/yy/"],
        "Beam": None,
        "YY_dir": "/data/Raid0/obs_data/33exoplanets/yy/",
        "XX_dir": "/data/Raid0/obs_data/33exoplanets/xx/",
        "stokes_mode": "I",
        "ignore_polarization": True,
    },

    "BLIS692NS": {
        # pipeline
        "dwtnet_ckpt": Path("./checkpoints/mswunet/bin256") / "final.pth",
        "unet_ckpt": Path("./checkpoints/unet") / "best_model.pth",
        "pred_steps": 9999999,
        "adaptive_scale": None,
        "obs_suffixes": [".fil", ".h5"],
        "obs_file_path": "./data/BLIS692NS/BLIS692NS_data",
        "Beam": None,
        "ignore_polarization": False,
    },

    "ce4": {
        # simulating
        "fchans": 256,
        "tchans": 256,
        "snr_min": 300.0,
        "snr_max": 600.0,
        "width_min": 1e3,
        "width_max": 1e4,
        "drift_min": -1000.0,  # 19000 // 10
        "drift_max": 1000.0,
        "drift_min_abs": 20 * (40 - 1.016) * 1e6 // (2001 - 1) // (256 * 9.755183743169399),
        "num_signals": (1, 1),
        "background_fil": list(Path("./data/CE4").rglob("*.2C")),

        # pipeline
        "detect_backend": "trackline",
        "patch_t": 256,
        "patch_f": 256,
        "overlap_pct": 0.02,
        "dwtnet_ckpt": Path("./checkpoints/CE4/mswunet/bin256") / "best_model.pth",
        "pred_steps": 9999999,
        "t_adaptive": True,
        "adaptive_scale": [1, 256],
        "obs_suffixes": [".2C"],
        "obs_file_path": "/data/Raid0/obs_data/CE4_LFRS_2C",
        "Beam": None,
        "ignore_polarization": False,
        "drift": [-1000.0, 1000.0],
    },

    "train1024bin": {
        # extracted from abandoned/main_b1024.py
        "fchans": 1024,
        "snr_min": 30.0,
        "snr_max": 50.0,
        "width_min": 7.5,
        "width_max": 20,
        "fil_folder": Path("./data/33exoplanets/bg/clean"),
        "background_fil": list(Path("./data/33exoplanets/bg/clean").rglob("*.fil")),
        "batch_size": 16,
        "num_epochs": 99999,
        "checkpoint_dir": "./checkpoints/mswunet/bin1024",
        "N": 10,
        "detector_args": {
            "fchans": 1024,
            "N": 10,
            "num_classes": 2,
            "feat_channels": 64,
            "dropout": 0.001,
        },
        "regress_loss_args": {
            "lambda_denoise": 0.5,
            "loss_type": "mse",
            "lambda_learnable": False,
            "regression_loss_kwargs": {
                "num_classes": 2,
                "N": 10,
                "w_loc": 1.75,
                "w_class": 0.05,
                "w_conf": 0.5,
                "eps": 1e-8,
            },
        },
    },

    "train512bin": {
        # extracted from abandoned/main_b512.py
        "fchans": 512,
        "snr_min": 30.0,
        "snr_max": 50.0,
        "width_min": 7.5,
        "width_max": 20,
        "fil_folder": Path("./data/33exoplanets/bg/clean"),
        "background_fil": list(Path("./data/33exoplanets/bg/clean").rglob("*.fil")),
        "batch_size": 16,
        "num_epochs": 99999,
        "freeze_backbone": False,
        "checkpoint_dir": "./checkpoints/mswunet/bin512",
        "N": 10,
        "detector_args": {
            "fchans": 512,
            "N": 10,
            "num_classes": 2,
            "feat_channels": 64,
            "dropout": 0.005,
        },
        "regress_loss_args": {
            "lambda_denoise": 1.0,
            "loss_type": "mse",
            "lambda_learnable": False,
            "regression_loss_kwargs": {
                "num_classes": 2,
                "N": 10,
                "w_loc": 1.5,
                "w_class": 0.1,
                "w_conf": 0.3,
                "eps": 1e-8,
            },
        },
    },

    "train256bin": {
        # extracted from abandoned/main_b256.py
        "snr_min": 30.0,
        "snr_max": 50.0,
        "width_min": 7.5,
        "width_max": 20,
        "fil_folder": Path("./data/33exoplanets/bg/clean"),
        "background_fil": list(Path("./data/33exoplanets/bg/clean").rglob("*.fil")),
        "batch_size": 64,
        "num_epochs": 99999,
        "eta_min": 1.0e-11,
        "regress_loss_args": {
            "lambda_denoise": 0.5,
            "loss_type": "mse",
            "lambda_learnable": False,
            "regression_loss_kwargs": {
                "num_classes": 2,
                "N": 5,
                "w_loc": 1.5,
                "w_class": 0.05,
                "w_conf": 0.3,
                "eps": 1e-8,
            },
        },
    },

    "ce4train256bin": {
        "fchans": 256,
        "tchans": 256,
        "fch1": 0.,
        "snr_min": 300.0,
        "snr_max": 600.0,
        "width_min": 1e3,
        "width_max": 1e4,
        "drift_min": -1000.0,  # 19000 // 10
        "drift_max": 1000.0,
        "drift_min_abs": 20 * (40 - 1.016) * 1e6 // (2001 - 1) // (256 * 9.755183743169399),
        "num_signals": (0, 1),
        "background_fil": list(Path("./data/CE4/bg").rglob("*.2C")),
        "batch_size": 16,
        "checkpoint_dir": "./checkpoints/CE4/mswunet/bin256",
        "N": 5,
        "detector_args": {
            "fchans": 256,
            "N": 5,
            "num_classes": 2,
            "feat_channels": 64,
            "dropout": 0.05,
        },
    },

    "RAW_pred": {
        # extracted from abandoned/pred_raw.py
        "XX_dir": "/data/Raid0/obs_data/33exoplanets/xx/",
        "YY_dir": "/data/Raid0/obs_data/33exoplanets/yy/",
        "Beam": [1],
        "obs_file_path": ["/data/Raid0/obs_data/33exoplanets/xx/", "/data/Raid0/obs_data/33exoplanets/yy/"],
        "RAW": True,
        "num_workers": 4,
        "pred_steps": 1000,
        "dwtnet_ckpt": Path("./checkpoints/mswunet/bin256") / "best_model.pth",
        "nms_kargs": {
            "iou_thresh": 1.0,
            "score_thresh": 0.5,
        },
    },
}


def load_config(config_name=None):
    name = (config_name if config_name is not None else Settings.CONFIG).strip()
    if not name:
        name = "default"

    config_key = next((key for key in CONFIGS if key.lower() == name.lower()), None)
    if config_key is None:
        available = ", ".join(CONFIGS.keys())
        raise ValueError(f"Unknown config '{name}'. Available configs: {available}")

    config = deepcopy(CONFIGS["default"])
    if config_key != "default":
        config.update(deepcopy(CONFIGS[config_key]))
    return config
