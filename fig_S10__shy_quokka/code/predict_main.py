import numpy as np
import pandas as pd
import pathlib as pl
import pickle
import argparse
import scipy as sp
import logging

import candas as can
import gumbi as gmb
from candas.test import FluorescenceData, QuantStudio

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Setup paths
code_pth = pl.Path(__file__).parent
fig_pth = code_pth.parent
data_pth = fig_pth / "data"
gen_pth = fig_pth / "generated"
gen_pth.mkdir(exist_ok=True)

logger.info("Loading pre-processed data...")
# Load and process data


def load_models():
    """Load the fitted GP models"""
    logger.info("Loading fitted GP models...")
    with open(gen_pth / "gp_model.pkl", "rb") as f:
        gp = pickle.load(f)
    with open(gen_pth / "gp_zero_model.pkl", "rb") as f:
        gp_zero = pickle.load(f)
    with open(gen_pth / "uncertainty_params.pkl", "rb") as f:
        uncertainty_params = pickle.load(f)
    return gp, gp_zero, uncertainty_params


def prepare_predictions(predictions):
    """Prepare predictions for saving"""
    items = list(predictions.items())
    for k, v in items:
        if hasattr(v, "names"):
            predictions[k + "_names"] = v.names
        if hasattr(v, "name"):
            predictions[k + "_name"] = v.name
    return predictions


def predict_weak_yaksha(gp, blocker_um):
    """Make predictions for the weak_yaksha heatmap plot"""
    logger.info(
        f"Making weak_yaksha heatmap predictions for blocker concentration {blocker_um} μM..."
    )
    at = gp.parray(**{"Blocker μM": blocker_um})
    limits = gp.parray(SNV_lg10_Copies=[0, 8], WT_lg10_Copies=[0, 9])
    XY = gp.prepare_grid(at=at, limits=limits)
    SNV = XY["SNV_lg10_Copies"]
    WT = XY["WT_lg10_Copies"]
    sig = gp.predict_grid(with_noise=False)

    predictions = {"SNV": SNV, "WT": WT, "signal": sig}
    return prepare_predictions(predictions)


def predict_big_zebra(gp, blocker_um, wt_copies):
    """Make predictions for signal vs SNV copies plot"""
    logger.info(
        f"Making signal vs SNV predictions for blocker {blocker_um} μM and WT copies {wt_copies}..."
    )
    at = gp.parray(**{"Blocker μM": blocker_um, "WT_lg10_Copies": wt_copies})
    limits = gp.parray(SNV_lg10_Copies=[0, 8])
    X = gp.prepare_grid(at=at, limits=limits)["SNV_lg10_Copies"]
    sig = gp.predict_grid(with_noise=True)

    predictions = {"SNV": X, "signal": sig}
    return prepare_predictions(predictions)


def predict_vibrant_ibis(gp, gp_zero, blocker_um, min_detectable_diff):
    """Make predictions for VAF analysis"""
    logger.info(f"Making VAF predictions for blocker concentration {blocker_um} μM...")
    at = gp.parray(**{"Blocker μM": blocker_um})
    limits = gp.parray(SNV_lg10_Copies=[0, 8], WT_lg10_Copies=[0, 8])
    X = gp.prepare_grid(at=at, limits=limits, resolution=810)[
        ["WT_lg10_Copies", "SNV_lg10_Copies"]
    ]
    sig = gp.predict_grid(with_noise=False)

    at = gp_zero.parray(**{"Blocker μM": blocker_um})
    limits = gp_zero.parray(WT_lg10_Copies=[0, 8])
    X_zero = gp_zero.prepare_grid(at=at, limits=limits, resolution=810)[
        "WT_lg10_Copies"
    ]
    sig_zero = gp_zero.predict_grid(with_noise=False)

    pred_diff_m = sig.μ - sig_zero.μ.reshape(-1, 1)
    pred_diff_σ = np.sqrt(sig.σ**2 + sig_zero.σ.reshape(-1, 1) ** 2)
    pred_diff_dist = sp.stats.norm(loc=pred_diff_m, scale=pred_diff_σ)

    pred_diff_l, pred_diff_u = pred_diff_dist.interval(0.95)

    snv_vec = gp.grid_vectors["SNV_lg10_Copies"].values().flatten()
    wt_vec = gp.grid_vectors["WT_lg10_Copies"].values().flatten()

    def get_min_vaf(pred, snv_vec, wt_vec, min_detectable_diff):
        detected = pred > min_detectable_diff
        # chgpt = np.diff(detected.astype(float), prepend=1.0).argmax(axis=1)
        chgpt = (np.diff(detected[:, ::-1].astype(float), prepend=0.0).argmin(axis=1))
        chgpt = detected.shape[1] - 1 - chgpt

        all_detected = detected.all(axis=1)
        none_detected = ~detected.any(axis=1)

        lod = snv_vec[chgpt]
        lod = np.where(all_detected, -np.inf, lod)
        lod = np.where(none_detected, np.inf, lod)

        vaf = lod - wt_vec

        return lod, vaf

    lod_l, vaf_l = get_min_vaf(pred_diff_l, snv_vec, wt_vec, min_detectable_diff)
    lod_m, vaf_m = get_min_vaf(pred_diff_m, snv_vec, wt_vec, min_detectable_diff)
    lod_u, vaf_u = get_min_vaf(pred_diff_u, snv_vec, wt_vec, min_detectable_diff)

    predictions = {"wt_vec": wt_vec, "snv_vec": snv_vec, "vaf_l": vaf_l, "vaf_m": vaf_m, "vaf_u": vaf_u, "pred_diff_l": pred_diff_l, "pred_diff_m": pred_diff_m, "pred_diff_u": pred_diff_u}
    return prepare_predictions(predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make predictions for specific panel and axis"
    )
    parser.add_argument(
        "panel",
        type=int,
        choices=[0, 1, 2],
        help="Panel number (0=weak_yaksha, 1=big_zebra, 2=vibrant_ibis)",
    )
    parser.add_argument("axis", type=int, help="Axis number (index of the subplot)")
    parser.add_argument(
        "--wt-copies",
        type=float,
        help="WT copies for signal vs SNV plot (only needed for panel 1)",
    )
    args = parser.parse_args()

    logger.info(f"Starting predictions for panel {args.panel}, axis {args.axis}")

    # Load models
    gp, gp_zero, uncertainty_params = load_models()

    # Get blocker concentration for this axis
    blocker_concs = sorted(gp.data.wide["Blocker μM"].unique())
    blocker_um = blocker_concs[args.axis]

    # Make predictions based on panel type
    if args.panel == 0:
        predictions = predict_weak_yaksha(gp, blocker_um)
        outfile = f"weak_yaksha_predictions_axis_{args.axis}.pkl"
    elif args.panel == 1:
        if args.wt_copies is None:
            raise ValueError("wt-copies argument required for panel 1")
        predictions = predict_big_zebra(gp, blocker_um, args.wt_copies)
        outfile = f"big_zebra_predictions_axis_{args.axis}_wt_{args.wt_copies}.pkl"
    else:  # panel 2
        predictions = predict_vibrant_ibis(
            gp, gp_zero, blocker_um, uncertainty_params["min_detectable_diff"]
        )
        outfile = f"vibrant_ibis_predictions_axis_{args.axis}.pkl"

    for k, v in predictions.items():
        if hasattr(v, "shape"):
            logger.debug(f"Shape of {k}: {v.shape}")

    # Save predictions
    logger.info(f"Saving predictions to {outfile}")
    with open(gen_pth / outfile, "wb") as f:
        pickle.dump(predictions, f)

    logger.info("Predictions complete!")
