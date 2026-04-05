#!/usr/bin/env python3
"""
Evaluation script for mouth landmark detection with FSR and PCK metrics.

Usage:
  python -m Unet_training_script.src.evaluate_predictions_unet_fsr   \
    --checkpoint /inwdata2a/sudhanshu/mouth_keypoints/outputs_landmarks_unet_op/best.pt \
    --config     /inwdata2a/sudhanshu/Unet_training_script/configs/default.yaml \
    --visualize  --show_gt  --limit 200
"""

import argparse
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

# Package path setup
THIS_FILE    = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if __package__ in (None, ""):
    __package__ = "Unet_training_script.src"

from .dataset import MouthDataset
from .transforms import build_val_transforms
from .utils import (
    load_config, load_checkpoint, ensure_dir,
    load_unet_segmentation_model, compute_fsr_torch
)
from ..models.unet_encoder_model import MouthLandmarkModel, generate_gaussian_heatmaps


# ── COORDINATE HELPERS ────────────────────────────────────────────────────────

def denormalize_landmarks(lm_norm: torch.Tensor, mouth_bbox: torch.Tensor):
    """
    Convert [0,1] normalized mouth-crop-local landmarks back to face crop pixel space.
    Inverse of training normalization: lx = (x - mx1) / (mx2 - mx1 + 1)

    Args:
        lm_norm:    [L, 2] tensor in [0,1]
        mouth_bbox: [4]    tensor (x1, y1, x2, y2) in face crop pixel space

    Returns:
        list of (x, y) in face crop pixel space
    """
    x1, y1, x2, y2 = mouth_bbox.tolist()
    bw = max(1.0, x2 - x1 + 1)
    bh = max(1.0, y2 - y1 + 1)
    return [(xn * bw + x1, yn * bh + y1)
            for xn, yn in lm_norm.tolist()]


# ── VISUALISATION ─────────────────────────────────────────────────────────────

def visualize_sample(img_path: str,
                     face_bbox: torch.Tensor,
                     mouth_bbox: torch.Tensor,
                     pred_abs, gt_abs,
                     out_png: str,
                     show_gt: bool = True):
    """
    Save two-panel visualization:
      Left:  face crop with mouth bbox + landmarks overlaid
      Right: zoomed mouth crop at original resolution

    Coordinate spaces:
      img_path   → full DMS frame
      face_bbox  → face crop location in DMS frame space
      mouth_bbox → mouth crop location in face crop space
      pred_abs, gt_abs → landmarks in face crop space (after denormalize)
    """
    # ── Reconstruct face crop from DMS frame ──────────────────────────────────
    dms_img      = Image.open(img_path)
    fx1, fy1, fx2, fy2 = [int(v) for v in face_bbox.tolist()]
    w_dms, h_dms = dms_img.size
    fx1, fx2 = sorted([max(0, min(fx1, w_dms-1)), max(0, min(fx2, w_dms-1))])
    fy1, fy2 = sorted([max(0, min(fy1, h_dms-1)), max(0, min(fy2, h_dms-1))])
    face_crop = dms_img.crop((fx1, fy1, fx2+1, fy2+1))

    # ── Mouth bbox in face crop space ─────────────────────────────────────────
    mx1, my1, mx2, my2 = [int(v) for v in mouth_bbox.tolist()]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # ── Left: face crop + mouth bbox + landmarks ───────────────────────────────
    axes[0].imshow(face_crop, cmap='gray' if face_crop.mode == 'L' else None)
    axes[0].axis('off')
    axes[0].set_title("Face crop")
    axes[0].add_patch(patches.Rectangle(
        (mx1, my1), mx2 - mx1, my2 - my1,
        linewidth=1, edgecolor='yellow', facecolor='none'
    ))
    if pred_abs:
        axes[0].scatter([x for x, y in pred_abs], [y for x, y in pred_abs],
                        c='lime', s=20, marker='o', edgecolors='black',
                        linewidths=0.5, label='pred', zorder=5)
    if show_gt and gt_abs:
        axes[0].scatter([x for x, y in gt_abs], [y for x, y in gt_abs],
                        c='red', s=20, marker='x', label='gt', zorder=5)
    axes[0].legend(loc='lower right', fontsize=7)

    # ── Right: zoomed mouth crop at original resolution ────────────────────────
    pad  = 10
    zx1  = max(0, mx1 - pad)
    zy1  = max(0, my1 - pad)
    zx2  = min(face_crop.size[0], mx2 + pad)
    zy2  = min(face_crop.size[1], my2 + pad)
    zoom = face_crop.crop((zx1, zy1, zx2, zy2))

    axes[1].imshow(zoom, cmap='gray' if zoom.mode == 'L' else None)
    axes[1].axis('off')
    axes[1].set_title("Mouth crop (original resolution)")
    if pred_abs:
        axes[1].scatter([x - zx1 for x, y in pred_abs],
                        [y - zy1 for x, y in pred_abs],
                        c='lime', s=30, marker='o', edgecolors='black',
                        linewidths=0.5, label='pred', zorder=5)
    if show_gt and gt_abs:
        axes[1].scatter([x - zx1 for x, y in gt_abs],
                        [y - zy1 for x, y in gt_abs],
                        c='red', s=30, marker='x', label='gt', zorder=5)
    axes[1].legend(loc='lower right', fontsize=7)

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches='tight', dpi=150)
    plt.close(fig)


# ── ARGS ──────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",     required=True)
    ap.add_argument("--config",         required=True)
    ap.add_argument("--limit",          type=int, default=None,
                    help="Evaluate only first N samples")
    ap.add_argument("--visible_only",   action="store_true",
                    help="Skip frames where mouth_visibility=0")
    ap.add_argument("--visualize",      action="store_true",
                    help="Save overlay images")
    ap.add_argument("--show_gt",        action="store_true",
                    help="Show GT landmarks in overlay")
    ap.add_argument("--save_per_point", action="store_true",
                    help="Include per-point errors in output CSV")
    ap.add_argument("--percentiles",    type=str, default="90,95,99")
    return ap.parse_args()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Output dirs ───────────────────────────────────────────────────────────
    outdir      = cfg["paths"]["output_dir"]
    overlay_dir = os.path.join(outdir, "overlays")
    report_txt  = os.path.join(outdir, "evaluation_report.txt")
    csv_out     = os.path.join(outdir, "per_sample_results.csv")
    ensure_dir(outdir)
    if args.visualize:
        ensure_dir(overlay_dir)

    # ── Model — identical to training script ──────────────────────────────────
    pretrain_path    = cfg['model']['pretrain_encoder_ckpt']
    backbone = load_unet_segmentation_model(pretrain_path, device=device)

    model = MouthLandmarkModel(
        backbone=backbone,
        num_landmarks=int(cfg['data']['num_landmarks']),
    ).to(device)

    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state'], strict=False)
    model.eval()
    print(f"[INFO] Loaded checkpoint: {args.checkpoint}")

    # ── Data — identical pipeline to training ─────────────────────────────────
    dataset = MouthDataset(
        csv_path  = cfg['paths']['test_csv'],
        cfg       = cfg,
        transform = build_val_transforms(cfg),
        is_train  = False
    )
    if args.limit is not None:
        dataset.valid_indices = dataset.valid_indices[:args.limit]

    loader = DataLoader(
        dataset,
        batch_size  = int(cfg['training']['batch_size']),
        shuffle     = False,
        num_workers = int(cfg['training']['num_workers']),
        pin_memory  = True
    )
    print(f"[INFO] Evaluating on {len(dataset)} samples")

    # ── Evaluation loop ───────────────────────────────────────────────────────
    num_landmarks  = int(cfg['data']['num_landmarks'])
    pck_thresholds = cfg['inference']['pck_thresholds']
    pct_list       = [int(p.strip()) for p in args.percentiles.split(",")
                      if p.strip().isdigit()]

    per_sample_rows    = []
    all_point_l2       = []
    all_point_norm     = []
    per_image_nme      = []
    all_frame_max_norm = []
    samples_used       = 0
    skipped            = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating")):
        img         = batch['image'].to(device)      # [B, 1, H, W]
        lmk_gt      = batch['landmarks']             # [B, L, 2] normalized [0,1]
        mask        = batch['mask']                  # [B, L]
        vis         = batch['visibility'].view(-1)   # [B]
        mouth_bboxes = batch['bbox']                 # [B, 4] mouth bbox in face crop space
        face_bboxes  = batch['face_bbox']            # [B, 4] face bbox in DMS frame space
        img_paths    = batch['img_path']             # list of B paths

        with torch.no_grad():
            out       = model(img)
            pred_norm = out['coords'].cpu()          # [B, L, 2] normalized [0,1]

        for i in range(img.shape[0]):
            gt_vis = int(vis[i].item())
            if args.visible_only and gt_vis == 0:
                skipped += 1
                continue

            if mask[i].sum() == 0:
                skipped += 1
                continue

            mouth_bbox = mouth_bboxes[i]   # [4] mouth bbox in face crop space
            face_bbox  = face_bboxes[i]    # [4] face bbox in DMS frame space
            pred_lm    = pred_norm[i]      # [L, 2] normalized
            gt_lm      = lmk_gt[i]        # [L, 2] normalized

            # Denormalize both to face crop pixel space for L2 metric
            # mouth_bbox is used — same bbox dataset used for normalization
            pred_abs = denormalize_landmarks(pred_lm, mouth_bbox)
            gt_abs   = denormalize_landmarks(gt_lm,   mouth_bbox)

            pred_arr = np.array(pred_abs)   # [L, 2]
            gt_arr   = np.array(gt_abs)     # [L, 2]

            # Normalize errors by mouth width (x-span of GT landmarks)
            mouth_width = max(1.0, gt_arr[:, 0].max() - gt_arr[:, 0].min())
            point_l2    = np.sqrt(((pred_arr - gt_arr) ** 2).sum(axis=1))  # [L]
            point_norm  = point_l2 / mouth_width                            # [L]

            all_point_l2.extend(point_l2.tolist())
            all_point_norm.extend(point_norm.tolist())
            per_image_nme.append(float(point_norm.mean()))
            all_frame_max_norm.append(float(point_norm.max()))
            samples_used += 1

            row_dict = {
                "img_path":      img_paths[i],
                "gt_visibility": gt_vis,
                "mean_l2":       float(point_l2.mean()),
                "mean_norm":     float(point_norm.mean()),
                "max_norm":      float(point_norm.max()),
                "mouth_width":   mouth_width,
            }
            if args.save_per_point:
                for j in range(num_landmarks):
                    row_dict[f"l2_pt{j}"]   = float(point_l2[j])
                    row_dict[f"norm_pt{j}"] = float(point_norm[j])
            per_sample_rows.append(row_dict)

            if args.visualize:
                out_png = os.path.join(
                    overlay_dir,
                    f"sample_{batch_idx * loader.batch_size + i}.png"
                )
                visualize_sample(
                    img_paths[i], face_bbox, mouth_bbox,
                    pred_abs, gt_abs if args.show_gt else None,
                    out_png, show_gt=args.show_gt
                )

    # ── Metrics summary ───────────────────────────────────────────────────────
    all_l2_arr         = np.array(all_point_l2)
    all_norm_arr       = np.array(all_point_norm)
    image_nme_arr      = np.array(per_image_nme)
    frame_max_norm_arr = np.array(all_frame_max_norm)

    if all_l2_arr.size == 0:
        summary = "No valid samples evaluated."
    else:
        lines = [
            f"Samples evaluated : {samples_used}  (skipped: {skipped})",
            f"Points evaluated  : {all_l2_arr.size}",
            f"",
            f"── Pixel L2 ───────────────────────────────────────",
            f"Mean   : {all_l2_arr.mean():.4f} px",
            f"Median : {np.median(all_l2_arr):.4f} px",
            f"Std    : {all_l2_arr.std():.4f} px",
        ]
        for p in pct_list:
            lines.append(f"{p}th pct : {np.percentile(all_l2_arr, p):.4f} px")

        lines += [
            f"",
            f"── NME (normalized by mouth width) ────────────────",
            f"Mean per-image NME : {image_nme_arr.mean():.6f}",
            f"Global NME         : {all_norm_arr.mean():.6f}",
            f"",
            f"── PCK / FSR ──────────────────────────────────────",
            f"{'Threshold':<10} | {'PCK (point-wise)':<18} | {'FSR (frame-wise)':<18}",
            "-" * 54,
        ]
        for thr in pck_thresholds:
            pck = (all_norm_arr       <= thr).sum() / max(1, all_norm_arr.size)
            fsr = (frame_max_norm_arr <= thr).sum() / max(1, samples_used)
            lines.append(f"{thr:<10.3f} | {pck:<18.4f} | {fsr:<18.4f}")

        summary = "\n".join(lines)

    print("\n" + summary)
    with open(report_txt, "w") as f:
        f.write(summary + "\n")

    pd.DataFrame(per_sample_rows).to_csv(csv_out, index=False)
    print(f"\n[INFO] Report  : {report_txt}")
    print(f"[INFO] CSV     : {csv_out}")
    if args.visualize:
        print(f"[INFO] Overlays: {overlay_dir}")


if __name__ == "__main__":
    main()