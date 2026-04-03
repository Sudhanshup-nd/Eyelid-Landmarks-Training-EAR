
# python -m Unet_training_script.src.train_unet --config /inwdata2a/sudhanshu/Unet_training_script/configs/default.yaml

import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import EyeDataset
from .transforms import build_train_transforms, build_val_transforms
from .losses import build_landmark_loss, HeatmapLossWrapper
from .utils import (
    load_config, seed_everything, ensure_dir, save_checkpoint,
    load_frozen_unet_segmentation_model, compute_fsr_torch, save_val_heatmaps
)
from ..models.unet_encoder_model import EyeLandmarkWithFrozenSegmentationBackbone, generate_gaussian_heatmaps


# ── DATA ──────────────────────────────────────────────────────────────────────

def build_dataloaders(cfg):
    train_ds = EyeDataset(cfg['paths']['train_csv'], cfg, transform=build_train_transforms(cfg), is_train=True)
    val_ds   = EyeDataset(cfg['paths']['val_csv'],   cfg, transform=build_val_transforms(cfg),   is_train=False)

    bs = int(cfg['training']['batch_size'])
    nw = int(cfg['training']['num_workers'])

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    print(f"[INFO] Train samples: {len(train_ds)}  Val samples: {len(val_ds)}")
    print(f"[INFO] Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")
    return train_loader, val_loader


# ── TRAINING LOOP ─────────────────────────────────────────────────────────────

def run_train_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0

    train_bar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(train_bar):
        img  = batch['image'].to(device)
        lmk  = batch['landmarks'].to(device)
        mask = batch['mask'].to(device)
        vis  = batch['visibility'].to(device)

        out           = model(img)
        heatmaps_pred = out['heatmaps']   # [B, L, H, W]
        coords_pred   = out['coords']     # [B, L, 2]

        # Generate GT heatmaps matching predicted heatmap spatial size
        _, _, H_hm, W_hm = heatmaps_pred.shape
        gt_hm = generate_gaussian_heatmaps(lmk, mask, H_hm, W_hm, sigma=2.5)

        # Compute loss
        if isinstance(criterion, HeatmapLossWrapper):
            total_loss, _ = criterion(heatmaps_pred, gt_hm, mask, vis,
                                      coords_pred=coords_pred, coords_gt=lmk)
        else:
            total_loss, _ = criterion(coords_pred, lmk, mask, vis)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()
        if batch_idx % 20 == 0:
            train_bar.set_postfix({"loss": f"{total_loss.item():.4f}"})

    return running_loss / max(1, len(loader))


# ── VALIDATION LOOP ───────────────────────────────────────────────────────────

def run_val_epoch(model, loader, criterion, device, epoch, heatmap_save_dir):
    model.eval()
    all_pred, all_gt, all_mask, all_vis = [], [], [], []
    first_batch_heatmaps = None   # saved for visual debugging

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Epoch {epoch} [Val]"):
            img  = batch['image'].to(device)
            lmk  = batch['landmarks'].to(device)
            mask = batch['mask'].to(device)
            vis  = batch['visibility'].to(device)

            out           = model(img)
            coords_pred   = out['coords']     # [B, L, 2]
            heatmaps_pred = out['heatmaps']   # [B, L, H, W]

            # Save first batch heatmaps for visual inspection
            if first_batch_heatmaps is None:
                _, _, H_hm, W_hm = heatmaps_pred.shape
                gt_hm = generate_gaussian_heatmaps(lmk, mask, H_hm, W_hm, sigma=1.5)
                first_batch_heatmaps = {
                    "pred": heatmaps_pred[0].detach().cpu(),  # first sample in batch
                    "gt":   gt_hm[0].detach().cpu()
                }

            all_pred.append(coords_pred)
            all_gt.append(lmk)
            all_mask.append(mask)
            all_vis.append(vis)

    # Save heatmap visualizations for the first val sample
    save_val_heatmaps(first_batch_heatmaps, epoch, heatmap_save_dir)

    # Concatenate all predictions across batches
    pred_cat = torch.cat(all_pred, 0)   # [N, L, 2]
    gt_cat   = torch.cat(all_gt,   0)   # [N, L, 2]
    mask_cat = torch.cat(all_mask, 0)   # [N, L]
    vis_cat  = torch.cat(all_vis,  0).view(-1)  # [N]

    # L1 loss on valid landmarks only (mask=1 and visible)
    eff_mask    = (mask_cat * vis_cat.unsqueeze(1)).unsqueeze(-1)  # [N, L, 1]
    valid_count = eff_mask.sum() * 2.0  # ×2 for x and y
    val_l1      = (torch.abs(pred_cat - gt_cat) * eff_mask).sum().item() / max(valid_count.item(), 1e-6)

    # FSR metrics
    val_fsr_10 = compute_fsr_torch(pred_cat, gt_cat, visibility=vis_cat, threshold=0.10)
    val_fsr_50 = compute_fsr_torch(pred_cat, gt_cat, visibility=vis_cat, threshold=0.50)

    return val_l1, val_fsr_10, val_fsr_50


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg['seed'])

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(cfg)

    # ── Model ─────────────────────────────────────────────────────────────────
    pretrain_path = cfg['model']['pretrain_encoder_ckpt']
    print(f"[INFO] Loading frozen segmentation backbone from: {pretrain_path}")
    frozen_seg_model = load_frozen_unet_segmentation_model(pretrain_path, device=device)

    model = EyeLandmarkWithFrozenSegmentationBackbone(
        segmentation_model=frozen_seg_model,
        num_landmarks=int(cfg['data']['num_landmarks']),
        return_segmentation_outputs=True
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"[INFO] Trainable params: {trainable_params}  |  Frozen params: {frozen_params}")

    # ── Optimizer and loss ────────────────────────────────────────────────────
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(cfg['training']['lr']),
        weight_decay=float(cfg['training']['weight_decay'])
    )
    criterion = build_landmark_loss(cfg)
    print(f"[INFO] Loss: {criterion}")

    # ── Output paths ──────────────────────────────────────────────────────────
    outdir = cfg['paths']['output_dir']
    ensure_dir(outdir)
    ckpt_best      = os.path.join(outdir, "best.pt")
    ckpt_last      = os.path.join(outdir, "last.pt")
    heatmap_dir    = os.path.join(outdir, "debug_heatmaps")

    # ── Training ──────────────────────────────────────────────────────────────
    best_fsr       = -1.0
    epochs_no_improve = 0
    patience       = int(cfg['training']['early_stop_patience'])

    for epoch in range(int(cfg['training']['epochs'])):
        print(f"\n===== EPOCH {epoch} =====")

        avg_train_loss = run_train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_l1, val_fsr_10, val_fsr_50 = run_val_epoch(model, val_loader, criterion, device, epoch, heatmap_dir)

        print(f"[Epoch {epoch}] "
              f"train_loss={avg_train_loss:.5f}  "
              f"val_L1={val_l1:.5f}  "
              f"val_FSR@0.10={val_fsr_10:.4f} (objective)  "
              f"val_FSR@0.50={val_fsr_50:.4f}")

        # ── Checkpoint + early stopping ───────────────────────────────────────
        ckpt = {
            "epoch":          epoch,
            "model_state":    model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_fsr":       best_fsr,
        }

        if val_fsr_10 > best_fsr:
            best_fsr = val_fsr_10
            epochs_no_improve = 0
            save_checkpoint(ckpt, ckpt_best)
            print(f"[INFO] FSR improved → {best_fsr:.4f}. Saved best checkpoint.")
        else:
            epochs_no_improve += 1
            print(f"[INFO] No improvement ({val_fsr_10:.4f} <= {best_fsr:.4f}). "
                  f"Early stop: {epochs_no_improve}/{patience}")

        save_checkpoint(ckpt, ckpt_last)

        if epochs_no_improve >= patience:
            print(f"[INFO] Early stopping triggered after {patience} epochs without improvement.")
            break


if __name__ == "__main__":
    main()