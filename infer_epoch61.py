#!/usr/bin/env python3
"""
BINARY TUMOR inference (standalone, per-frame).

NOTE ON NAME: This file is named after the original 4-class "epoch-61" model it
replaced, but it is now the standard inference script for the BINARY TUMOR model
(checkpoints_tumor_binary*). The name is kept for continuity — do not be misled.
For MARGIN (red) inference, use infer_sam2_red.py instead.

Combines the model's tumor predictions into a single colour for paper-friendly
visualisation of Stage 1 (whole-tumour detection), without conflating it with
the margin-segmentation Stage 2 output. For the 2-class binary model, class 1
(tumor) is rendered; for the legacy 4-class model, red (remaining) and blue
(resected) are combined.

Default colour: blue (0, 0, 255).  Override with --color R,G,B.

Usage:
    python infer_epoch61.py \\
        --model   checkpoints_v2/best_model_epoch_61.pth \\
        --metadata checkpoints_v2/best_model_epoch_61_metadata.json \\
        --input   "E:/IMERSE/Segmentation/test_videos/caoG_a" \\
        --output-dir "E:/IMERSE/test_output_caoG_a_ep61_combined" \\
        --visualize

Optional flags:
    --color 0,0,255       RGB colour for combined tumor mask (default: blue)
    --alpha 0.5           Overlay transparency
    --ema-alpha 0.3       EMA temporal smoothing (1.0 = off)
    --device cuda
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.inference_engine import InferenceEngine
from src.visualization import VisualizationGenerator


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}


def collect_frames(input_dir: Path):
    return sorted([
        p for p in input_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and not p.stem.endswith('_mask')
        and not p.stem.endswith('_org')
        and not p.stem.endswith('_overlay')
    ])


def main():
    parser = argparse.ArgumentParser(
        description='Epoch-61 combined-tumor inference for paper visualisation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model',      required=True, help='Path to epoch-61 .pth checkpoint')
    parser.add_argument('--metadata',   required=True, help='Path to epoch-61 metadata .json')
    parser.add_argument('--input',      required=True, help='Directory of input frames')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--device',     default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--visualize',  action='store_true', help='Save overlay images')
    parser.add_argument('--alpha',      type=float, default=0.5, help='Overlay transparency')
    parser.add_argument('--ema-alpha',  type=float, default=1.0,
                        help='EMA smoothing across frames (1.0 = off, 0.3 = smooth)')
    parser.add_argument('--color',      type=str, default='0,0,255',
                        help='RGB colour for combined tumor mask, e.g. "0,0,255" for blue')
    parser.add_argument('--fill-holes', action='store_true',
                        help='Fill enclosed holes in the predicted tumor mask using binary_fill_holes. '
                             'Fixes interior gaps caused by boundary-weighted training.')
    parser.add_argument('--close-kernel', type=int, default=0,
                        help='If >0, apply morphological closing with this kernel size (pixels) '
                             'before hole filling. Useful for bridging small gaps between regions.')
    parser.add_argument('--largest-only', action='store_true',
                        help='Keep only the largest connected component in the mask, removing all '
                             'smaller disconnected blobs. Recommended for single-tumor cases.')
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    try:
        r, g, b = [int(x) for x in args.color.split(',')]
        tumor_color = (r, g, b)
    except ValueError:
        print(f"Error: --color must be R,G,B (e.g. 0,0,255), got: {args.color}")
        return

    input_dir = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = collect_frames(input_dir)
    if not frame_paths:
        print(f"No frames found in {input_dir}")
        return
    print(f"Found {len(frame_paths)} frames.")

    print("Loading epoch-61 model...")
    engine = InferenceEngine(args.model, args.metadata, device=args.device)
    print("Model loaded.")

    viz = VisualizationGenerator() if args.visualize else None
    first_img = np.array(Image.open(frame_paths[0]).convert('RGB'))
    H, W = first_img.shape[:2]

    ema_tumor: np.ndarray | None = None
    t0 = time.perf_counter()

    for i, fp in enumerate(frame_paths, 1):
        img = np.array(Image.open(fp).convert('RGB'))
        probs, _ = engine.predict_single(img)   # (C, 512, 512), C=3 or 4

        # Probability of any tumor class (1=red, 2=blue)
        tumor_prob_512 = probs[1:].sum(dim=0)   # shape (512, 512), values in [0, 1]

        # Upsample to original resolution
        tumor_conf = F.interpolate(
            tumor_prob_512.unsqueeze(0).unsqueeze(0),
            size=(H, W), mode='bilinear', align_corners=False,
        ).squeeze().cpu().numpy()

        # EMA temporal smoothing
        if args.ema_alpha < 1.0:
            if ema_tumor is None:
                ema_tumor = tumor_conf.copy()
            else:
                ema_tumor = args.ema_alpha * tumor_conf + (1.0 - args.ema_alpha) * ema_tumor
            tumor_mask = ema_tumor >= 0.5
        else:
            tumor_mask = tumor_conf >= 0.5

        # Post-processing: close small gaps then fill enclosed holes
        if args.close_kernel > 0 or args.fill_holes:
            import cv2
            from scipy.ndimage import binary_fill_holes
            m = tumor_mask.astype(np.uint8)
            if args.close_kernel > 0:
                k = args.close_kernel | 1  # ensure odd
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
            if args.fill_holes:
                m = binary_fill_holes(m).astype(np.uint8)
            tumor_mask = m.astype(bool)

        # Keep only the largest connected component
        if args.largest_only and tumor_mask.any():
            from scipy.ndimage import label as ndlabel
            labeled, num_features = ndlabel(tumor_mask)
            if num_features > 1:
                sizes = np.bincount(labeled.ravel())
                sizes[0] = 0  # ignore background label
                largest_label = sizes.argmax()
                tumor_mask = labeled == largest_label

        # Compose single-colour RGB mask
        rgb_mask = np.zeros((H, W, 3), dtype=np.uint8)
        rgb_mask[tumor_mask] = tumor_color

        Image.fromarray(rgb_mask).save(output_dir / f'{fp.stem}_mask.png')
        Image.open(fp).save(output_dir / f'{fp.stem}_org.png')

        if viz is not None:
            overlay = viz.generate_overlay(img, rgb_mask, alpha=args.alpha)
            Image.fromarray(overlay.astype(np.uint8)).save(
                output_dir / f'{fp.stem}_overlay.png'
            )

        print(f"  [{i}/{len(frame_paths)}] {fp.name}")

    elapsed = time.perf_counter() - t0
    print(f"\n{'=' * 60}")
    print(f"Done. {len(frame_paths)} frames in {elapsed:.1f}s ({len(frame_paths)/elapsed:.1f} FPS)")
    print(f"Output: {output_dir}")
    print(f"{'=' * 60}")
    print(f"\nFrames to video (overlay):")
    print(
        f'ffmpeg -framerate 30 -pattern_type glob -i "{output_dir}/*_overlay.png" '
        f'-c:v libx264 -pix_fmt yuv420p "{output_dir}/overlay_video.mp4"'
    )
    print(f"\nFrames to video (mask only):")
    print(
        f'ffmpeg -framerate 30 -pattern_type glob -i "{output_dir}/*_mask.png" '
        f'-c:v libx264 -pix_fmt yuv420p "{output_dir}/mask_video.mp4"'
    )


if __name__ == '__main__':
    main()
