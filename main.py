"""
TacticsAI CLI — process a video file and save annotated output.

Usage::
    python main.py data/test_clip.mp4
    python main.py data/test_clip.mp4 --device cpu --skip-frames 2
    python main.py data/test_clip.mp4 --max-frames 500 --output output/preview.mp4
"""

import argparse
import sys
from pathlib import Path

import cv2

from config import DEFAULT_MODEL, OUTPUT_DIR
from src.pipeline import get_video_info, process_video


# ── CLI argument parser ───────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python main.py",
        description="TacticsAI — Week 1: Detection & Tracking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("video", help="Input video path (.mp4 / .avi)")
    p.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="YOLOv8 weights. 'best.pt' after Colab fine-tuning.",
    )
    p.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"],
        help="Inference device.",
    )
    p.add_argument(
        "--skip-frames", type=int, default=0, metavar="N",
        help="Process every (N+1)-th frame. 0 = every frame.",
    )
    p.add_argument(
        "--max-frames", type=int, default=None, metavar="N",
        help="Stop after N processed frames (useful for quick previews).",
    )
    p.add_argument(
        "--output", default=None,
        help="Output video path. Defaults to output/<input_stem>_annotated.mp4",
    )
    p.add_argument(
        "--no-save", action="store_true",
        help="Don't write an output video (just print stats).",
    )
    return p


# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner(info: dict, args: argparse.Namespace) -> None:
    w = 54
    print(f"\n{'─' * w}")
    print(f"  TacticsAI — Week 1: Detection & Tracking")
    print(f"{'─' * w}")
    print(f"  Input   : {Path(args.video).name}")
    print(f"  Duration: {info['duration_s']:.1f}s  "
          f"({info['total_frames']} frames @ {info['fps']:.0f} fps)")
    print(f"  Size    : {info['width']} × {info['height']}")
    print(f"  Model   : {args.model}   device: {args.device}")
    if args.skip_frames:
        effective_fps = info["fps"] / (args.skip_frames + 1)
        print(f"  Skip    : every {args.skip_frames + 1} frames  (~{effective_fps:.1f} fps effective)")
    if args.max_frames:
        print(f"  Max frm : {args.max_frames}")
    print(f"{'─' * w}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = build_parser().parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[ERROR] File not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    try:
        info = get_video_info(str(video_path))
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    _banner(info, args)

    # ── Set up VideoWriter ────────────────────────────────────────────────────
    writer = None
    out_path = None

    if not args.no_save:
        out_path = args.output or str(
            OUTPUT_DIR / f"{video_path.stem}_annotated.mp4"
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            out_path,
            fourcc,
            info["fps"],
            (info["width"], info["height"]),
        )
        print(f"  Output  : {out_path}\n")

    # ── Run pipeline ──────────────────────────────────────────────────────────
    stats = {"frames": 0, "players": 0, "ball_frames": 0}

    try:
        for result in process_video(
            str(video_path),
            model_path=args.model,
            device=args.device,
            skip_frames=args.skip_frames,
            max_frames=args.max_frames,
            show_progress=True,
        ):
            if writer:
                writer.write(result["annotated_frame"])

            stats["frames"] += 1
            stats["players"] += result["player_count"]
            if result["ball_detected"]:
                stats["ball_frames"] += 1

    except KeyboardInterrupt:
        print("\n  Interrupted by user.")

    finally:
        if writer:
            writer.release()

    # ── Summary ───────────────────────────────────────────────────────────────
    n = max(stats["frames"], 1)
    print(f"\n{'─' * 54}")
    print(f"  Processed       : {stats['frames']} frames")
    print(f"  Avg players/frm : {stats['players'] / n:.1f}")
    print(f"  Ball detected   : {100 * stats['ball_frames'] / n:.1f}% of frames")
    if out_path:
        print(f"  Saved to        : {out_path}")
    print(f"{'─' * 54}\n")


if __name__ == "__main__":
    main()
