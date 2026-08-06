#!/usr/bin/env python3
"""Stitch live-dashboard frames into an animated learning replay (GIF/MP4).

    python scripts/make_learning_replay.py --frames results/frames --out results/learning_replay.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="results/frames")
    ap.add_argument("--out", default="results/learning_replay.gif")
    ap.add_argument("--fps", type=int, default=4)
    args = ap.parse_args()

    frames = sorted(Path(args.frames).glob("frame_*.png"))
    if not frames:
        raise SystemExit(f"no frames in {args.frames} — enable live_dashboard in config and train first")
    images = [imageio.imread(f) for f in frames]
    imageio.mimsave(args.out, images, duration=1.0 / args.fps, loop=0)
    print(f"{len(frames)} frames -> {args.out}")


if __name__ == "__main__":
    main()
