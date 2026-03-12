#!/usr/bin/env python3

"""Download a small Semantic Gaussians-compatible sample scene.

This script currently targets a single Mip-NeRF 360 scene in COLMAP layout.
It mirrors the scene extraction logic used by nerfbaselines for
`external://mipnerf360/<scene>` and keeps only the files needed by the
Semantic Gaussians loaders.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


MIPNERF360_BASE_URL = "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
DEFAULT_OUTPUT_ROOT = Path("data/semantic_gaussians_samples")

SCENES = {
    "stump": {
        "url": MIPNERF360_BASE_URL,
        "downscale_factor": 4,
        "description": "Small outdoor Mip-NeRF 360 scene in COLMAP format.",
    },
    "bonsai": {
        "url": MIPNERF360_BASE_URL,
        "downscale_factor": 2,
        "description": "Indoor Mip-NeRF 360 scene, slightly larger than stump.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a tiny Semantic Gaussians sample dataset.",
    )
    parser.add_argument(
        "--dataset",
        default="mipnerf360",
        choices=("mipnerf360",),
        help="Dataset family to download.",
    )
    parser.add_argument(
        "--scene",
        default="stump",
        choices=tuple(SCENES),
        help="Scene name. `stump` is the recommended lightweight default.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory where the sample scene will be extracted.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite the target scene if it already exists.",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the downloaded zip under the output root.",
    )
    parser.add_argument(
        "--test-download-mb",
        type=int,
        default=0,
        help=(
            "Only download the first N MB with an HTTP Range request to verify "
            "network access and URL validity. This does not extract the dataset."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def download_file(url: str, output_path: Path) -> None:
    logging.info("Downloading %s", url)
    with urllib.request.urlopen(url) as response, open(output_path, "wb") as handle:
        shutil.copyfileobj(response, handle)


def download_partial_file(url: str, output_path: Path, megabytes: int) -> int:
    if megabytes <= 0:
        raise ValueError("megabytes must be positive")

    num_bytes = megabytes * 1024 * 1024
    end_byte = num_bytes - 1
    request = urllib.request.Request(url, headers={"Range": f"bytes=0-{end_byte}"})
    logging.info("Testing partial download: first %d MB from %s", megabytes, url)
    try:
        with urllib.request.urlopen(request) as response, open(output_path, "wb") as handle:
            shutil.copyfileobj(response, handle)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            "Server did not accept the test Range request. "
            "Try downloading the full scene without --test-download-mb."
        ) from exc
    return output_path.stat().st_size


def extract_scene(zip_path: Path, scene: str, downscale_factor: int, target_dir: Path) -> None:
    images_dir_name = "images" if downscale_factor == 1 else f"images_{downscale_factor}"
    sparse_dir_name = "sparse" if downscale_factor == 1 else f"sparse_{downscale_factor}"

    extracted_any = False
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if not info.filename.startswith(scene + "/"):
                continue
            keep = (
                info.filename.startswith(f"{scene}/{images_dir_name}/")
                or info.filename.startswith(f"{scene}/sparse/")
            )
            if not keep:
                continue

            extracted_any = True
            relative_name = info.filename[len(scene) + 1 :]
            if relative_name.startswith("sparse/") and sparse_dir_name != "sparse":
                relative_name = relative_name.replace("sparse/", f"{sparse_dir_name}/", 1)

            destination = target_dir / relative_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                with archive.open(info) as source, open(destination, "wb") as sink:
                    shutil.copyfileobj(source, sink)

    if not extracted_any:
        raise RuntimeError(f"Scene '{scene}' was not found in {zip_path.name}")


def validate_scene_layout(scene_dir: Path, downscale_factor: int) -> None:
    image_dir = scene_dir / ("images" if downscale_factor == 1 else f"images_{downscale_factor}")
    sparse_dir = scene_dir / ("sparse" if downscale_factor == 1 else f"sparse_{downscale_factor}") / "0"

    if not image_dir.is_dir():
        raise RuntimeError(f"Missing image directory: {image_dir}")
    if not sparse_dir.is_dir():
        raise RuntimeError(f"Missing sparse COLMAP directory: {sparse_dir}")
    if not any(image_dir.iterdir()):
        raise RuntimeError(f"No images found in {image_dir}")

    required_sparse = ("cameras.bin", "images.bin", "points3D.bin")
    missing = [name for name in required_sparse if not (sparse_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Missing COLMAP files in {sparse_dir}: {', '.join(missing)}")


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    scene_meta = SCENES[args.scene]
    scene_dir = args.output_root / args.dataset / args.scene
    zip_path = args.output_root / args.dataset / f"{args.scene}.zip"
    partial_path = args.output_root / args.dataset / f"{args.scene}.partial"

    scene_dir.parent.mkdir(parents=True, exist_ok=True)

    if args.test_download_mb > 0:
        bytes_downloaded = download_partial_file(scene_meta["url"], partial_path, args.test_download_mb)
        print(
            "\nPartial download test succeeded:\n"
            f"  file: {partial_path}\n"
            f"  bytes: {bytes_downloaded}\n"
            f"  note: this only verifies URL/network access; it does not extract the dataset."
        )
        return 0

    if scene_dir.exists() and not args.force:
        logging.info("Scene already exists at %s", scene_dir)
        validate_scene_layout(scene_dir, scene_meta["downscale_factor"])
        print_next_steps(scene_dir)
        return 0

    if scene_dir.exists():
        logging.info("Removing existing scene directory %s", scene_dir)
        shutil.rmtree(scene_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_zip = Path(tmpdir) / "sample_scene.zip"
        download_file(scene_meta["url"], temp_zip)

        if args.keep_zip:
            shutil.copy2(temp_zip, zip_path)
            logging.info("Saved downloaded archive to %s", zip_path)

        extract_scene(temp_zip, args.scene, scene_meta["downscale_factor"], scene_dir)

    validate_scene_layout(scene_dir, scene_meta["downscale_factor"])
    logging.info("Prepared %s sample at %s", args.dataset, scene_dir)
    print_next_steps(scene_dir)
    return 0


def print_next_steps(scene_dir: Path) -> None:
    print("\nSample scene is ready:")
    print(f"  {scene_dir}")
    print("\nSuggested next step:")
    print("  1. Set `semantic-gaussians/config/official_train.yaml` -> `scene.scene_path`")
    print(f"     to `{scene_dir}`")
    print("  2. Train a small 3DGS scene with `python train.py` inside `semantic-gaussians/`")


if __name__ == "__main__":
    sys.exit(main())
