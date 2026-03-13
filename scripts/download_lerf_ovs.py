#!/usr/bin/env python3

"""Download and extract the LERF-OVS benchmark scenes.

References:
- LangSplat README dataset instructions:
  https://github.com/minghanqin/LangSplat?tab=readme-ov-file#data

The upstream release currently ships a single archive containing multiple scenes.
This helper downloads that archive once, then extracts only the requested scenes.
"""

from __future__ import annotations

import argparse
import html
import http.cookiejar
import logging
import re
import shutil
import ssl
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable

DEFAULT_TIMEOUT_SECONDS = 120

DEFAULT_OUTPUT_ROOT = Path("data/lerf_ovs")
GOOGLE_DRIVE_FILE_ID = "14qscEhHdToKrcKYDssjsQoc1RkWG7M2j"
ARCHIVE_URL = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}"
ARCHIVE_NAME = "lerf_ovs.zip"
SCENE_NAMES = (
    "figurines",
    "ramen",
    "teatime",
    "waldo_kitchen",
)
SCENE_DESCRIPTIONS = {
    "figurines": "LERF-OVS tabletop figurines scene.",
    "ramen": "LERF-OVS ramen meal scene.",
    "teatime": "LERF-OVS tea set scene.",
    "waldo_kitchen": "LERF-OVS kitchen scene.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download selected LERF-OVS scenes.")
    parser.add_argument(
        "--scene",
        nargs="+",
        default=["figurines"],
        help=(
            "One or more LERF-OVS scenes to extract. Use `all` to extract every supported "
            f"scene: {', '.join(SCENE_NAMES)}."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where extracted scenes will be stored.",
    )
    parser.add_argument(
        "--archive-path",
        type=Path,
        help=(
            "Optional path to an existing lerf_ovs zip archive. If omitted, the script "
            "downloads the official release archive."
        ),
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the downloaded archive under <output-root>/archives/.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract requested scenes even if their target directories already exist.",
    )
    parser.add_argument(
        "--list-scenes",
        action="store_true",
        help="Print the supported scene names and exit.",
    )
    parser.add_argument(
        "--test-download-mb",
        type=int,
        default=0,
        help=(
            "Only download the first N MB from the archive to verify access. "
            "This does not extract scenes."
        ),
    )
    parser.add_argument(
        "--cacert",
        type=Path,
        help="Optional custom CA bundle path used to verify HTTPS certificates.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable HTTPS certificate verification. Use only in trusted proxy/internal environments.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args()

    if args.test_download_mb < 0:
        parser.error("--test-download-mb must be non-negative.")
    if args.cacert is not None and not args.cacert.is_file():
        parser.error(f"--cacert file does not exist: {args.cacert}")
    return args


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def normalize_requested_scenes(values: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    requested = [value.strip() for value in values if value and value.strip()]
    if not requested:
        raise ValueError("At least one --scene value is required.")

    if any(value.lower() == "all" for value in requested):
        return list(SCENE_NAMES)

    valid = set(SCENE_NAMES)
    invalid = [value for value in requested if value not in valid]
    if invalid:
        raise ValueError(
            "Unknown scene(s): "
            + ", ".join(invalid)
            + ". Supported scenes: "
            + ", ".join(SCENE_NAMES)
        )

    for value in requested:
        if value not in seen:
            normalized.append(value)
            seen.add(value)
    return normalized


def _format_size(num_bytes: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{num_bytes}B"


def _copy_with_progress(response, output_path: Path, expected_bytes: int | None = None) -> int:
    chunk_size = 1024 * 1024
    downloaded = 0
    started_at = time.time()
    last_reported = started_at - 1.0

    with open(output_path, "wb") as handle:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)

            now = time.time()
            if now - last_reported < 0.2 and expected_bytes is not None and downloaded < expected_bytes:
                continue
            last_reported = now
            elapsed = max(now - started_at, 1e-6)
            speed = downloaded / elapsed
            if expected_bytes:
                percent = min(downloaded / expected_bytes, 1.0) * 100.0
                status = (
                    f"\rDownloading: {percent:5.1f}% "
                    f"({_format_size(downloaded)}/{_format_size(expected_bytes)}) "
                    f"at {_format_size(int(speed))}/s"
                )
            else:
                status = f"\rDownloading: {_format_size(downloaded)} at {_format_size(int(speed))}/s"
            print(status, end="", flush=True)

    print()
    return downloaded


def build_ssl_context(cacert: Path | None = None, insecure: bool = False) -> ssl.SSLContext:
    if insecure:
        return ssl._create_unverified_context()
    if cacert is not None:
        return ssl.create_default_context(cafile=str(cacert))
    return ssl.create_default_context()


def _build_google_drive_opener(ssl_context: ssl.SSLContext | None = None) -> urllib.request.OpenerDirector:
    cookie_jar = http.cookiejar.CookieJar()
    handlers = [urllib.request.HTTPCookieProcessor(cookie_jar)]
    if ssl_context is not None:
        handlers.append(urllib.request.HTTPSHandler(context=ssl_context))
    return urllib.request.build_opener(*handlers)


def _extract_confirm_token(response_url: str, body: bytes) -> str | None:
    parsed = urllib.parse.urlparse(response_url)
    query = urllib.parse.parse_qs(parsed.query)
    if "confirm" in query and query["confirm"]:
        return query["confirm"][-1]

    text = body.decode("utf-8", errors="ignore")
    for pattern in (
        r'name="confirm" value="([^"]+)"',
        r'confirm=([0-9A-Za-z_\-]+)',
        r'"confirm":"([^"]+)"',
    ):
        match = re.search(pattern, text)
        if match:
            return html.unescape(match.group(1))
    return None


def _open_google_drive_download(request_or_url, ssl_context: ssl.SSLContext | None = None):
    opener = _build_google_drive_opener(ssl_context=ssl_context)
    try:
        response = opener.open(request_or_url, timeout=DEFAULT_TIMEOUT_SECONDS)
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", None)
        if isinstance(reason, ssl.SSLCertVerificationError):
            raise RuntimeError(
                "HTTPS certificate verification failed while downloading LERF-OVS. "
                "If your network uses a proxy with a custom root certificate, rerun with "
                "--cacert /path/to/ca.pem. As a last resort, use --insecure to disable "
                "certificate verification."
            ) from exc
        raise
    content_type = response.headers.get("Content-Type", "")
    if "text/html" not in content_type:
        return response

    preview = response.read()
    token = _extract_confirm_token(response.geturl(), preview)
    if token is None:
        raise RuntimeError(
            "Google Drive returned an HTML confirmation page, but no download token "
            "could be extracted. Provide --archive-path with a manually downloaded archive."
        )

    parsed = urllib.parse.urlparse(response.geturl())
    params = urllib.parse.parse_qs(parsed.query)
    params["confirm"] = [token]
    confirmed_url = urllib.parse.urlunparse(
        parsed._replace(query=urllib.parse.urlencode(params, doseq=True))
    )

    if isinstance(request_or_url, urllib.request.Request):
        confirmed_request = urllib.request.Request(confirmed_url, headers=dict(request_or_url.header_items()))
        return opener.open(confirmed_request, timeout=DEFAULT_TIMEOUT_SECONDS)
    return opener.open(confirmed_url, timeout=DEFAULT_TIMEOUT_SECONDS)


def download_file(url: str, output_path: Path, cacert: Path | None = None, insecure: bool = False) -> None:
    logging.info("Downloading %s", url)
    with _open_google_drive_download(url, ssl_context=build_ssl_context(cacert=cacert, insecure=insecure)) as response:
        content_length = response.headers.get("Content-Length")
        expected_bytes = int(content_length) if content_length is not None else None
        _copy_with_progress(response, output_path, expected_bytes=expected_bytes)


def download_partial_file(url: str, output_path: Path, megabytes: int, cacert: Path | None = None, insecure: bool = False) -> int:
    if megabytes <= 0:
        raise ValueError("megabytes must be positive")

    num_bytes = megabytes * 1024 * 1024
    end_byte = num_bytes - 1
    request = urllib.request.Request(url, headers={"Range": f"bytes=0-{end_byte}"})
    logging.info("Testing partial download: first %d MB from %s", megabytes, url)
    try:
        with _open_google_drive_download(request, ssl_context=build_ssl_context(cacert=cacert, insecure=insecure)) as response:
            bytes_downloaded = _copy_with_progress(response, output_path, expected_bytes=num_bytes)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            "Server did not accept the test Range request. Try downloading the full archive "
            "without --test-download-mb."
        ) from exc
    return bytes_downloaded


def find_scene_root(parts: tuple[str, ...], scenes: set[str]) -> tuple[str, int] | None:
    for index, part in enumerate(parts):
        if part in scenes:
            return part, index
    return None


def extract_scenes(zip_path: Path, scenes: Iterable[str], output_root: Path, force: bool = False) -> dict[str, int]:
    requested = list(scenes)
    requested_set = set(requested)
    extracted_counts = {scene: 0 for scene in requested}

    if force:
        for scene in requested:
            scene_dir = output_root / scene
            if scene_dir.exists():
                logging.info("Removing existing scene directory %s", scene_dir)
                shutil.rmtree(scene_dir)

    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            raw_name = info.filename.replace("\\", "/")
            parts = tuple(part for part in Path(raw_name).parts if part not in ("", "."))
            if not parts:
                continue

            scene_root = find_scene_root(parts, requested_set)
            if scene_root is None:
                continue
            scene_name, scene_index = scene_root
            relative_parts = parts[scene_index + 1 :]
            destination = output_root / scene_name
            if relative_parts:
                destination = destination.joinpath(*relative_parts)

            if info.is_dir() or not relative_parts:
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, open(destination, "wb") as sink:
                    shutil.copyfileobj(source, sink)
            extracted_counts[scene_name] += 1

    missing = [scene for scene, count in extracted_counts.items() if count == 0]
    if missing:
        raise RuntimeError(
            "Requested scene(s) were not found in archive: "
            + ", ".join(missing)
            + ". The archive layout may have changed."
        )
    return extracted_counts


def validate_scene_layout(scene_dir: Path) -> None:
    if not scene_dir.is_dir():
        raise RuntimeError(f"Missing scene directory: {scene_dir}")
    files = [path for path in scene_dir.rglob("*") if path.is_file()]
    if not files:
        raise RuntimeError(f"Scene directory is empty: {scene_dir}")

    has_images = any("images" in path.parts for path in files)
    has_pose_or_sparse = any(
        path.name in {"transforms.json", "poses_bounds.npy", "cameras.bin", "images.bin", "points3D.bin"}
        or "sparse" in path.parts
        for path in files
    )
    if not has_images:
        logging.warning("Did not detect an images directory under %s", scene_dir)
    if not has_pose_or_sparse:
        logging.warning("Did not detect transforms/poses/COLMAP metadata under %s", scene_dir)


def print_supported_scenes() -> None:
    print("Supported LERF-OVS scenes:")
    for scene in SCENE_NAMES:
        print(f"  - {scene}: {SCENE_DESCRIPTIONS.get(scene, '')}")


def print_next_steps(output_root: Path, scenes: Iterable[str]) -> None:
    print("\nLERF-OVS scene(s) are ready:")
    for scene in scenes:
        print(f"  {output_root / scene}")
    print("\nSuggested next step:")
    print("  Point your training/evaluation config at one of the extracted scene directories.")


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    if args.list_scenes:
        print_supported_scenes()
        return 0

    try:
        scenes = normalize_requested_scenes(args.scene)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.test_download_mb > 0:
        partial_path = args.output_root / f"{ARCHIVE_NAME}.partial"
        bytes_downloaded = download_partial_file(ARCHIVE_URL, partial_path, args.test_download_mb, cacert=args.cacert, insecure=args.insecure)
        print(
            "\nPartial download test succeeded:\n"
            f"  file: {partial_path}\n"
            f"  bytes: {bytes_downloaded}\n"
            "  note: this only verifies URL/network access; it does not extract scenes."
        )
        return 0

    existing = [scene for scene in scenes if (args.output_root / scene).exists()]
    if existing and not args.force:
        logging.info(
            "Skipping existing scene(s) without --force: %s",
            ", ".join(existing),
        )
        remaining = [scene for scene in scenes if scene not in existing]
    else:
        remaining = list(scenes)

    for scene in existing:
        validate_scene_layout(args.output_root / scene)

    if not remaining:
        print_next_steps(args.output_root, scenes)
        return 0

    archive_path = args.archive_path
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    try:
        if archive_path is None:
            if args.keep_archive:
                archive_dir = args.output_root / "archives"
                archive_dir.mkdir(parents=True, exist_ok=True)
                archive_path = archive_dir / ARCHIVE_NAME
            else:
                temp_dir = tempfile.TemporaryDirectory()
                archive_path = Path(temp_dir.name) / ARCHIVE_NAME

            if args.force or not archive_path.exists():
                download_file(ARCHIVE_URL, archive_path, cacert=args.cacert, insecure=args.insecure)
            else:
                logging.info("Reusing existing archive at %s", archive_path)

        extract_counts = extract_scenes(archive_path, remaining, args.output_root, force=args.force)
        for scene in remaining:
            validate_scene_layout(args.output_root / scene)
            logging.info("Prepared scene %s at %s (%d archive entries)", scene, args.output_root / scene, extract_counts[scene])
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    print_next_steps(args.output_root, scenes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
