#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Download filtered subsets of Objaverse-XL with the official Python API.

References:
- https://objaverse.allenai.org/docs/objaverse-xl
- https://github.com/allenai/objaverse-xl
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Iterable, Sequence

import fcntl

DEFAULT_DATA_ROOT = Path("data/objaverse")
DEFAULT_SUMMARY_PATH = DEFAULT_DATA_ROOT / "download_results.json"
DEFAULT_FAILED_LOG_PATH = DEFAULT_DATA_ROOT / "failed_objects.csv"
DEFAULT_PROGRESS_PATH = DEFAULT_DATA_ROOT / "download_progress.json"
DEFAULT_ANNOTATIONS_PATH = DEFAULT_DATA_ROOT / "annotations.parquet"
STATE_DIR_NAME = ".objaverse_download"
DOWNLOADED_IDS_FILENAME = "downloaded_file_identifiers.txt"
HF_MIRROR_ENV_NAME = "SAM3D_OBJAVERSE_HF_ENDPOINT"
HF_UPSTREAM_BASE = "https://huggingface.co"
VALID_SOURCES = ("github", "thingiverse", "sketchfab", "smithsonian")
VALID_GITHUB_REPO_FORMATS = ("files", "zip", "tar", "tar.gz")
MIRROR_PRESETS = {
    "none": {
        "description": "Use the original upstream endpoints.",
    },
    "huggingface": {
        "description": "Use a Hugging Face mirror for annotation/object downloads.",
        "hf_endpoint": "https://hf-mirror.com",
    },
    "china-full": {
        "description": "Use a Hugging Face mirror and a GitHub clone mirror.",
        "hf_endpoint": "https://hf-mirror.com",
        "git_instead_of": (
            "https://ghproxy.com/https://github.com/",
            "https://github.com/",
        ),
    },
    "proxy": {
        "description": "Route requests through a user-provided HTTP/HTTPS proxy.",
    },
}

_REQUEST_PATCHED = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download Objaverse-XL annotations and filtered object subsets using "
            "the official objaverse.xl API."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Directory used for cached annotations and downloaded objects.",
    )
    parser.add_argument(
        "--source",
        nargs="+",
        choices=VALID_SOURCES,
        help="Restrict downloads to one or more Objaverse-XL sources.",
    )
    parser.add_argument(
        "--file-type",
        nargs="+",
        metavar="EXT",
        help="Restrict downloads to one or more file types, e.g. glb obj fbx.",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        help="Maximum number of filtered objects to keep for this run.",
    )
    parser.add_argument(
        "--uids-file",
        type=Path,
        help=(
            "Optional text file with one identifier per line. Each line may be a full "
            "fileIdentifier or a Sketchfab UID."
        ),
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle filtered annotations before applying --max-objects.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used with --shuffle.",
    )
    parser.add_argument(
        "--annotations-only",
        action="store_true",
        help="Only fetch and cache annotations; do not download objects.",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of worker processes passed to objaverse.xl.download_objects.",
    )
    parser.add_argument(
        "--refresh-annotations",
        action="store_true",
        help="Force re-download of remote annotation parquet files.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Path to write the run summary JSON.",
    )
    parser.add_argument(
        "--failed-log",
        type=Path,
        default=DEFAULT_FAILED_LOG_PATH,
        help="Path to write the failed object CSV for this run.",
    )
    parser.add_argument(
        "--progress-json",
        type=Path,
        default=DEFAULT_PROGRESS_PATH,
        help="Path to write the run progress JSON.",
    )
    parser.add_argument(
        "--annotations-path",
        type=Path,
        default=DEFAULT_ANNOTATIONS_PATH,
        help="Path to persist the merged annotations parquet.",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Skip fileIdentifiers that were already completed by a previous run of this "
            "script. Enabled by default."
        ),
    )
    parser.add_argument(
        "--github-repo-format",
        choices=VALID_GITHUB_REPO_FORMATS,
        default="files",
        help=(
            "Storage format for GitHub-backed objects. The official downloader needs "
            "this to persist GitHub source files."
        ),
    )
    parser.add_argument(
        "--mirror",
        choices=tuple(MIRROR_PRESETS),
        default="none",
        help=(
            "Enable a built-in mirror preset. `huggingface` only rewrites Hugging "
            "Face traffic, `china-full` also mirrors GitHub clones, and `proxy` "
            "uses --proxy-url."
        ),
    )
    parser.add_argument(
        "--proxy-url",
        type=str,
        help=(
            "HTTP/HTTPS proxy URL used with --mirror proxy, for example "
            "http://127.0.0.1:7890."
        ),
    )
    parser.add_argument(
        "--hf-endpoint",
        type=str,
        help=(
            "Optional Hugging Face mirror endpoint, for example https://hf-mirror.com. "
            "If omitted, the official default endpoint is used."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    if args.max_objects is not None and args.max_objects <= 0:
        parser.error("--max-objects must be a positive integer.")
    if args.processes <= 0:
        parser.error("--processes must be a positive integer.")
    if args.mirror == "proxy" and not args.proxy_url:
        parser.error("--proxy-url is required when --mirror proxy is enabled.")

    return args


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def import_objaverse_xl():
    try:
        import objaverse.xl as oxl
    except ImportError as exc:
        raise SystemExit(
            "objaverse is not installed. Install it with `pip install objaverse` "
            "or reinstall this project after updating requirements."
        ) from exc
    return oxl


def normalize_file_types(file_types: Sequence[str] | None) -> set[str]:
    if not file_types:
        return set()
    normalized = set()
    for value in file_types:
        value = value.strip().lower()
        if not value:
            continue
        normalized.add(value[1:] if value.startswith(".") else value)
    return normalized


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def resolve_output_path(path: Path, data_root: Path) -> Path:
    defaults = {
        DEFAULT_SUMMARY_PATH: DEFAULT_SUMMARY_PATH.name,
        DEFAULT_FAILED_LOG_PATH: DEFAULT_FAILED_LOG_PATH.name,
        DEFAULT_PROGRESS_PATH: DEFAULT_PROGRESS_PATH.name,
        DEFAULT_ANNOTATIONS_PATH: DEFAULT_ANNOTATIONS_PATH.name,
    }
    return data_root / defaults.get(path, path)


def current_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_endpoint(endpoint: str) -> str:
    endpoint = endpoint.strip()
    if not endpoint:
        raise ValueError("Mirror endpoint cannot be empty.")
    if "://" not in endpoint:
        endpoint = f"https://{endpoint}"
    return endpoint.rstrip("/")


def rewrite_hf_url(url: str) -> str:
    endpoint = os.environ.get(HF_MIRROR_ENV_NAME) or os.environ.get("HF_ENDPOINT")
    if not endpoint or not url.startswith(HF_UPSTREAM_BASE):
        return url

    upstream = urllib.parse.urlsplit(HF_UPSTREAM_BASE)
    mirror = urllib.parse.urlsplit(normalize_endpoint(endpoint))
    target = urllib.parse.urlsplit(url)
    if target.netloc != upstream.netloc:
        return url

    return urllib.parse.urlunsplit(
        (mirror.scheme, mirror.netloc, target.path, target.query, target.fragment)
    )


def apply_hf_request_mirror() -> None:
    global _REQUEST_PATCHED
    if _REQUEST_PATCHED:
        return

    endpoint = os.environ.get(HF_MIRROR_ENV_NAME) or os.environ.get("HF_ENDPOINT")
    if not endpoint:
        return

    import requests

    original_request = requests.sessions.Session.request
    if not getattr(original_request, "__sam3d_hf_mirror__", False):

        def mirrored_request(self, method, url, *args, **kwargs):
            if isinstance(url, str):
                url = rewrite_hf_url(url)
            return original_request(self, method, url, *args, **kwargs)

        mirrored_request.__sam3d_hf_mirror__ = True
        requests.sessions.Session.request = mirrored_request

    original_urlopen = urllib.request.urlopen
    if not getattr(original_urlopen, "__sam3d_hf_mirror__", False):

        def mirrored_urlopen(url, *args, **kwargs):
            if isinstance(url, str):
                url = rewrite_hf_url(url)
            return original_urlopen(url, *args, **kwargs)

        mirrored_urlopen.__sam3d_hf_mirror__ = True
        urllib.request.urlopen = mirrored_urlopen

    _REQUEST_PATCHED = True


def append_git_config_env(key: str, value: str) -> None:
    count = int(os.environ.get("GIT_CONFIG_COUNT", "0"))
    os.environ[f"GIT_CONFIG_KEY_{count}"] = key
    os.environ[f"GIT_CONFIG_VALUE_{count}"] = value
    os.environ["GIT_CONFIG_COUNT"] = str(count + 1)


def configure_mirror(args: argparse.Namespace) -> dict[str, str]:
    applied: dict[str, str] = {"mirror": args.mirror}
    preset = MIRROR_PRESETS[args.mirror]
    logging.info("Mirror preset: %s", args.mirror)
    logging.info("Mirror description: %s", preset["description"])

    hf_endpoint = args.hf_endpoint or preset.get("hf_endpoint")
    if hf_endpoint:
        normalized_hf_endpoint = normalize_endpoint(hf_endpoint)
        os.environ[HF_MIRROR_ENV_NAME] = normalized_hf_endpoint
        os.environ["HF_ENDPOINT"] = normalized_hf_endpoint
        apply_hf_request_mirror()
        applied["hf_endpoint"] = normalized_hf_endpoint
        logging.info("Using Hugging Face mirror endpoint %s", normalized_hf_endpoint)

    proxy_url = args.proxy_url
    if args.mirror == "proxy" and proxy_url:
        for env_name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            os.environ[env_name] = proxy_url
        applied["proxy_url"] = proxy_url
        logging.info("Using proxy=%s", proxy_url)

    git_instead_of = preset.get("git_instead_of")
    if git_instead_of:
        mirror_url, original_url = git_instead_of
        append_git_config_env(f"url.{mirror_url}.insteadof", original_url)
        applied["git_mirror"] = mirror_url
        logging.info("Using Git mirror %s for %s", mirror_url, original_url)

    return applied


apply_hf_request_mirror()


def load_identifier_filter(path: Path | None) -> set[str]:
    if path is None:
        return set()
    identifiers: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            identifiers.add(line)
    return identifiers


def load_completed_identifiers(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def append_completed_identifiers(path: Path, file_identifiers: Iterable[str]) -> None:
    identifiers = sorted({value for value in file_identifiers if value})
    if not identifiers:
        return
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as handle:
        for value in identifiers:
            handle.write(f"{value}\n")


def write_json(path: Path, payload: dict) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def append_jsonl(path: Path, payload: dict) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def record_event(event_path: str, event_type: str, **kwargs) -> None:
    payload = {"event": event_type, "timestamp": current_utc_iso(), **kwargs}
    append_jsonl(Path(event_path), payload)


def write_failed_log(path: Path, failures: Sequence[dict]) -> None:
    ensure_parent(path)
    fieldnames = ["file_identifier", "sha256", "metadata", "timestamp"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for failure in failures:
            writer.writerow(
                {
                    "file_identifier": failure.get("file_identifier", ""),
                    "sha256": failure.get("sha256", ""),
                    "metadata": json.dumps(failure.get("metadata", {}), ensure_ascii=False),
                    "timestamp": failure.get("timestamp", ""),
                }
            )


def apply_filters(annotations, args: argparse.Namespace, identifier_filter: set[str]):
    filtered = annotations
    if args.source:
        filtered = filtered[filtered["source"].isin(args.source)]

    file_types = normalize_file_types(args.file_type)
    if file_types:
        normalized_types = filtered["fileType"].fillna("").astype(str).str.lower().str.lstrip(".")
        filtered = filtered[normalized_types.isin(file_types)]

    if identifier_filter:
        file_identifiers = filtered["fileIdentifier"].fillna("").astype(str)
        sketchfab_uids = file_identifiers.where(filtered["source"] == "sketchfab", "").str.rsplit("/", n=1).str[-1]
        filtered = filtered[file_identifiers.isin(identifier_filter) | sketchfab_uids.isin(identifier_filter)]

    if args.shuffle:
        filtered = filtered.sample(frac=1.0, random_state=args.seed)

    if args.max_objects is not None:
        filtered = filtered.head(args.max_objects)

    return filtered.reset_index(drop=True)


def build_progress_payload(
    *,
    phase: str,
    total_annotations: int,
    filtered_objects: int,
    pending_objects: int,
    skipped_existing: int,
    success_count: int,
    modified_count: int,
    failed_count: int,
    sources: Sequence[str],
    started_at: str,
) -> dict:
    return {
        "phase": phase,
        "start_time": started_at,
        "last_update": current_utc_iso(),
        "total_annotations": total_annotations,
        "filtered_objects": filtered_objects,
        "pending_objects": pending_objects,
        "skipped_existing": skipped_existing,
        "downloaded_objects": success_count,
        "modified_objects": modified_count,
        "failed_objects": failed_count,
        "sources": list(sources),
    }


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    applied_mirror = configure_mirror(args)

    oxl = import_objaverse_xl()

    data_root = args.data_root.expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    summary_path = resolve_output_path(args.summary_json.expanduser(), data_root)
    failed_log_path = resolve_output_path(args.failed_log.expanduser(), data_root)
    progress_path = resolve_output_path(args.progress_json.expanduser(), data_root)
    annotations_path = resolve_output_path(args.annotations_path.expanduser(), data_root)

    state_dir = data_root / STATE_DIR_NAME
    completed_ids_path = state_dir / DOWNLOADED_IDS_FILENAME
    run_dir = state_dir / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    found_events_path = run_dir / "found.jsonl"
    modified_events_path = run_dir / "modified.jsonl"
    missing_events_path = run_dir / "missing.jsonl"

    started_at = current_utc_iso()

    logging.info("Loading Objaverse-XL annotations into %s", data_root)
    annotations = oxl.get_annotations(
        download_dir=str(data_root),
        refresh=args.refresh_annotations,
    )
    ensure_parent(annotations_path)
    annotations.to_parquet(annotations_path, index=False)
    logging.info("Saved merged annotations to %s", annotations_path)

    identifier_filter = (
        load_identifier_filter(args.uids_file.expanduser()) if args.uids_file else set()
    )
    filtered = apply_filters(annotations, args, identifier_filter)

    logging.info(
        "Loaded %s annotations; %s objects remain after filtering.",
        len(annotations),
        len(filtered),
    )

    if args.annotations_only:
        summary = {
            "status": "annotations_only",
            "start_time": started_at,
            "end_time": current_utc_iso(),
            "data_root": str(data_root),
            "annotations_path": str(annotations_path),
            "mirror": applied_mirror,
            "total_annotations": int(len(annotations)),
            "filtered_objects": int(len(filtered)),
            "sources": sorted(filtered["source"].unique().tolist()) if len(filtered) else [],
        }
        write_json(summary_path, summary)
        write_json(
            progress_path,
            build_progress_payload(
                phase="annotations_only",
                total_annotations=len(annotations),
                filtered_objects=len(filtered),
                pending_objects=0,
                skipped_existing=0,
                success_count=0,
                modified_count=0,
                failed_count=0,
                sources=summary["sources"],
                started_at=started_at,
            ),
        )
        logging.info("Annotations-only run complete.")
        return 0

    completed_ids = load_completed_identifiers(completed_ids_path) if args.skip_existing else set()
    skipped_existing = 0
    if completed_ids:
        before_count = len(filtered)
        filtered = filtered[~filtered["fileIdentifier"].isin(completed_ids)].reset_index(drop=True)
        skipped_existing = before_count - len(filtered)
        if skipped_existing:
            logging.info(
                "Skipped %s objects already completed by previous runs.",
                skipped_existing,
            )

    selected_sources = sorted(filtered["source"].unique().tolist()) if len(filtered) else []
    write_json(
        progress_path,
        build_progress_payload(
            phase="downloading",
            total_annotations=len(annotations),
            filtered_objects=len(filtered) + skipped_existing,
            pending_objects=len(filtered),
            skipped_existing=skipped_existing,
            success_count=0,
            modified_count=0,
            failed_count=0,
            sources=selected_sources,
            started_at=started_at,
        ),
    )

    if filtered.empty:
        summary = {
            "status": "no_matches",
            "start_time": started_at,
            "end_time": current_utc_iso(),
            "data_root": str(data_root),
            "annotations_path": str(annotations_path),
            "mirror": applied_mirror,
            "total_annotations": int(len(annotations)),
            "filtered_objects": 0,
            "skipped_existing": skipped_existing,
            "requested_sources": args.source or [],
            "requested_file_types": sorted(normalize_file_types(args.file_type)),
        }
        write_json(summary_path, summary)
        write_failed_log(failed_log_path, [])
        write_json(
            progress_path,
            build_progress_payload(
                phase="no_matches",
                total_annotations=len(annotations),
                filtered_objects=skipped_existing,
                pending_objects=0,
                skipped_existing=skipped_existing,
                success_count=0,
                modified_count=0,
                failed_count=0,
                sources=[],
                started_at=started_at,
            ),
        )
        logging.warning("No objects matched the requested filters.")
        return 0

    handle_found = partial(record_event, str(found_events_path), "found")
    handle_modified = partial(record_event, str(modified_events_path), "modified")
    handle_missing = partial(record_event, str(missing_events_path), "missing")

    download_kwargs: dict[str, str] = {}
    if "github" in selected_sources:
        download_kwargs["save_repo_format"] = args.github_repo_format

    logging.info(
        "Starting download of %s objects across %s sources with %s processes.",
        len(filtered),
        len(selected_sources),
        args.processes,
    )

    download_results = oxl.download_objects(
        filtered,
        download_dir=str(data_root),
        processes=args.processes,
        handle_found_object=handle_found,
        handle_modified_object=handle_modified,
        handle_missing_object=handle_missing,
        **download_kwargs,
    )

    found_events = read_jsonl(found_events_path)
    modified_events = read_jsonl(modified_events_path)
    missing_events = read_jsonl(missing_events_path)

    verified_identifiers = {
        row.get("file_identifier", "")
        for row in found_events + modified_events
        if row.get("file_identifier")
    }
    returned_identifiers = {file_identifier for file_identifier in download_results if file_identifier}
    successful_identifiers = returned_identifiers | verified_identifiers
    cached_identifiers = returned_identifiers - verified_identifiers

    append_completed_identifiers(completed_ids_path, successful_identifiers)
    write_failed_log(failed_log_path, missing_events)

    end_time = current_utc_iso()
    summary = {
        "status": "completed",
        "start_time": started_at,
        "end_time": end_time,
        "duration_seconds": (
            datetime.fromisoformat(end_time) - datetime.fromisoformat(started_at)
        ).total_seconds(),
        "data_root": str(data_root),
        "annotations_path": str(annotations_path),
        "mirror": applied_mirror,
        "total_annotations": int(len(annotations)),
        "filtered_objects": int(len(filtered) + skipped_existing),
        "attempted_objects": int(len(filtered)),
        "skipped_existing": int(skipped_existing),
        "successful_objects": int(len(successful_identifiers)),
        "cached_objects": int(len(cached_identifiers)),
        "modified_objects": int(len(modified_events)),
        "failed_objects": int(len(missing_events)),
        "returned_paths": int(len(download_results)),
        "sources": selected_sources,
        "file_types": sorted(filtered["fileType"].fillna("").astype(str).str.lower().str.lstrip(".").unique().tolist()),
        "summary_json": str(summary_path),
        "failed_log": str(failed_log_path),
        "progress_json": str(progress_path),
    }
    write_json(summary_path, summary)
    write_json(
        progress_path,
        build_progress_payload(
            phase="completed",
            total_annotations=len(annotations),
            filtered_objects=len(filtered) + skipped_existing,
            pending_objects=0,
            skipped_existing=skipped_existing,
            success_count=len(successful_identifiers),
            modified_count=len(modified_events),
            failed_count=len(missing_events),
            sources=selected_sources,
            started_at=started_at,
        ),
    )

    logging.info(
        "Run complete. success=%s, modified=%s, failed=%s, skipped_existing=%s",
        len(successful_identifiers),
        len(modified_events),
        len(missing_events),
        skipped_existing,
    )
    return 0 if not missing_events else 1


if __name__ == "__main__":
    raise SystemExit(main())
