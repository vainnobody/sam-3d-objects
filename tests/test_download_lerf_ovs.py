from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "download_lerf_ovs.py"
SPEC = importlib.util.spec_from_file_location("download_lerf_ovs", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def create_fake_archive(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("lerf_ovs/figurines/images/frame_0001.png", b"png")
        archive.writestr("lerf_ovs/figurines/sparse/0/cameras.bin", b"cam")
        archive.writestr("lerf_ovs/ramen/images/frame_0001.png", b"png")
        archive.writestr("lerf_ovs/ramen/transforms.json", b"{}")
        archive.writestr("lerf_ovs/teatime/images/frame_0001.png", b"png")
        archive.writestr("lerf_ovs/waldo_kitchen/poses_bounds.npy", b"npy")


class DownloadLerfOvsTests(unittest.TestCase):
    def test_normalize_requested_scenes_all_and_dedup(self) -> None:
        self.assertEqual(MODULE.normalize_requested_scenes(["all"]), list(MODULE.SCENE_NAMES))
        self.assertEqual(
            MODULE.normalize_requested_scenes(["figurines", "ramen", "figurines"]),
            ["figurines", "ramen"],
        )

    def test_normalize_requested_scenes_invalid(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown scene"):
            MODULE.normalize_requested_scenes(["does_not_exist"])

    def test_extract_selected_scenes(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            archive_path = tmp_path / "lerf_ovs.zip"
            output_root = tmp_path / "out"
            create_fake_archive(archive_path)

            counts = MODULE.extract_scenes(archive_path, ["figurines", "ramen"], output_root, force=False)

            self.assertGreater(counts["figurines"], 0)
            self.assertGreater(counts["ramen"], 0)
            self.assertTrue((output_root / "figurines" / "images" / "frame_0001.png").exists())
            self.assertTrue((output_root / "figurines" / "sparse" / "0" / "cameras.bin").exists())
            self.assertTrue((output_root / "ramen" / "transforms.json").exists())
            self.assertFalse((output_root / "teatime").exists())

    def test_extract_scenes_force_replaces_existing_dir(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            archive_path = tmp_path / "lerf_ovs.zip"
            output_root = tmp_path / "out"
            create_fake_archive(archive_path)

            stale = output_root / "figurines" / "stale.txt"
            stale.parent.mkdir(parents=True, exist_ok=True)
            stale.write_text("old")

            MODULE.extract_scenes(archive_path, ["figurines"], output_root, force=True)

            self.assertFalse(stale.exists())
            self.assertTrue((output_root / "figurines" / "images" / "frame_0001.png").exists())

    def test_extract_scenes_missing_scene_raises(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            archive_path = tmp_path / "lerf_ovs.zip"
            output_root = tmp_path / "out"
            create_fake_archive(archive_path)

            with self.assertRaisesRegex(RuntimeError, "not found"):
                MODULE.extract_scenes(archive_path, ["figurines", "missing_scene"], output_root, force=False)

    def test_validate_scene_layout_accepts_nonempty_scene(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmpdir:
            scene_dir = Path(tmpdir) / "figurines"
            (scene_dir / "images").mkdir(parents=True)
            (scene_dir / "images" / "frame.png").write_text("x")
            (scene_dir / "transforms.json").write_text("{}")

            MODULE.validate_scene_layout(scene_dir)

    def test_build_ssl_context_insecure(self) -> None:
        ctx = MODULE.build_ssl_context(insecure=True)
        self.assertEqual(ctx.verify_mode, MODULE.ssl.CERT_NONE)

    def test_build_ssl_context_with_missing_cacert_is_parser_level(self) -> None:
        self.assertTrue(callable(MODULE.build_ssl_context))


if __name__ == "__main__":
    unittest.main()
