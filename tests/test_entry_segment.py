"""
Tests for the parts of kaggle/entry_segment.py that don't require a real
Kaggle container. Everything reused from entry.py (source-tree copy,
GPU/torch compat, /kaggle/input search) is already covered by
tests/test_entry.py and not re-tested here -- only what's new to this file.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))

import entry_segment


def test_find_checkpoint_file_at_root(tmp_path):
    mount = tmp_path / "mount"
    mount.mkdir()
    ckpt = mount / "checkpoint_best.pt"
    ckpt.write_bytes(b"fake")
    assert entry_segment._find_checkpoint_file(mount) == ckpt


def test_find_checkpoint_file_nested():
    """Same lesson as entry.py's own input-mount history -- don't assume a
    fixed depth for a dataset's contents."""
    with tempfile.TemporaryDirectory() as tmp:
        mount = Path(tmp) / "mount"
        (mount / "datasets" / "dme-oct-classifier-ckpt").mkdir(parents=True)
        ckpt = mount / "datasets" / "dme-oct-classifier-ckpt" / "checkpoint_best.pt"
        ckpt.write_bytes(b"fake")
        assert entry_segment._find_checkpoint_file(mount) == ckpt


def test_find_checkpoint_file_raises_when_absent(tmp_path):
    mount = tmp_path / "mount"
    mount.mkdir()
    (mount / "readme.txt").write_text("no checkpoint here")
    try:
        entry_segment._find_checkpoint_file(mount)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_find_checkpoint_file_warns_and_picks_first_on_multiple(tmp_path, capsys):
    mount = tmp_path / "mount"
    mount.mkdir()
    a = mount / "checkpoint_a.pt"
    b = mount / "checkpoint_b.pt"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    found = entry_segment._find_checkpoint_file(mount)
    assert found == sorted([a, b])[0]
    assert "multiple" in capsys.readouterr().out.lower()


def test_segment_smoke_marker_name_is_distinct_from_classifier_marker():
    """Guards against the exact cross-pipeline stomping bug this marker
    name was chosen to avoid -- see the module docstring."""
    import entry

    assert entry_segment.SEGMENT_SMOKE_MARKER_NAME != "SMOKE_MODE"
    assert entry_segment.SEGMENT_SMOKE_MARKER_NAME != getattr(entry, "SMOKE_MARKER_NAME", "SMOKE_MODE")
