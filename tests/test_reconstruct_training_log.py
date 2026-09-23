"""
Tests for reconstruct_training_log.py -- the recovery script that pulls
per-epoch train_loss/val_loss/val_auc back out of a fetched Kaggle JSON
log, for the one real case this project hit: training_log.csv itself
never fully downloaded after a mid-download connection reset (see that
script's module docstring).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruct_training_log import (
    extract_stdout_text,
    load_kaggle_log_text,
    parse_epoch_rows,
)

# A short, realistic slice in the exact format run_phase() actually prints,
# copied verbatim from the real fetched log for this project's run
# (2026-09-16 full push) -- not invented text.
_SAMPLE_LOG_TEXT = (
    "[entry] copying source tree: /kaggle/input/datasets/x -> /kaggle/working\n"
    "[A epoch 1/5 | global 1] train_loss=0.8336 val_loss=0.3303 val_auc=0.9521 *best* (6035.0s)\n"
    "[A epoch 2/5 | global 2] train_loss=0.4578 val_loss=0.2353 val_auc=0.9629 *best* (189.2s)\n"
    "[B epoch 1/25 | global 6] train_loss=0.1818 val_loss=0.1116 val_auc=0.9875 *best* (543.6s)\n"
    "[B epoch 6/25 | global 11] train_loss=0.0695 val_loss=0.0737 val_auc=0.9942  (543.9s)\n"
    "Training complete. Best epoch: 20, best val_auc: 0.9958\n"
)


def test_parse_epoch_rows_extracts_only_epoch_lines():
    rows = parse_epoch_rows(_SAMPLE_LOG_TEXT)
    assert len(rows) == 4  # not the "[entry] copying..." line, not the summary line
    assert [r["global_epoch"] for r in rows] == [1, 2, 6, 11]


def test_parse_epoch_rows_extracts_correct_values_and_best_flag():
    rows = parse_epoch_rows(_SAMPLE_LOG_TEXT)
    first = rows[0]
    assert first["phase"] == "A"
    assert first["epoch_in_phase"] == 1
    assert first["train_loss"] == 0.8336
    assert first["val_loss"] == 0.3303
    assert first["val_auc"] == 0.9521
    assert first["epoch_time_sec"] == 6035.0
    assert first["is_best"] is True

    not_best = rows[3]  # "[B epoch 6/25 | global 11] ... (no *best*)"
    assert not_best["global_epoch"] == 11
    assert not_best["is_best"] is False


def test_parse_epoch_rows_empty_on_no_matching_lines():
    assert parse_epoch_rows("nothing relevant here\njust noise\n") == []


def test_parse_epoch_rows_is_order_preserving_not_sorted():
    # Deliberately out of numeric order -- the parser must not silently sort.
    text = (
        "[B epoch 2/25 | global 7] train_loss=0.1 val_loss=0.1 val_auc=0.9  (1.0s)\n"
        "[B epoch 1/25 | global 6] train_loss=0.2 val_loss=0.2 val_auc=0.8  (1.0s)\n"
    )
    rows = parse_epoch_rows(text)
    assert [r["global_epoch"] for r in rows] == [7, 6]


def test_extract_stdout_text_from_kaggle_json_array():
    records = [
        {"stream_name": "stdout", "time": 1.0, "data": "[entry] hello\n"},
        {"stream_name": "stdout", "time": 2.0, "data": "[A epoch 1/5 | global 1] train_loss=0.5 val_loss=0.4 val_auc=0.9 *best* (10.0s)\n"},
        {"stream_name": "stderr", "time": 3.0, "data": "some warning\n"},
    ]
    text = extract_stdout_text(json.dumps(records))
    rows = parse_epoch_rows(text)
    assert len(rows) == 1
    assert rows[0]["global_epoch"] == 1


def test_extract_stdout_text_falls_back_to_plain_text_when_not_json():
    # Not valid JSON -- e.g. someone points this at a raw piped log instead
    # of the `kernels logs` JSON array output. Must not raise, must pass
    # the text through unchanged so parse_epoch_rows can still work on it.
    plain = "[A epoch 1/5 | global 1] train_loss=0.5 val_loss=0.4 val_auc=0.9 *best* (10.0s)\n"
    assert extract_stdout_text(plain) == plain


def test_load_kaggle_log_text_handles_cp1252_bytes(tmp_path):
    # The real bug hit on this project: `kernels logs > file.txt` on Windows
    # writes cp1252, and a GPU name like "Tesla T4\nTesla T4" wrapped in a
    # JSON string with a smart quote breaks a naive UTF-8 read.
    path = tmp_path / "log.txt"
    text_with_smart_quote = '[{"stream_name": "stdout", "time": 1.0, "data": "caf’e\\n"}]'
    path.write_bytes(text_with_smart_quote.encode("cp1252"))

    loaded = load_kaggle_log_text(path)
    assert json.loads(loaded)  # decodes and is valid JSON again


def test_load_kaggle_log_text_handles_plain_utf8(tmp_path):
    path = tmp_path / "log.txt"
    path.write_bytes('[{"data": "hello ☃\\n"}]'.encode("utf-8"))
    loaded = load_kaggle_log_text(path)
    assert json.loads(loaded)
