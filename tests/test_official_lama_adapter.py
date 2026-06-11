from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def test_official_lama_paths_accept_single_component_checkpoint(monkeypatch) -> None:
    pytest.importorskip("cv2")
    import src.restoration.official_lama_adapter as adapter

    with monkeypatch.context() as environment:
        environment.setenv("OFFICIAL_LAMA_REPO", "lama_repo")
        environment.setenv("OFFICIAL_LAMA_CHECKPOINT", "best.ckpt")
        environment.delenv("OFFICIAL_LAMA_MODEL_DIR", raising=False)

        adapter = importlib.reload(adapter)

        assert adapter.OFFICIAL_LAMA_REPO == Path("lama_repo")
        assert adapter.OFFICIAL_LAMA_CHECKPOINT == Path("best.ckpt")
        assert adapter.OFFICIAL_LAMA_MODEL_DIR == Path(".")

    importlib.reload(adapter)
