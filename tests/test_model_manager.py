"""Tests for the ONNX model manager."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from recognizex.config import Settings
from recognizex.ml.model_manager import MODEL_REGISTRY, OnnxModelManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "device": "cpu",
        "accept_insightface_license": False,
        "models_dir": "/tmp/recognizex_test_models",
        "model_ttl": 300,
        "intra_op_threads": 0,
        "inter_op_threads": 1,
        "gpu_mem_limit": 2_147_483_648,
        "max_concurrent": 2,
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Model registry tests
# ---------------------------------------------------------------------------


class TestModelRegistry:
    def test_known_model_lookup(self) -> None:
        spec = MODEL_REGISTRY["auraface_v1"]
        assert spec.name == "auraface_v1"
        assert spec.task == "face_recognition"

    def test_unknown_model_raises_keyerror(self) -> None:
        with pytest.raises(KeyError):
            MODEL_REGISTRY["nonexistent_model"]

    def test_insightface_flag_correct(self) -> None:
        assert MODEL_REGISTRY["scrfd_10g_kps"].insightface is True
        assert MODEL_REGISTRY["w600k_r50"].insightface is True
        assert MODEL_REGISTRY["auraface_v1"].insightface is False
        assert MODEL_REGISTRY["retinaface_resnet34"].insightface is False

    def test_registry_has_five_models(self) -> None:
        assert len(MODEL_REGISTRY) == 5


# ---------------------------------------------------------------------------
# OnnxModelManager tests
# ---------------------------------------------------------------------------


class TestOnnxModelManager:
    @patch("recognizex.ml.model_manager.hf_hub_download")
    def test_ensure_downloaded_calls_hf_hub_download(self, mock_download: MagicMock) -> None:
        mock_download.return_value = "/tmp/recognizex_test_models/glintr100.onnx"
        settings = _make_settings()
        mgr = OnnxModelManager(settings)

        path = mgr.ensure_downloaded("auraface_v1")

        mock_download.assert_called_once_with(
            repo_id="fal/AuraFace-v1",
            filename="glintr100.onnx",
            subfolder=None,
            local_dir="/tmp/recognizex_test_models",
        )
        assert path == Path("/tmp/recognizex_test_models/glintr100.onnx")

    @patch("recognizex.ml.model_manager.hf_hub_download")
    def test_ensure_downloaded_skips_existing(self, mock_download: MagicMock, tmp_path: Path) -> None:
        model_file = tmp_path / "glintr100.onnx"
        model_file.touch()

        settings = _make_settings(models_dir=str(tmp_path))
        mgr = OnnxModelManager(settings)
        # Simulate a previous download by setting the cached path.
        mgr._model_paths["auraface_v1"] = model_file

        path = mgr.ensure_downloaded("auraface_v1")

        mock_download.assert_not_called()
        assert path == model_file

    def test_insightface_blocked_without_license(self) -> None:
        settings = _make_settings(accept_insightface_license=False)
        mgr = OnnxModelManager(settings)

        with pytest.raises(RuntimeError, match="RECOGNIZEX_ACCEPT_INSIGHTFACE_LICENSE"):
            mgr.ensure_downloaded("w600k_r50")

    @patch("recognizex.ml.model_manager.hf_hub_download")
    def test_insightface_allowed_with_license(self, mock_download: MagicMock) -> None:
        mock_download.return_value = "/tmp/recognizex_test_models/models/buffalo_l/w600k_r50.onnx"
        settings = _make_settings(accept_insightface_license=True)
        mgr = OnnxModelManager(settings)

        path = mgr.ensure_downloaded("w600k_r50")

        assert path == Path("/tmp/recognizex_test_models/models/buffalo_l/w600k_r50.onnx")
        mock_download.assert_called_once()

    @patch("recognizex.ml.model_manager.InferenceSession")
    @patch("recognizex.ml.model_manager.hf_hub_download")
    def test_get_session_creates_and_caches(self, mock_download: MagicMock, mock_session_cls: MagicMock) -> None:
        mock_download.return_value = "/tmp/recognizex_test_models/glintr100.onnx"
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        settings = _make_settings()
        mgr = OnnxModelManager(settings)

        session1 = mgr.get_session("auraface_v1")
        session2 = mgr.get_session("auraface_v1")

        assert session1 is mock_session
        assert session2 is mock_session
        mock_session_cls.assert_called_once()

    @patch("recognizex.ml.model_manager.InferenceSession")
    @patch("recognizex.ml.model_manager.hf_hub_download")
    def test_get_loaded_models(self, mock_download: MagicMock, mock_session_cls: MagicMock) -> None:
        mock_download.return_value = "/tmp/recognizex_test_models/glintr100.onnx"
        settings = _make_settings()
        mgr = OnnxModelManager(settings)

        assert mgr.get_loaded_models() == []
        mgr.get_session("auraface_v1")
        assert mgr.get_loaded_models() == ["auraface_v1"]

    @patch("recognizex.ml.model_manager.InferenceSession")
    @patch("recognizex.ml.model_manager.hf_hub_download")
    def test_unload_idle_models_removes_expired(self, mock_download: MagicMock, mock_session_cls: MagicMock) -> None:
        mock_download.return_value = "/tmp/recognizex_test_models/glintr100.onnx"
        settings = _make_settings(model_ttl=1)
        mgr = OnnxModelManager(settings)
        mgr.get_session("auraface_v1")

        # Fake the last_used time to be in the past.
        mgr._sessions["auraface_v1"].last_used = time.monotonic() - 10

        mgr.unload_idle_models()
        assert mgr.get_loaded_models() == []

    def test_unload_idle_skipped_when_ttl_zero(self) -> None:
        settings = _make_settings(model_ttl=0)
        mgr = OnnxModelManager(settings)
        # Should not raise or do anything.
        mgr.unload_idle_models()
        assert mgr.get_loaded_models() == []

    def test_provider_building_cpu(self) -> None:
        settings = _make_settings(device="cpu")
        mgr = OnnxModelManager(settings)
        assert mgr._providers == ["CPUExecutionProvider"]

    def test_provider_building_cuda(self) -> None:
        settings = _make_settings(device="cuda")
        mgr = OnnxModelManager(settings)
        assert len(mgr._providers) == 2
        provider_name, provider_opts = mgr._providers[0]  # type: ignore[misc]
        assert provider_name == "CUDAExecutionProvider"
        assert provider_opts["device_id"] == 0
        assert mgr._providers[1] == "CPUExecutionProvider"

    def test_provider_building_openvino(self) -> None:
        settings = _make_settings(device="openvino")
        mgr = OnnxModelManager(settings)
        assert len(mgr._providers) == 2
        provider_name, _provider_opts = mgr._providers[0]  # type: ignore[misc]
        assert provider_name == "OpenVINOExecutionProvider"
        assert mgr._providers[1] == "CPUExecutionProvider"

    @patch("recognizex.ml.model_manager.InferenceSession")
    @patch("recognizex.ml.model_manager.hf_hub_download")
    def test_shutdown_clears_sessions(self, mock_download: MagicMock, mock_session_cls: MagicMock) -> None:
        mock_download.return_value = "/tmp/recognizex_test_models/glintr100.onnx"
        settings = _make_settings()
        mgr = OnnxModelManager(settings)
        mgr.get_session("auraface_v1")
        assert len(mgr.get_loaded_models()) == 1

        mgr.shutdown()
        assert mgr.get_loaded_models() == []

    def test_unknown_model_raises_keyerror(self) -> None:
        settings = _make_settings()
        mgr = OnnxModelManager(settings)
        with pytest.raises(KeyError, match="Unknown model"):
            mgr.ensure_downloaded("totally_fake_model")
