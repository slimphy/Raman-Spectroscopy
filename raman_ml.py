"""Stable, serialized Raman spectrum inference.

Drop-in replacement for the original ``raman_ml.py``. The same model classes
and ``RamanMLProcessor.enhance_spectrum`` API are retained.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

_LOG = logging.getLogger(__name__)


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=1),
            nn.BatchNorm1d(ch),
            nn.ReLU(),
            nn.Conv1d(ch, ch, 3, padding=1),
            nn.BatchNorm1d(ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x + self.net(x))


class RamanSRNetV21(nn.Module):
    def __init__(self, num_features: int = 64, num_blocks: int = 6):
        super().__init__()
        self.head = nn.Sequential(nn.Conv1d(1, num_features, 9, padding=4), nn.ReLU())
        self.body = nn.Sequential(*[ResBlock(num_features) for _ in range(num_blocks)])
        self.tail = nn.Sequential(
            nn.Conv1d(num_features, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 1, 3, padding=1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tail(self.body(self.head(x)))


class RamanMLProcessor:
    """Thread-safe spectrum ML processor.

    A single instance is shared by live view and mapping in the current UI.
    Serializing inference prevents concurrent CUDA/CPU execution against the
    same module and avoids repeated host/device synchronization storms.
    """

    def __init__(self, model_path: str):
        force_cpu = os.getenv("RAMAN_ML_FORCE_CPU", "0") == "1"
        self.device = torch.device(
            "cpu" if force_cpu or not torch.cuda.is_available() else "cuda"
        )
        self.model: Optional[RamanSRNetV21] = None
        self._lock = threading.RLock()
        self._model_path = str(model_path)
        self.last_error: Optional[str] = None
        self._load_model(self._model_path)

    def _load_state_dict(self, model_path: str):
        # weights_only blocks arbitrary object deserialization on modern PyTorch.
        try:
            return torch.load(model_path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(model_path, map_location="cpu")

    def _load_model(self, model_path: str) -> None:
        with self._lock:
            try:
                path = Path(model_path)
                if not path.exists():
                    raise FileNotFoundError(path)

                state_dict = self._load_state_dict(str(path))
                if not isinstance(state_dict, dict):
                    raise TypeError("모델 파일이 state_dict 형식이 아닙니다.")

                head_weight = state_dict.get("head.0.weight")
                num_features = int(head_weight.shape[0]) if head_weight is not None else 64
                num_blocks = len(
                    [
                        key
                        for key in state_dict
                        if key.startswith("body.") and key.endswith("net.0.weight")
                    ]
                )
                if num_blocks <= 0:
                    num_blocks = 6

                model = RamanSRNetV21(
                    num_features=num_features,
                    num_blocks=num_blocks,
                )
                model.load_state_dict(state_dict, strict=True)
                model.eval()
                model.requires_grad_(False)
                self.model = model.to(self.device)
                self.last_error = None
                _LOG.info("ML model loaded on %s", self.device)
            except Exception as exc:
                self.model = None
                self.last_error = str(exc)
                _LOG.exception("Failed to load ML model: %s", model_path)

    def _infer_locked(self, normalized: np.ndarray) -> np.ndarray:
        assert self.model is not None
        contiguous = np.ascontiguousarray(normalized, dtype=np.float32)
        input_tensor = torch.from_numpy(contiguous).view(1, 1, -1)
        input_tensor = input_tensor.to(self.device, non_blocking=False)
        with torch.inference_mode():
            output = self.model(input_tensor)
        return output.squeeze(0).squeeze(0).detach().cpu().numpy()

    def _fallback_to_cpu_locked(self) -> bool:
        if self.model is None or self.device.type == "cpu":
            return False
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.device = torch.device("cpu")
            self.model = self.model.to(self.device)
            _LOG.warning("ML inference switched to CPU after a CUDA failure")
            return True
        except Exception:
            _LOG.exception("Could not move ML model to CPU")
            return False

    def enhance_spectrum(
        self,
        spectrum: np.ndarray,
        noise_cutoff_ratio: float = 0.0015,
    ) -> np.ndarray:
        """Return an enhanced 1-D spectrum without mutating the input."""
        array = np.asarray(spectrum)
        if array.ndim != 1:
            raise ValueError("spectrum must be a one-dimensional array")
        if array.size == 0:
            return array.copy()
        if self.model is None:
            return array.copy()

        finite = np.isfinite(array)
        if not finite.any():
            return np.zeros_like(array)

        safe = np.where(finite, array, 0.0).astype(np.float32, copy=False)
        original_max = float(np.max(safe))
        if not np.isfinite(original_max) or original_max < 1e-12:
            return np.zeros_like(array)

        normalized = safe / original_max
        with self._lock:
            try:
                output = self._infer_locked(normalized)
            except RuntimeError as exc:
                # CUDA OOM/driver-reset errors should not bring down the GUI.
                message = str(exc).lower()
                if self.device.type == "cuda" and (
                    "cuda" in message or "out of memory" in message
                ) and self._fallback_to_cpu_locked():
                    try:
                        output = self._infer_locked(normalized)
                    except Exception:
                        self.last_error = str(exc)
                        _LOG.exception("ML inference failed again on CPU")
                        return array.copy()
                else:
                    self.last_error = str(exc)
                    _LOG.exception("ML inference failed")
                    return array.copy()
            except Exception as exc:
                self.last_error = str(exc)
                _LOG.exception("ML inference failed")
                return array.copy()

        enhanced = output.astype(np.float32, copy=False) * original_max
        maximum = float(np.max(enhanced)) if enhanced.size else 0.0
        if maximum > 0:
            threshold = maximum * max(0.0, float(noise_cutoff_ratio))
            enhanced = enhanced.copy()
            enhanced[enhanced < threshold] = 0.0

        # Preserve the caller's floating dtype where practical.
        if np.issubdtype(array.dtype, np.floating):
            return enhanced.astype(array.dtype, copy=False)
        return enhanced
