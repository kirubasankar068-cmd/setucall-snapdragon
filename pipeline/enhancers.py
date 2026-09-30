"""
SetuCall Pipeline: ONNX Runtime ModelBackend Execution Engine.

Implements real-time AI Video Enhancement (Zero-DCE-Lite) and Audio Enhancement (RNNoise-Lite)
with Qualcomm Hexagon NPU QNN Execution Provider support and automatic CPU fallback.
"""

import os
import time
import logging
from abc import ABC, abstractmethod
import numpy as np
import cv2
import av
import onnxruntime as ort

from pipeline.export_models import ensure_models_exist

logger = logging.getLogger("SetuCall.Pipeline")

# Global artifact directory for before/after comparison captures
ARTIFACTS_DIR = os.path.join(
    os.path.expanduser("~"),
    ".gemini",
    "antigravity-ide",
    "brain",
    "4bef7b8e-9c34-4cef-8ba9-be7f8b9a6e1e"
)


def resolve_execution_providers(npu_mode: str = "auto") -> list:
    """
    Resolves ONNX Runtime execution provider hierarchy.
    Attempts Qualcomm QNN Execution Provider (Snapdragon NPU) first,
    falling back to CPUExecutionProvider if QNN is unavailable.
    """
    available = ort.get_available_providers()
    logger.info(f"[ExecutionProvider] Available system providers: {available}")

    mode = npu_mode.lower()
    providers = []

    if mode in ["npu", "auto"]:
        if "QNNExecutionProvider" in available:
            providers.append(("QNNExecutionProvider", {"backend_path": "QnnHtp.dll"}))
            logger.info("[ExecutionProvider] Registered QNNExecutionProvider for Qualcomm Hexagon NPU acceleration.")
        elif mode == "npu":
            logger.warning("[ExecutionProvider] QNNExecutionProvider requested but not available. Falling back to CPU.")

    providers.append("CPUExecutionProvider")
    return providers


class BaseEnhancer(ABC):
    """Abstract base class for SetuCall ONNX AI enhancement models."""

    def __init__(self, enabled: bool = True, npu_mode: str = "auto"):
        self.enabled = enabled
        self.npu_mode = npu_mode
        self.processed_frames_count = 0
        self.total_processing_time_ms = 0.0
        self.active_provider = "Unknown"

    @abstractmethod
    def process(self, frame):
        pass

    def get_stats(self) -> dict:
        avg_time = (
            (self.total_processing_time_ms / self.processed_frames_count)
            if self.processed_frames_count > 0
            else 0.0
        )
        return {
            "processed_frames": self.processed_frames_count,
            "avg_latency_ms": round(avg_time, 2),
            "active_provider": self.active_provider,
            "enabled": self.enabled,
        }


class VideoEnhancer(BaseEnhancer):
    """
    Real-Time Video Enhancement Model using Zero-DCE-Lite ONNX Runtime.
    Performs low-light enhancement and edge sharpening.
    """

    def __init__(self, enabled: bool = True, npu_mode: str = "auto", model_path: str = "models/video_enhancer.onnx"):
        super().__init__(enabled=enabled, npu_mode=npu_mode)
        self.model_path = model_path
        self.session = None
        self.snapshot_saved = False

        if self.enabled:
            ensure_models_exist()
            providers = resolve_execution_providers(npu_mode=npu_mode)
            try:
                self.session = ort.InferenceSession(self.model_path, providers=providers)
                actual_providers = self.session.get_providers()
                self.active_provider = actual_providers[0] if actual_providers else "CPUExecutionProvider"
                logger.info(f"[VideoEnhancer] Loaded Zero-DCE-Lite ONNX model. Active Provider: '{self.active_provider}'")
            except Exception as e:
                logger.error(f"[VideoEnhancer] Failed to load ONNX model ({e}). Defaulting to CPU.")
                self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
                self.active_provider = "CPUExecutionProvider"

    def process(self, frame: av.VideoFrame) -> av.VideoFrame:
        if not self.enabled or frame is None or self.session is None:
            return frame

        start_t = time.perf_counter()

        # Convert PyAV VideoFrame to BGR NumPy array
        img_bgr = frame.to_ndarray(format="bgr24")
        orig_h, orig_w, _ = img_bgr.shape

        # Resize to model input dimension [480, 640] if needed
        if orig_w != 640 or orig_h != 480:
            img_input = cv2.resize(img_bgr, (640, 480))
        else:
            img_input = img_bgr

        # Convert BGR -> RGB and normalize [0, 255] -> [0.0, 1.0] (NCHW format: [1, 3, 480, 640])
        img_rgb = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
        tensor_in = (img_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]

        # Run ONNX Runtime Inference
        outputs = self.session.run(["output_frame"], {"input_frame": tensor_in})
        tensor_out = outputs[0]

        # Convert NCHW -> HWC RGB [0.0, 1.0] -> [0, 255] uint8
        out_rgb = (np.clip(tensor_out[0], 0.0, 1.0).transpose(1, 2, 0) * 255.0).astype(np.uint8)
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

        # Resize back to original dimensions if resized
        if orig_w != 640 or orig_h != 480:
            out_bgr = cv2.resize(out_bgr, (orig_w, orig_h))

        # Save before/after visual snapshot comparison for deliverables
        if not self.snapshot_saved and self.processed_frames_count > 10:
            self._save_snapshots(img_bgr, out_bgr)
            self.snapshot_saved = True

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        self.processed_frames_count += 1
        self.total_processing_time_ms += elapsed_ms

        out_frame = av.VideoFrame.from_ndarray(out_bgr, format="bgr24")
        out_frame.pts = frame.pts
        out_frame.time_base = frame.time_base
        return out_frame

    def _save_snapshots(self, raw_bgr: np.ndarray, enhanced_bgr: np.ndarray):
        """Saves raw vs enhanced frames to workspace and artifact directories."""
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        os.makedirs("artifacts", exist_ok=True)

        raw_path_art = os.path.join(ARTIFACTS_DIR, "raw_frame.png")
        enh_path_art = os.path.join(ARTIFACTS_DIR, "enhanced_frame.png")
        raw_path_ws = os.path.join("artifacts", "raw_frame.png")
        enh_path_ws = os.path.join("artifacts", "enhanced_frame.png")

        cv2.imwrite(raw_path_art, raw_bgr)
        cv2.imwrite(enh_path_art, enhanced_bgr)
        cv2.imwrite(raw_path_ws, raw_bgr)
        cv2.imwrite(enh_path_ws, enhanced_bgr)

        logger.info(f"[VideoEnhancer] Saved visual comparison snapshot: '{raw_path_art}' and '{enh_path_art}'.")


class AudioEnhancer(BaseEnhancer):
    """
    Real-Time Audio Enhancement Model using RNNoise-Lite ONNX Runtime.
    Performs spectral gain masking for background noise suppression.
    """

    def __init__(self, enabled: bool = True, npu_mode: str = "auto", model_path: str = "models/audio_enhancer.onnx"):
        super().__init__(enabled=enabled, npu_mode=npu_mode)
        self.model_path = model_path
        self.session = None

        if self.enabled:
            ensure_models_exist()
            providers = resolve_execution_providers(npu_mode=npu_mode)
            try:
                self.session = ort.InferenceSession(self.model_path, providers=providers)
                actual_providers = self.session.get_providers()
                self.active_provider = actual_providers[0] if actual_providers else "CPUExecutionProvider"
                logger.info(f"[AudioEnhancer] Loaded RNNoise-Lite ONNX model. Active Provider: '{self.active_provider}'")
            except Exception as e:
                logger.error(f"[AudioEnhancer] Failed to load ONNX model ({e}). Defaulting to CPU.")
                self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
                self.active_provider = "CPUExecutionProvider"

    def process(self, frame: av.AudioFrame) -> av.AudioFrame:
        if not self.enabled or frame is None or self.session is None:
            return frame

        start_t = time.perf_counter()

        # Convert PyAV AudioFrame to NumPy int16 array
        pcm_np = frame.to_ndarray()
        channels, num_samples = pcm_np.shape

        # Model expects [1, 960] float32 tensor
        mono_samples = pcm_np[0] if channels > 1 else pcm_np.flatten()
        if len(mono_samples) != 960:
            # Zero-pad or truncate to 960 samples
            padded = np.zeros((960,), dtype=np.float32)
            valid_len = min(len(mono_samples), 960)
            padded[:valid_len] = mono_samples[:valid_len] / 32768.0
            tensor_in = padded[None, :]
        else:
            tensor_in = (mono_samples.astype(np.float32) / 32768.0)[None, :]

        # Run ONNX Inference
        outputs = self.session.run(["output_audio"], {"input_audio": tensor_in})
        tensor_out = outputs[0]

        # Denormalize to int16 PCM
        out_pcm_mono = (np.clip(tensor_out[0], -1.0, 1.0) * 32767.0).astype(np.int16)
        if channels > 1:
            out_pcm = np.tile(out_pcm_mono, (channels, 1))
        else:
            out_pcm = out_pcm_mono[None, :]

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        self.processed_frames_count += 1
        self.total_processing_time_ms += elapsed_ms

        out_frame = av.AudioFrame.from_ndarray(out_pcm, format="s16", layout=frame.layout.name)
        out_frame.sample_rate = frame.sample_rate
        out_frame.pts = frame.pts
        out_frame.time_base = frame.time_base
        return out_frame
