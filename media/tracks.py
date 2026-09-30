"""
SetuCall Media Tracks: Real Camera/Mic & Synthetic Media Generators.

Provides WebRTC MediaStreamTrack implementations for Video and Audio:
1. SyntheticVideoTrack: Generates dynamic color test video with bouncing ball & timestamps.
2. SyntheticAudioTrack: Generates 48000 Hz PCM sine wave audio.
3. CameraVideoTrack: Wraps physical OpenCV webcam feed with auto-fallback to synthetic media.
All tracks directly pass frames through VideoEnhancer and AudioEnhancer stubs.
"""

import asyncio
import fractions
import math
import time
import logging
import cv2
import numpy as np
import av
from aiortc import MediaStreamTrack

from pipeline.enhancers import VideoEnhancer, AudioEnhancer

logger = logging.getLogger("SetuCall.Media")


class SyntheticVideoTrack(MediaStreamTrack):
    """
    Synthetic WebRTC Video Stream Track (640x480 @ 30 FPS).
    Generates dynamic visuals for local testing without physical camera hardware.
    """
    kind = "video"

    def __init__(self, label: str = "Local", enhancer: VideoEnhancer = None):
        super().__init__()
        self.label = label
        self.enhancer = enhancer or VideoEnhancer()
        self.frame_count = 0
        self.fps = 30
        self.width = 640
        self.height = 480
        self.time_base = fractions.Fraction(1, 90000)
        self._start_time = time.time()
        logger.info(f"[SyntheticVideoTrack] Initialized '{label}' video track (640x480 @ 30fps).")

    async def recv(self) -> av.VideoFrame:
        """Generates next video frame with pacing at 30 FPS."""
        pts = int((time.time() - self._start_time) * 90000)
        await asyncio.sleep(1 / self.fps)

        self.frame_count += 1
        t = time.time() - self._start_time

        # Create dynamic gradient canvas
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Smooth color cycle
        r_val = int((math.sin(t) + 1) * 60) + 20
        g_val = int((math.cos(t * 0.7) + 1) * 60) + 20
        b_val = int((math.sin(t * 1.3) + 1) * 80) + 40
        img[:, :] = (b_val, g_val, r_val)

        # Draw tech grid pattern
        grid_size = 40
        for x in range(0, self.width, grid_size):
            cv2.line(img, (x, 0), (x, self.height), (b_val + 20, g_val + 20, r_val + 20), 1)
        for y in range(0, self.height, grid_size):
            cv2.line(img, (0, y), (self.width, y), (b_val + 20, g_val + 20, r_val + 20), 1)

        # Draw bouncing ball
        ball_x = int((math.sin(t * 2.5) + 1) * 0.5 * (self.width - 60)) + 30
        ball_y = int((abs(math.cos(t * 3.0))) * (self.height - 100)) + 50
        cv2.circle(img, (ball_x, ball_y), 24, (0, 255, 240), -1)
        cv2.circle(img, (ball_x, ball_y), 26, (255, 255, 255), 2)

        # Draw SetuCall Header Box
        cv2.rectangle(img, (20, 20), (self.width - 20, 110), (15, 15, 25), -1)
        cv2.rectangle(img, (20, 20), (self.width - 20, 110), (0, 220, 255), 2)

        # Status text overlays
        cv2.putText(img, f"SetuCall Offline P2P Video Call [{self.label}]", (35, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(img, f"Frame: #{self.frame_count} | Mode: Pure LAN (No Cloud)", (35, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 200), 1)
        cv2.putText(img, f"Enhancer: {self.enhancer.__class__.__name__} (Hexagon NPU Pass-Through)", (35, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

        # Draw live timestamp at bottom
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        cv2.putText(img, f"Time: {timestamp_str}.{int((t % 1) * 100):02d}", (self.width - 250, self.height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Convert numpy image array to PyAV VideoFrame
        new_frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = pts
        new_frame.time_base = self.time_base

        # Process through VideoEnhancer stub before sending
        enhanced_frame = self.enhancer.process(new_frame)
        return enhanced_frame


class SyntheticAudioTrack(MediaStreamTrack):
    """
    Synthetic WebRTC Audio Stream Track (48000 Hz, 16-bit PCM, Mono).
    Generates a gentle 440 Hz sine wave tone for audio P2P verification.
    """
    kind = "audio"

    def __init__(self, label: str = "Local", enhancer: AudioEnhancer = None):
        super().__init__()
        self.label = label
        self.enhancer = enhancer or AudioEnhancer()
        self.sample_rate = 48000
        self.samples_per_frame = 960  # 20ms per frame
        self.frequency = 440.0  # A4 pitch tone
        self.phase = 0.0
        self.pts = 0
        self.time_base = fractions.Fraction(1, self.sample_rate)
        logger.info(f"[SyntheticAudioTrack] Initialized '{label}' audio track (48000 Hz mono).")

    async def recv(self) -> av.AudioFrame:
        """Generates next audio PCM frame with 20ms pacing."""
        await asyncio.sleep(self.samples_per_frame / self.sample_rate)

        t = np.arange(self.samples_per_frame)
        # Soft modulated sine wave
        samples = 0.1 * np.sin(2 * np.pi * self.frequency * (t / self.sample_rate) + self.phase)
        self.phase += 2 * np.pi * self.frequency * (self.samples_per_frame / self.sample_rate)
        self.phase %= 2 * np.pi

        pcm_data = (samples * 32767).astype(np.int16)
        pcm_data = pcm_data.reshape(1, -1)  # 1 channel (mono)

        audio_frame = av.AudioFrame.from_ndarray(pcm_data, format="s16", layout="mono")
        audio_frame.sample_rate = self.sample_rate
        audio_frame.pts = self.pts
        audio_frame.time_base = self.time_base
        self.pts += self.samples_per_frame

        # Process through AudioEnhancer stub before sending
        enhanced_frame = self.enhancer.process(audio_frame)
        return enhanced_frame


class CameraVideoTrack(MediaStreamTrack):
    """
    WebRTC Video Stream Track wrapping physical OpenCV camera capture.
    Falls back gracefully to SyntheticVideoTrack if hardware camera is inaccessible.
    """
    kind = "video"

    def __init__(self, device_index: int = 0, label: str = "Camera", enhancer: VideoEnhancer = None):
        super().__init__()
        self.label = label
        self.enhancer = enhancer or VideoEnhancer()
        self.device_index = device_index
        self.cap = cv2.VideoCapture(device_index)
        self.fps = 30
        self.time_base = fractions.Fraction(1, 90000)
        self._start_time = time.time()
        self.fallback_track = None

        if not self.cap.isOpened():
            logger.warning(f"[CameraVideoTrack] Camera index {device_index} unavailable. Switching to SyntheticVideoTrack.")
            self.fallback_track = SyntheticVideoTrack(label=f"{label} (Fallback)", enhancer=enhancer)

    async def recv(self) -> av.VideoFrame:
        """Reads frame from camera or fallback synthetic track."""
        if self.fallback_track is not None:
            return await self.fallback_track.recv()

        pts = int((time.time() - self._start_time) * 90000)
        await asyncio.sleep(1 / self.fps)

        ret, frame = self.cap.read()
        if not ret or frame is None:
            logger.warning("[CameraVideoTrack] Failed to read frame from webcam. Switching to synthetic fallback.")
            self.fallback_track = SyntheticVideoTrack(label=f"{self.label} (Fallback)", enhancer=self.enhancer)
            return await self.fallback_track.recv()

        # Resize to standard 640x480 if needed
        if frame.shape[1] != 640 or frame.shape[0] != 480:
            frame = cv2.resize(frame, (640, 480))

        # Overlay SetuCall badge on real camera feed
        cv2.putText(frame, f"SetuCall Live Camera [{self.label}]", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        new_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        new_frame.pts = pts
        new_frame.time_base = self.time_base

        return self.enhancer.process(new_frame)

    def stop(self):
        super().stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()


def create_video_track(label: str = "Local", enhancer: VideoEnhancer = None, force_synthetic: bool = False) -> MediaStreamTrack:
    """Factory creating camera track or synthetic test track."""
    if force_synthetic:
        logger.info(f"[MediaFactory] Creating synthetic video track ({label}).")
        return SyntheticVideoTrack(label=label, enhancer=enhancer)

    # Test camera hardware accessibility
    cap = cv2.VideoCapture(0)
    has_cam = cap.isOpened()
    if has_cam:
        ret, _ = cap.read()
        has_cam = ret
    cap.release()

    if has_cam:
        logger.info(f"[MediaFactory] Physical camera detected! Initializing CameraVideoTrack ({label}).")
        return CameraVideoTrack(device_index=0, label=label, enhancer=enhancer)
    else:
        logger.info(f"[MediaFactory] No webcam detected. Defaulting to SyntheticVideoTrack ({label}).")
        return SyntheticVideoTrack(label=label, enhancer=enhancer)


def create_audio_track(label: str = "Local", enhancer: AudioEnhancer = None, force_synthetic: bool = False) -> MediaStreamTrack:
    """Factory creating synthetic audio track."""
    logger.info(f"[MediaFactory] Creating audio track ({label}).")
    return SyntheticAudioTrack(label=label, enhancer=enhancer)
