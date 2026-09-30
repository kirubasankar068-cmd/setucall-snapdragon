"""
SetuCall Media Display: OpenCV Video Rendering & Remote Track Consumption.

Consumes incoming WebRTC video frames from remote peers, calculates frame stats
(FPS, latency, frame counts), and displays the live feed in an OpenCV GUI window.
Supports headless fallback (--no-gui) for automated testing.
"""

import asyncio
import logging
import time
import cv2
import av

logger = logging.getLogger("SetuCall.Display")


class VideoDisplayHandler:
    """
    Consumes remote video track and displays frames using OpenCV GUI.
    """

    def __init__(self, track, window_name: str = "SetuCall Remote Video Feed", no_gui: bool = False):
        self.track = track
        self.window_name = window_name
        self.no_gui = no_gui
        self.running = False
        self.task = None
        self.received_frames = 0
        self.start_time = None
        self.last_frame_time = None

    def start(self):
        """Starts the video frame consumption loop."""
        if not self.running:
            self.running = True
            self.start_time = time.time()
            self.task = asyncio.create_task(self._consume_loop())
            logger.info(f"[VideoDisplayHandler] Started consuming remote track '{self.window_name}'.")

    async def stop(self):
        """Stops the video consumption loop and closes GUI windows."""
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        if not self.no_gui:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass
        logger.info(f"[VideoDisplayHandler] Stopped track '{self.window_name}'. Total frames: {self.received_frames}.")

    async def _consume_loop(self):
        """Async loop reading frames from remote MediaStreamTrack."""
        while self.running:
            try:
                frame: av.VideoFrame = await self.track.recv()
                self.received_frames += 1
                now = time.time()
                
                if self.last_frame_time is not None:
                    fps = 1.0 / max(0.001, now - self.last_frame_time)
                else:
                    fps = 30.0
                self.last_frame_time = now

                # Convert PyAV VideoFrame to BGR OpenCV image array
                img = frame.to_ndarray(format="bgr24")

                # Overlay remote stats banner on the rendered view
                h, w, _ = img.shape
                cv2.putText(img, f"REMOTE FEED | {w}x{h} @ {fps:.1f} FPS | Frames: {self.received_frames}",
                            (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

                if self.received_frames % 30 == 0:
                    elapsed = now - self.start_time
                    avg_fps = self.received_frames / max(0.1, elapsed)
                    logger.info(f"[{self.window_name}] Received frame #{self.received_frames} ({w}x{h}, Avg FPS: {avg_fps:.1f})")

                # GUI Window display if not headless
                if not self.no_gui:
                    try:
                        cv2.imshow(self.window_name, img)
                        key = cv2.waitKey(1) & 0xFF
                        if key == 27 or key == ord('q'):  # Esc or 'q' key
                            logger.info(f"[{self.window_name}] User requested display closure.")
                            break
                    except cv2.error as e:
                        logger.warning(f"[{self.window_name}] OpenCV GUI error ({e}). Switching display to headless mode.")
                        self.no_gui = True

            except (asyncio.CancelledError, Exception) as e:
                # Handle track ending or cancellation cleanly
                if not self.running or "ended" in str(e).lower() or isinstance(e, asyncio.CancelledError):
                    break
                logger.debug(f"[{self.window_name}] Exception during frame receive: {e}")
                await asyncio.sleep(0.05)


class AudioConsumptionHandler:
    """
    Consumes remote audio track, logging chunk statistics for P2P connection verification.
    """

    def __init__(self, track):
        self.track = track
        self.running = False
        self.task = None
        self.received_chunks = 0

    def start(self):
        if not self.running:
            self.running = True
            self.task = asyncio.create_task(self._consume_loop())
            logger.info("[AudioConsumptionHandler] Started consuming remote audio stream.")

    async def stop(self):
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info(f"[AudioConsumptionHandler] Stopped remote audio stream. Total chunks: {self.received_chunks}.")

    async def _consume_loop(self):
        while self.running:
            try:
                frame: av.AudioFrame = await self.track.recv()
                self.received_chunks += 1
                if self.received_chunks % 100 == 0:
                    logger.info(f"[AudioHandler] Received remote audio chunk #{self.received_chunks} ({frame.sample_rate} Hz, layout={frame.layout.name})")
            except (asyncio.CancelledError, Exception) as e:
                if not self.running or "ended" in str(e).lower() or isinstance(e, asyncio.CancelledError):
                    break
                logger.debug(f"[AudioHandler] Exception during audio frame receive: {e}")
                await asyncio.sleep(0.05)
