"""
SetuCall WebRTC Peer Connection Core with E2EE P2P Data Channel Chat.

Configures pure LAN P2P connections (iceServers=[]), attaches local media streams,
and manages encrypted text chat over WebRTC RTCDataChannel.
"""

import asyncio
import logging
from typing import Optional, Callable
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration

from pipeline.enhancers import VideoEnhancer, AudioEnhancer
from media.tracks import create_video_track, create_audio_track
from media.display import VideoDisplayHandler, AudioConsumptionHandler
from security.encryption import ChatCipher, generate_key

logger = logging.getLogger("SetuCall.Peer")


class SetuCallPeer:
    """
    WebRTC P2P Peer Connection & E2EE Data Channel Manager.
    """

    def __init__(
        self,
        mode: str = "host",
        force_synthetic_media: bool = False,
        no_gui: bool = False,
        npu_mode: str = "auto",
        chat_key: Optional[str] = None,
        video_enhancer: Optional[VideoEnhancer] = None,
        audio_enhancer: Optional[AudioEnhancer] = None,
        on_chat_callback: Optional[Callable] = None,
    ):
        self.mode = mode
        self.force_synthetic_media = force_synthetic_media
        self.no_gui = no_gui
        self.npu_mode = npu_mode
        self.video_enhancer = video_enhancer or VideoEnhancer(npu_mode=npu_mode)
        self.audio_enhancer = audio_enhancer or AudioEnhancer(npu_mode=npu_mode)
        self.on_chat_callback = on_chat_callback

        # Initialize E2EE Chat Cipher (share session key or generate new)
        self.chat_key = chat_key or generate_key()
        self.cipher = ChatCipher(key=self.chat_key)

        # WebRTC Configuration for pure LAN (No STUN/TURN servers)
        config = RTCConfiguration(iceServers=[])
        self.pc = RTCPeerConnection(configuration=config)

        self.local_video_track = None
        self.local_audio_track = None
        self.remote_video_display: Optional[VideoDisplayHandler] = None
        self.remote_audio_handler: Optional[AudioConsumptionHandler] = None
        self.data_channel = None
        self.connected_event = asyncio.Event()
        self.data_channel_open_event = asyncio.Event()

        self._setup_event_listeners()

        # Host creates DataChannel; Join receives it on "datachannel" event
        if self.mode == "host":
            self.data_channel = self.pc.createDataChannel("setucall-chat")
            self._setup_data_channel(self.data_channel)

    def _setup_event_listeners(self):
        """Sets up WebRTC connection state, track, and data channel listeners."""

        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            state = self.pc.connectionState
            logger.info(f"[WebRTC Peer] Connection state changed: -> '{state.upper()}'")
            if state == "connected":
                self.connected_event.set()
            elif state in ["failed", "closed", "disconnected"]:
                self.connected_event.clear()

        @self.pc.on("track")
        def on_track(track):
            logger.info(f"[WebRTC Peer] Received remote '{track.kind}' track.")
            if track.kind == "video":
                window_name = f"SetuCall Remote Feed ({'Client' if self.mode == 'host' else 'Host'})"
                self.remote_video_display = VideoDisplayHandler(track, window_name=window_name, no_gui=self.no_gui)
                self.remote_video_display.start()
            elif track.kind == "audio":
                self.remote_audio_handler = AudioConsumptionHandler(track)
                self.remote_audio_handler.start()

        @self.pc.on("datachannel")
        def on_datachannel(channel):
            logger.info(f"[WebRTC Peer] Received remote RTCDataChannel '{channel.label}'.")
            self.data_channel = channel
            self._setup_data_channel(channel)

    def _setup_data_channel(self, channel):
        """Configures RTCDataChannel event handlers."""
        if channel.readyState == "open":
            logger.info(f"[DataChannel] P2P DataChannel '{channel.label}' ALREADY OPEN and ready for E2EE chat!")
            self.data_channel_open_event.set()

        @channel.on("open")
        def on_open():
            logger.info(f"[DataChannel] P2P DataChannel '{channel.label}' OPEN and ready for E2EE chat!")
            self.data_channel_open_event.set()

        @channel.on("message")
        def on_message(message):
            self._handle_incoming_chat(message)

        @channel.on("close")
        def on_close():
            logger.info(f"[DataChannel] DataChannel '{channel.label}' CLOSED.")
            self.data_channel_open_event.clear()

    def _handle_incoming_chat(self, raw_message: str):
        """Decrypts incoming E2EE chat message payload."""
        try:
            payload = self.cipher.decrypt(raw_message)
            sender = payload.get("sender", "Peer")
            text = payload.get("text", "")
            logger.info(f"\n[CHAT E2EE] {sender}: {text}\nSetuCall> ", extra={"raw": True})
            
            if self.on_chat_callback:
                self.on_chat_callback(payload)
        except Exception as e:
            logger.error(f"[DataChannel] Failed to process incoming encrypted chat message: {e}")

    async def send_chat_message(self, text: str) -> bool:
        """
        Encrypts text and sends over active P2P RTCDataChannel.
        Waits for WebRTC connection and DataChannel to open if connecting.
        """
        if self.pc.connectionState != "connected":
            logger.info("[DataChannel] Waiting for P2P WebRTC connection...")
            try:
                await asyncio.wait_for(self.connected_event.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                pass

        if not self.data_channel_open_event.is_set():
            logger.info("[DataChannel] Waiting for P2P DataChannel open state...")
            try:
                await asyncio.wait_for(self.data_channel_open_event.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass

        if not self.data_channel or self.data_channel.readyState != "open":
            logger.warning("[DataChannel] Cannot send chat message: DataChannel is not open.")
            return False

        try:
            sender_name = "Host" if self.mode == "host" else "Join"
            encrypted_token = self.cipher.encrypt(text, sender=sender_name)
            self.data_channel.send(encrypted_token)
            logger.info(f"[CHAT SENT] You ({sender_name}): {text}")
            return True
        except Exception as e:
            logger.error(f"[DataChannel] Failed to send chat message: {e}")
            return False

    def add_local_tracks(self):
        """Creates and attaches local video and audio tracks to the peer connection."""
        label = "Host" if self.mode == "host" else "Join"
        self.local_video_track = create_video_track(
            label=label, enhancer=self.video_enhancer, force_synthetic=self.force_synthetic_media
        )
        self.local_audio_track = create_audio_track(
            label=label, enhancer=self.audio_enhancer, force_synthetic=self.force_synthetic_media
        )

        self.pc.addTrack(self.local_video_track)
        self.pc.addTrack(self.local_audio_track)
        logger.info(f"[WebRTC Peer] Local video and audio tracks added to RTCPeerConnection.")

    async def create_offer(self) -> dict:
        """Generates SDP offer and includes chat session encryption key."""
        self.add_local_tracks()
        if not self.data_channel:
            self.data_channel = self.pc.createDataChannel("setucall-chat")
            self._setup_data_channel(self.data_channel)

        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)
        logger.info("[WebRTC Peer] Generated local SDP Offer with E2EE DataChannel & key.")
        return {
            "sdp": self.pc.localDescription.sdp,
            "type": self.pc.localDescription.type,
            "chat_key": self.cipher.get_key_str(),
        }

    async def handle_offer(self, offer_dict: dict) -> dict:
        """Processes incoming SDP offer, aligns session key, and generates SDP answer."""
        self.add_local_tracks()
        
        # Align encryption cipher key with offer peer
        if "chat_key" in offer_dict and offer_dict["chat_key"]:
            self.chat_key = offer_dict["chat_key"]
            self.cipher = ChatCipher(key=self.chat_key)
            logger.info("[WebRTC Peer] E2EE session cipher key aligned with offer peer.")

        offer = RTCSessionDescription(sdp=offer_dict["sdp"], type=offer_dict["type"])
        await self.pc.setRemoteDescription(offer)

        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        return {
            "sdp": self.pc.localDescription.sdp,
            "type": self.pc.localDescription.type,
            "chat_key": self.cipher.get_key_str(),
        }

    async def set_remote_answer(self, answer_dict: dict):
        """Sets SDP answer received from host peer and aligns cipher key if provided."""
        if "chat_key" in answer_dict and answer_dict["chat_key"]:
            self.chat_key = answer_dict["chat_key"]
            self.cipher = ChatCipher(key=self.chat_key)

        answer = RTCSessionDescription(sdp=answer_dict["sdp"], type=answer_dict["type"])
        await self.pc.setRemoteDescription(answer)
        logger.info("[WebRTC Peer] Set remote SDP Answer description.")

    async def close(self):
        """Cleanly closes peer connection and data channel."""
        if self.data_channel:
            self.data_channel.close()
        if self.remote_video_display:
            await self.remote_video_display.stop()
        if self.remote_audio_handler:
            await self.remote_audio_handler.stop()
        if self.local_video_track:
            self.local_video_track.stop()
        if self.local_audio_track:
            self.local_audio_track.stop()
        await self.pc.close()
        logger.info("[WebRTC Peer] Peer connection and DataChannel closed.")
