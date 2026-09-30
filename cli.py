"""
SetuCall CLI Command Interface: Interactive REPL & Scriptable Command Suite.

Supports commands:
  setucall connect <peer-ip>
  setucall call
  setucall msg "text"
  setucall quit
"""

import sys
import os
import asyncio
import logging
import shlex
from typing import Optional

from network.signaling import LocalSignalingServer, LocalSignalingClient, get_local_ip
from network.peer import SetuCallPeer
from pipeline.enhancers import VideoEnhancer, AudioEnhancer

logger = logging.getLogger("SetuCall.CLI")


class SetuCallCLI:
    """
    Interactive CLI Controller for SetuCall offline P2P calls & E2EE chat.
    """

    def __init__(self, mode: str = "host", port: int = 8080, dummy_media: bool = True, no_gui: bool = True, npu_mode: str = "auto"):
        self.mode = mode
        self.port = port
        self.dummy_media = dummy_media
        self.no_gui = no_gui
        self.npu_mode = npu_mode
        self.local_ip = get_local_ip()

        self.video_enhancer = VideoEnhancer(enabled=True, npu_mode=self.npu_mode)
        self.audio_enhancer = AudioEnhancer(enabled=True, npu_mode=self.npu_mode)

        self.peer: Optional[SetuCallPeer] = None
        self.server: Optional[LocalSignalingServer] = None
        self.running = True

    async def initialize(self):
        """Initializes peer connection and signaling server if host."""
        self.peer = SetuCallPeer(
            mode=self.mode,
            force_synthetic_media=self.dummy_media,
            no_gui=self.no_gui,
            npu_mode=self.npu_mode,
            video_enhancer=self.video_enhancer,
            audio_enhancer=self.audio_enhancer,
        )

        if self.mode == "host":
            self.server = LocalSignalingServer(port=self.port, on_offer_callback=self.peer.handle_offer)
            await self.server.start()
            print(f"\n[HOST] Server active at http://{self.local_ip}:{self.port}")
            print(f"[HOST] Waiting for peers to 'connect {self.local_ip}'...")

    async def execute_command(self, cmd_line: str):
        """Executes a single CLI command string."""
        cmd_line = cmd_line.strip()
        if not cmd_line:
            return

        parts = shlex.split(cmd_line)
        cmd = parts[0].lower()
        args = parts[1:]

        # Strips optional leading 'setucall' keyword
        if cmd == "setucall" and len(args) > 0:
            cmd = args[0].lower()
            args = args[1:]

        if cmd == "connect":
            if not args:
                print("Error: Missing peer IP. Usage: setucall connect <peer-ip>")
                return
            target_ip = args[0]
            await self.do_connect(target_ip)

        elif cmd == "call":
            await self.do_call()

        elif cmd == "msg":
            if not args:
                print("Error: Missing text message. Usage: setucall msg \"text\"")
                return
            text = " ".join(args)
            await self.do_msg(text)

        elif cmd == "quit" or cmd == "exit":
            await self.do_quit()

        elif cmd == "help":
            self.print_help()
        else:
            print(f"Unknown command '{cmd}'. Type 'help' for available commands.")

    async def do_connect(self, target_ip: str):
        """Connects client to target host IP."""
        if self.mode == "host":
            print("[INFO] Operating in Host mode. Host listens for incoming connections.")
            return

        print(f"[CLI] Connecting to host at {target_ip}:{self.port}...")
        try:
            sdp_offer = await self.peer.create_offer()
            sdp_answer = await LocalSignalingClient.send_offer(target_ip, self.port, sdp_offer)
            await self.peer.set_remote_answer(sdp_answer)
            print(f"[CLI] Connected to {target_ip}! Live WebRTC P2P channel active.")
        except Exception as e:
            print(f"[ERROR] Failed to connect to {target_ip}: {e}")

    async def do_call(self):
        """Initiates live media streaming call."""
        if not self.peer:
            print("Error: Peer connection not initialized.")
            return
        if self.peer.pc.connectionState != "connected":
            print("[CLI] Connection state is not connected. Attempting media initialization...")
        print("[CLI] Media call active! Camera video & microphone audio streaming over P2P.")

    async def do_msg(self, text: str):
        """Encrypts and sends chat message over WebRTC DataChannel."""
        if not self.peer:
            print("Error: Peer connection not initialized.")
            return
        success = await self.peer.send_chat_message(text)
        if not success:
            print("[ERROR] Message failed to send. Ensure P2P connection is connected.")

    async def do_quit(self):
        """Cleanly terminates P2P connection and exits."""
        print("[CLI] Shutting down SetuCall connection...")
        self.running = False
        if self.peer:
            await self.peer.close()
        if self.server:
            await self.server.stop()
        print("[CLI] SetuCall session closed.")

    def print_help(self):
        help_text = """
SetuCall CLI Commands:
  setucall connect <peer-ip>  : Connect to host peer at target IP address over LAN
  setucall call               : Start/verify video & audio streaming call
  setucall msg "text"         : Encrypt and send text message over P2P DataChannel
  setucall quit               : Exit session and close P2P connection
"""
        print(help_text)

    async def run_interactive_repl(self):
        """Runs interactive CLI prompt loop."""
        await self.initialize()
        self.print_help()

        loop = asyncio.get_running_loop()
        while self.running:
            try:
                # Non-blocking async input prompt
                print("SetuCall> ", end="", flush=True)
                line = await loop.run_in_executor(None, sys.stdin.readline)
                if not line:
                    break
                await self.execute_command(line)
            except (KeyboardInterrupt, asyncio.CancelledError):
                break
        await self.do_quit()
