"""
SetuCall: Offline P2P Video & Audio Calling Application.
Qualcomm Snapdragon AI Lab Challenge - E2EE P2P Chat & CLI Suite.

Usage:
  Host Mode: python main.py --mode=host
  Join Mode: python main.py --mode=join --ip=192.168.1.15 --msg="Hello Snapdragon!"
  CLI Shell: python setucall.py --mode=host
"""

import argparse
import asyncio
import logging
import sys

from network.signaling import LocalSignalingServer, LocalSignalingClient, get_local_ip
from network.peer import SetuCallPeer
from pipeline.enhancers import VideoEnhancer, AudioEnhancer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SetuCall.Main")


def print_banner(mode: str, local_ip: str, port: int, npu_mode: str, v_provider: str, a_provider: str):
    banner = f"""
===================================================================
                   SETUCALL - OFFLINE P2P CALL                    
       Qualcomm Snapdragon Hexagon NPU & E2EE P2P DataChannel      
===================================================================
 Call Mode       : {mode.upper()}
 Local IP        : {local_ip}
 Signaling Port  : {port}
 STUN/TURN       : DISABLED (Pure LAN P2P)
 E2EE Chat       : Fernet AES-128-CBC + HMAC-SHA256 (DataChannel)
 Video Enhancer  : Zero-DCE-Lite ONNX ({v_provider})
 Audio Enhancer  : RNNoise-Lite ONNX ({a_provider})
===================================================================
"""
    print(banner)


async def run_host(args):
    local_ip = get_local_ip()

    video_enhancer = VideoEnhancer(enabled=True, npu_mode=args.npu_mode)
    audio_enhancer = AudioEnhancer(enabled=True, npu_mode=args.npu_mode)

    print_banner("host", local_ip, args.port, args.npu_mode, video_enhancer.active_provider, audio_enhancer.active_provider)

    peer = SetuCallPeer(
        mode="host",
        force_synthetic_media=args.dummy_media,
        no_gui=args.no_gui,
        npu_mode=args.npu_mode,
        video_enhancer=video_enhancer,
        audio_enhancer=audio_enhancer,
    )

    server = LocalSignalingServer(port=args.port, on_offer_callback=peer.handle_offer)
    await server.start()

    logger.info("Host ready. Waiting for peer connection & E2EE chat over LAN...")
    logger.info(f"Join command: python main.py --mode=join --ip={local_ip} --port={args.port}")

    try:
        # If msg argument provided, send when data channel opens
        if args.msg:
            async def send_host_msg():
                await peer.send_chat_message(args.msg)
            asyncio.create_task(send_host_msg())

        if args.duration > 0:
            logger.info(f"Running call test for fixed duration: {args.duration}s...")
            await asyncio.sleep(args.duration)
        else:
            while True:
                await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutting down Host instance...")
    finally:
        v_stats = video_enhancer.get_stats()
        a_stats = audio_enhancer.get_stats()
        logger.info(f"[Stats] Video Enhancer (Zero-DCE-Lite): {v_stats}")
        logger.info(f"[Stats] Audio Enhancer (RNNoise-Lite): {a_stats}")
        await server.stop()
        await peer.close()


async def run_join(args):
    local_ip = get_local_ip()
    target_ip = args.ip
    port = args.port

    if not target_ip or args.auto_discover:
        discovered_ip, discovered_port = await LocalSignalingClient.discover_host(timeout=3.0)
        if discovered_ip:
            target_ip = discovered_ip
            if discovered_port:
                port = discovered_port
            logger.info(f"[Auto-Discovery] Found SetuCall host at {target_ip}:{port}")
        elif not target_ip:
            logger.error("No host IP provided and mDNS auto-discovery timed out.")
            logger.error("Please specify target host IP: python main.py --mode=join --ip=<HOST_IP>")
            sys.exit(1)

    logger.info(f"Connecting to host at {target_ip}:{port}...")

    video_enhancer = VideoEnhancer(enabled=True, npu_mode=args.npu_mode)
    audio_enhancer = AudioEnhancer(enabled=True, npu_mode=args.npu_mode)

    print_banner("join", local_ip, port, args.npu_mode, video_enhancer.active_provider, audio_enhancer.active_provider)

    peer = SetuCallPeer(
        mode="join",
        force_synthetic_media=args.dummy_media,
        no_gui=args.no_gui,
        npu_mode=args.npu_mode,
        video_enhancer=video_enhancer,
        audio_enhancer=audio_enhancer,
    )

    try:
        sdp_offer = await peer.create_offer()
        sdp_answer = await LocalSignalingClient.send_offer(target_ip, port, sdp_offer)
        await peer.set_remote_answer(sdp_answer)

        logger.info("Call established! Live P2P Video, Audio & E2EE Chat active.")

        # Send encrypted message if requested
        if args.msg:
            logger.info("Waiting for P2P DataChannel connection...")
            await peer.send_chat_message(args.msg)

        if args.duration > 0:
            logger.info(f"Running call test for fixed duration: {args.duration}s...")
            await asyncio.sleep(args.duration)
        else:
            while True:
                await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutting down Join instance...")
    except Exception as e:
        logger.error(f"Failed to establish P2P call: {e}")
    finally:
        v_stats = video_enhancer.get_stats()
        a_stats = audio_enhancer.get_stats()
        logger.info(f"[Stats] Video Enhancer (Zero-DCE-Lite): {v_stats}")
        logger.info(f"[Stats] Audio Enhancer (RNNoise-Lite): {a_stats}")
        await peer.close()


def main():
    parser = argparse.ArgumentParser(
        description="SetuCall: Offline P2P Calling & E2EE Chat App (Qualcomm Hexagon NPU)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["host", "join", "npu", "cpu", "auto"],
        required=True,
        help="Call mode ('host' or 'join').",
    )
    parser.add_argument(
        "--ip",
        type=str,
        default=None,
        help="Target host IPv4 address.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Signaling port (default: 8080).",
    )
    parser.add_argument(
        "--npu-mode",
        type=str,
        choices=["npu", "cpu", "auto"],
        default="auto",
        help="Execution provider choice.",
    )
    parser.add_argument(
        "--msg",
        type=str,
        default=None,
        help="Send an encrypted text chat message over P2P DataChannel once connected.",
    )
    parser.add_argument(
        "--dummy-media",
        action="store_true",
        help="Force synthetic media generation.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable OpenCV GUI rendering window.",
    )
    parser.add_argument(
        "--auto-discover",
        action="store_true",
        help="Use mDNS zeroconf to auto-discover host on LAN.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Fixed duration in seconds to keep call open.",
    )

    args = parser.parse_args()

    if args.mode in ["npu", "cpu", "auto"]:
        args.npu_mode = args.mode
        args.mode = "host"

    if args.mode == "host":
        asyncio.run(run_host(args))
    elif args.mode == "join":
        asyncio.run(run_join(args))


if __name__ == "__main__":
    main()
