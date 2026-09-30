"""
SetuCall Local Signaling: Server, Client, and Zeroconf mDNS Auto-Discovery.

Enables pure offline, serverless WebRTC signaling over local Wi-Fi / LAN without
requiring any internet connection or cloud servers.

Host Mode:
- Starts a local HTTP server on a specified port (default 8080).
- Advertises service on local network via zeroconf mDNS ('_setucall._tcp.local.').
- Receives WebRTC SDP Offers via HTTP POST and returns SDP Answers.

Join Mode:
- Auto-discovers host IP via mDNS or accepts target IP manually via CLI.
- Sends WebRTC SDP Offer via HTTP POST and receives SDP Answer.
"""

import asyncio
import socket
import logging
from typing import Optional, Tuple, Callable
from aiohttp import web, ClientSession
from zeroconf import Zeroconf, ServiceInfo, ServiceBrowser, ServiceListener

logger = logging.getLogger("SetuCall.Signaling")


def get_local_ip() -> str:
    """
    Returns the primary IPv4 address of the local network interface.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Dummy connection to determine active default network route interface
        s.connect(("10.255.255.255", 1))
        local_ip = s.getsockname()[0]
    except Exception:
        local_ip = "127.0.0.1"
    finally:
        s.close()
    return local_ip


class SetuCallMDNSListener(ServiceListener):
    """mDNS Service Listener for auto-discovering SetuCall hosts on LAN."""

    def __init__(self):
        self.found_ip: Optional[str] = None
        self.found_port: Optional[int] = None
        self.discovered_event = asyncio.Event()

    def remove_service(self, zeroconf, type, name):
        pass

    def add_service(self, zeroconf, type, name):
        info = zeroconf.get_service_info(type, name)
        if info and info.addresses:
            self.found_ip = socket.inet_ntoa(info.addresses[0])
            self.found_port = info.port
            logger.info(f"[mDNS] Discovered SetuCall Host '{name}' at {self.found_ip}:{self.found_port}")
            self.discovered_event.set()

    def update_service(self, zeroconf, type, name):
        pass


class LocalSignalingServer:
    """
    Offline LAN HTTP Signaling Server & mDNS Advertiser for Host Mode.
    """

    def __init__(self, port: int = 8080, on_offer_callback: Optional[Callable] = None):
        self.port = port
        self.on_offer_callback = on_offer_callback
        self.app = web.Application()
        self.app.router.add_post("/offer", self._handle_offer)
        self.app.router.add_get("/status", self._handle_status)
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self.zeroconf: Optional[Zeroconf] = None
        self.service_info: Optional[ServiceInfo] = None
        self.local_ip = get_local_ip()

    async def start(self):
        """Starts the local HTTP server and registers mDNS service."""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "0.0.0.0", self.port)
        await self.site.start()
        logger.info(f"[SignalingServer] HTTP listener running at http://{self.local_ip}:{self.port}")

        # Register mDNS service asynchronously in background task
        asyncio.create_task(self._register_mdns())

    async def _register_mdns(self):
        """Asynchronous mDNS service registration."""
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._do_register_mdns)
        except Exception as e:
            logger.warning(f"[mDNS] Failed to register mDNS service: {e}. Manual IP connection available.")

    def _do_register_mdns(self):
        try:
            self.zeroconf = Zeroconf()
            hostname = socket.gethostname()
            desc = {"service": "SetuCall P2P Video Call", "host": hostname}
            self.service_info = ServiceInfo(
                "_setucall._tcp.local.",
                f"SetuCall-Host-{hostname}._setucall._tcp.local.",
                addresses=[socket.inet_aton(self.local_ip)],
                port=self.port,
                properties=desc,
                server=f"{hostname}.local.",
            )
            self.zeroconf.register_service(self.service_info)
            logger.info(f"[mDNS] Registered service '_setucall._tcp.local.' as SetuCall-Host-{hostname}")
        except Exception as e:
            logger.warning(f"[mDNS] mDNS registration error: {e}")

    async def stop(self):
        """Stops the HTTP server and unregisters mDNS service."""
        if self.zeroconf and self.service_info:
            try:
                self.zeroconf.unregister_service(self.service_info)
                self.zeroconf.close()
            except Exception as e:
                logger.warning(f"[mDNS] Error during unregister: {e}")
        if self.runner:
            await self.runner.cleanup()
        logger.info("[SignalingServer] Stopped LAN signaling server.")

    async def _handle_offer(self, request: web.Request) -> web.Response:
        """Handles POST /offer from joining peer."""
        try:
            offer_data = await request.json()
            logger.info("[SignalingServer] Received SDP Offer from client peer.")
            
            if self.on_offer_callback:
                answer_data = await self.on_offer_callback(offer_data)
                return web.json_response(answer_data)
            else:
                return web.json_response({"error": "No offer handler configured"}, status=500)
        except Exception as e:
            logger.error(f"[SignalingServer] Error handling SDP Offer: {e}")
            return web.json_response({"error": str(e)}, status=400)

    async def _handle_status(self, request: web.Request) -> web.Response:
        """Health status endpoint."""
        return web.json_response({
            "status": "ready",
            "app": "SetuCall",
            "local_ip": self.local_ip,
            "port": self.port
        })


class LocalSignalingClient:
    """
    Offline LAN Signaling Client for Join Mode.
    """

    @staticmethod
    async def discover_host(timeout: float = 3.0) -> Tuple[Optional[str], Optional[int]]:
        """
        Uses mDNS zeroconf to discover SetuCall hosts on local LAN.
        """
        logger.info(f"[mDNS] Browsing LAN for SetuCall hosts (timeout {timeout}s)...")
        zeroconf = Zeroconf()
        listener = SetuCallMDNSListener()
        browser = ServiceBrowser(zeroconf, "_setucall._tcp.local.", listener)

        try:
            await asyncio.sleep(timeout)
            if listener.found_ip:
                return listener.found_ip, listener.found_port
        finally:
            zeroconf.close()

        logger.info("[mDNS] No auto-discovered host found within timeout.")
        return None, None

    @staticmethod
    async def send_offer(target_ip: str, port: int, sdp_offer: dict) -> dict:
        """
        Sends SDP offer to host peer via HTTP POST and returns the SDP answer.
        """
        url = f"http://{target_ip}:{port}/offer"
        logger.info(f"[SignalingClient] Sending SDP Offer to host at {url}...")

        async with ClientSession() as session:
            async with session.post(url, json=sdp_offer, timeout=10.0) as resp:
                if resp.status == 200:
                    answer_data = await resp.json()
                    logger.info("[SignalingClient] Received SDP Answer from host.")
                    return answer_data
                else:
                    text = await resp.text()
                    raise RuntimeError(f"Signaling failed (HTTP {resp.status}): {text}")
