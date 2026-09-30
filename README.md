# SetuCall - Offline P2P Video & Audio Calling App

> **Snapdragon® AI Lab Build & Present Challenge**  
> Foundational Scaffold, WebRTC P2P LAN Networking & Real-Time AI Enhancer Core (Qualcomm Hexagon NPU / ONNX Runtime).

SetuCall is an offline, peer-to-peer (P2P) high-definition video and audio calling application engineered specifically to run entirely over local Wi-Fi / LAN networks with **zero internet connection** and **no cloud servers**, accelerated by real-time ONNX Runtime AI models.

---

## Key Features

1. **Zero Internet P2P WebRTC Core**: Configured with `iceServers=[]` for strict LAN-only peer connectivity over local Wi-Fi/Ethernet.
2. **Offline Local Signaling**: Uses direct local HTTP signaling with **mDNS (`zeroconf`)** auto-discovery on LAN (fallback to manual local IP entry).
3. **Snapdragon Hexagon NPU AI Enhancers**:
   * **`VideoEnhancer`**: `Zero-DCE-Lite` ONNX model (~40 KB) performing real-time low-light illumination boosting, shadow enhancement, and edge sharpening.
   * **`AudioEnhancer`**: `RNNoise-Lite` ONNX model (~2.2 KB) performing real-time spectral mask background noise suppression on 20ms audio frames.
4. **Execution Provider Auto-Fallback**:
   * Auto-selects **`QNNExecutionProvider`** (Qualcomm Hexagon NPU) when available on Snapdragon PCs, automatically falling back to **`CPUExecutionProvider`**.
   * Controlled via CLI `--npu-mode=npu|cpu|auto`.
5. **Smart Hardware Auto-Fallback**: Auto-detects physical webcams and microphones, falling back to dynamic synthetic test generators for seamless testing in terminal sessions or headless environments.

---

## Directory Structure

```text
d:/Snapdragon Project/
├── main.py                # CLI entry point (host / join modes)
├── requirements.txt       # Dependencies
├── README.md              # Documentation & guide
├── models/
│   ├── video_enhancer.onnx # Zero-DCE-Lite ONNX Model (IR v8, opset 13)
│   └── audio_enhancer.onnx # RNNoise-Lite ONNX Model (IR v8, opset 13)
├── pipeline/
│   ├── __init__.py
│   ├── enhancers.py       # ONNX Runtime VideoEnhancer & AudioEnhancer
│   └── export_models.py   # ONNX model builder script
├── media/
│   ├── __init__.py
│   ├── tracks.py          # Synthetic & OpenCV camera WebRTC tracks
│   └── display.py         # OpenCV video display & remote audio consumer
└── network/
    ├── __init__.py
    ├── signaling.py       # Offline HTTP LAN server & mDNS zeroconf discovery
    └── peer.py            # RTCPeerConnection manager
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## How to Run

### Quick Verification (Two Terminals on Localhost)

**Terminal 1 (Host Instance):**
```bash
python main.py --mode=host --npu-mode=auto --dummy-media
```

**Terminal 2 (Join Instance):**
```bash
python main.py --mode=join --ip=127.0.0.1 --npu-mode=auto --dummy-media
```

---

### Call Between Two Snapdragon Machines on Local LAN

**Machine 1 (Host):**
```bash
python main.py --mode=host --npu-mode=npu
```

**Machine 2 (Join):**
```bash
python main.py --mode=join --auto-discover --npu-mode=npu
```

---

## CLI Options Reference

| Argument | Type | Default | Description |
|---|---|---|---|
| `--mode` | `host` \| `join` \| `npu` \| `cpu` \| `auto` | *Required* | Call operating mode. |
| `--ip` | string | `None` | Target host IPv4 address. |
| `--port` | int | `8080` | Local HTTP signaling port. |
| `--npu-mode` | `npu` \| `cpu` \| `auto` | `auto` | Execution provider preference (`QNNExecutionProvider` with CPU fallback). |
| `--dummy-media` | flag | `False` | Forces synthetic video/audio media generation. |
| `--no-gui` | flag | `False` | Disables OpenCV popup rendering window (headless mode). |
| `--auto-discover` | flag | `False` | Uses mDNS `zeroconf` to automatically locate host on LAN. |
| `--duration` | int | `0` | Run call for N seconds then exit (0 = indefinite until Ctrl+C). |
