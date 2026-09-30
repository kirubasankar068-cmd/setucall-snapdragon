"""
SetuCall E2EE Security Engine: Fernet (AES-128-CBC + HMAC-SHA256) Encryption for Data Channel.

Provides end-to-end encryption for P2P text messages sent across WebRTC RTCDataChannel.
No third-party key servers or cloud infrastructure required.
"""

import base64
import json
import logging
import time
from typing import Optional, Dict
from cryptography.fernet import Fernet

logger = logging.getLogger("SetuCall.Security")


def generate_key() -> str:
    """Generates a base64-encoded 32-byte symmetric Fernet key."""
    return Fernet.generate_key().decode("utf-8")


class ChatCipher:
    """
    Handles end-to-end encryption and decryption of WebRTC chat messages.
    """

    def __init__(self, key: Optional[str] = None):
        if key:
            self.key = key.encode("utf-8") if isinstance(key, str) else key
        else:
            self.key = Fernet.generate_key()

        self.cipher = Fernet(self.key)
        logger.info("[ChatCipher] Initialized E2EE Fernet cipher engine.")

    def get_key_str(self) -> str:
        """Returns string representation of session key for signaling exchange."""
        return self.key.decode("utf-8") if isinstance(self.key, bytes) else self.key

    def encrypt(self, text: str, sender: str = "Peer") -> str:
        """
        Encrypts plaintext text message and metadata into an E2EE base64 token string.
        """
        payload = {
            "text": text,
            "sender": sender,
            "timestamp": time.time(),
        }
        json_bytes = json.dumps(payload).encode("utf-8")
        token_bytes = self.cipher.encrypt(json_bytes)
        return token_bytes.decode("utf-8")

    def decrypt(self, token_str: str) -> Dict[str, str]:
        """
        Decrypts an E2EE base64 token string into plaintext message dict.
        """
        try:
            token_bytes = token_str.encode("utf-8") if isinstance(token_str, str) else token_str
            decrypted_bytes = self.cipher.decrypt(token_bytes)
            payload = json.loads(decrypted_bytes.decode("utf-8"))
            return payload
        except Exception as e:
            logger.error(f"[ChatCipher] Decryption failed: {e}")
            raise ValueError(f"Failed to decrypt message payload: {e}")
