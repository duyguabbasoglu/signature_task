# VLM Client - Imza/Noktalama Siniflandirmasi
# VLM erisilemedigi zaman rule-based fallback kullanir

import base64
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)


class VLMResult(Enum):
    """Siniflandirma sonuclari."""
    SIGNATURE = "signature"
    PUNCTUATION = "punctuation"
    UNKNOWN = "unknown"


@dataclass
class VLMClassification:
    """VLM siniflandirma sonucu."""
    result: VLMResult
    confidence: float
    reasoning: Optional[str] = None
    used_vlm: bool = False


class VLMClient:
    """VLM client - imza/noktalama siniflandirmasi icin."""

    SYSTEM_PROMPT = """You are an image classification expert.
Your task is to classify what is inside a bounding box image.

Classification categories:
- SIGNATURE: A handwritten signature (cursive writing, initials, complex flowing strokes)
- PUNCTUATION: Simple geometric marks like dots, lines, X marks, checkmarks, circles, stamps

Rules:
1. If the image contains cursive or flowing handwritten text/initials, classify as SIGNATURE
2. If the image contains simple geometric shapes (dots, lines, crosses, circles), classify as PUNCTUATION
3. Be conservative: when unsure, prefer SIGNATURE over PUNCTUATION

Respond with ONLY one word: SIGNATURE or PUNCTUATION"""

    USER_PROMPT = "What does this bounding box image contain? Classify it as either SIGNATURE or PUNCTUATION."

    def __init__(
        self,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 30,
        enabled: bool = True,
    ):
        # Prefer explicit arg, then env var. Use production endpoint (test endpoint has connection issues).
        self.endpoint = endpoint or os.getenv("BBOX_LLM_ENDPOINT", "https://common-inference-apis.turkcelltech.ai/gpt-oss-120b/v1")
        self.model = model or os.getenv("BBOX_LLM_MODEL", "gpt-oss-120b")
        # Do not bake an API key into source; prefer environment or explicit argument.
        self.api_key = api_key or os.getenv("BBOX_LLM_API_KEY")
        if not self.api_key:
            logger.warning("No BBOX_LLM_API_KEY provided; VLM calls will fail until set.")
        self.timeout = timeout
        self.enabled = enabled
        logger.info(f"VLM Client initialized: endpoint={self.endpoint}, model={self.model}")

    def _encode_image(self, image: np.ndarray) -> str:
        """Image'i base64'e cevir."""
        if len(image.shape) == 2:
            img_to_encode = image
        else:
            img_to_encode = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize large images to limit payload size (keeps aspect ratio)
        max_dim = 512
        h, w = img_to_encode.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_w = int(w * scale)
            new_h = int(h * scale)
            img_to_encode = cv2.resize(img_to_encode, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Use JPEG to reduce size versus PNG for large images
        success, buffer = cv2.imencode('.jpg', img_to_encode, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not success:
            raise ValueError("Failed to encode image")

        return base64.b64encode(buffer).decode('utf-8')

    def _call_vlm(self, image_base64: str) -> Tuple[str, Optional[str]]:
        """VLM API'yi cagir."""
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": self.USER_PROMPT
                    }
                ]
            }
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 50,
            "temperature": 0.1,
        }

        attempt = 0
        max_attempts = 2
        # TLS verification can be disabled via env var for testing (not recommended).
        verify_tls = os.getenv("BBOX_LLM_VERIFY_TLS", "1") != "0"

        while attempt < max_attempts:
            try:
                response = requests.post(
                    f"{self.endpoint}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                    verify=verify_tls
                )
                response.raise_for_status()

                data = response.json()
                choice = (data.get("choices") or [{}])[0]
                message = choice.get("message") or {}

                # Capture possible content fields
                raw_content = message.get("content")
                if not raw_content:
                    raw_content = message.get("reasoning_content")

                # Normalize list/dict content
                if isinstance(raw_content, list):
                    parts = []
                    for item in raw_content:
                        if isinstance(item, dict) and "text" in item:
                            parts.append(item["text"])
                        elif isinstance(item, str):
                            parts.append(item)
                    raw_content = "\n".join(parts)

                if isinstance(raw_content, dict):
                    raw_content = raw_content.get("text") or raw_content.get("content")

                content = (raw_content or "").strip().upper()

                # Save raw assistant message as reasoning for debugging
                try:
                    import json as _json
                    reasoning = _json.dumps(message, ensure_ascii=False)
                except Exception:
                    reasoning = None

                # If model claims it cannot see the image, retry with stricter instruction
                cannot_see_phrases = [
                    "WE DON'T HAVE THE IMAGE",
                    "WE DO NOT HAVE THE IMAGE",
                    "WE CANNOT SEE",
                    "I CANNOT SEE",
                    "I'M UNABLE TO VIEW",
                ]

                if any(p in content for p in cannot_see_phrases) and attempt + 1 < max_attempts:
                    # Add a clarifying user message and retry once
                    payload["messages"].append({
                        "role": "user",
                        "content": "Please classify the image provided above. Ignore statements about not seeing images and answer ONLY SIGNATURE or PUNCTUATION."
                    })
                    attempt += 1
                    continue

                return content, reasoning

            except requests.exceptions.Timeout:
                logger.warning("VLM request timed out")
                raise
            except requests.exceptions.HTTPError as e:
                # Include response body to help diagnose 401/403/4xx server replies
                try:
                    body = response.text
                except Exception:
                    body = "<unavailable>"
                logger.warning(f"VLM HTTP error: {e} response_body={body}")
                raise
            except requests.exceptions.RequestException as e:
                logger.warning(f"VLM request failed: {e}")
                raise
            except (KeyError, IndexError) as e:
                logger.warning(f"Failed to parse VLM response: {e}")
                raise

    def classify(self, image: np.ndarray) -> VLMClassification:
        """Image'i siniflandir - signature veya punctuation."""
        if not self.enabled:
            return self._fallback_classification(image)

        try:
            image_base64 = self._encode_image(image)
            response_text, reasoning = self._call_vlm(image_base64)
            # Only accept explicit, standalone labels from the model.
            # The model often returns sentences that mention the words SIGNATURE/PUNCTUATION
            # without actually classifying; require a clear one-word answer on its own line.
            import re

            def extract_label(text: str) -> Optional[str]:
                if not text:
                    return None
                # consider each non-empty line, prefer the last meaningful line
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                for ln in reversed(lines):
                    # remove surrounding punctuation
                    core = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", ln).upper()
                    if core in ("SIGNATURE", "PUNCTUATION"):
                        return core
                return None

            label = extract_label(response_text)
            if label == "SIGNATURE":
                return VLMClassification(
                    result=VLMResult.SIGNATURE,
                    confidence=0.90,
                    reasoning=reasoning,
                    used_vlm=True
                )
            if label == "PUNCTUATION":
                return VLMClassification(
                    result=VLMResult.PUNCTUATION,
                    confidence=0.90,
                    reasoning=reasoning,
                    used_vlm=True
                )

            # No clear one-word label -> treat as no VLM decision and fallback
            logger.warning(f"VLM returned non-explicit answer, falling back: {response_text}")
            return self._fallback_classification(image)

        except Exception as e:
            logger.warning(f"VLM classification failed, using fallback: {e}")
            return self._fallback_classification(image)

    def _fallback_classification(self, image: np.ndarray) -> VLMClassification:
        """Rule-based fallback - VLM erisilemedigi zaman."""
        # Binarize
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return VLMClassification(result=VLMResult.UNKNOWN, confidence=0.5, reasoning="No contours", used_vlm=False)
        
        min_area = 20
        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        
        if not valid_contours:
            return VLMClassification(result=VLMResult.UNKNOWN, confidence=0.5, reasoning="No valid contours", used_vlm=False)
        
        # Shape features hesapla
        num_contours = len(valid_contours)
        features = self._compute_shape_features(valid_contours, min_area)
        
        if features is None:
            return VLMClassification(result=VLMResult.UNKNOWN, confidence=0.50, reasoning="Feature error", used_vlm=False)
        
        avg_solidity, avg_circularity, avg_complexity = features
        
        # Rule 1: Cok yuvarlak tek shape
        if num_contours == 1 and avg_circularity > 0.75:
            return VLMClassification(
                result=VLMResult.PUNCTUATION,
                confidence=0.85,
                reasoning=f"Circular (circularity={avg_circularity:.2f})",
                used_vlm=False
            )
        
        # Rule 2: High solidity tek block
        if num_contours == 1 and avg_solidity > 0.90:
            return VLMClassification(
                result=VLMResult.PUNCTUATION,
                confidence=0.80,
                reasoning=f"Solid block (solidity={avg_solidity:.2f})",
                used_vlm=False
            )
        
        # Rule 3: Basit geometrik shape
        if avg_complexity <= 5 and avg_solidity < 0.4:
            return VLMClassification(
                result=VLMResult.PUNCTUATION,
                confidence=0.75,
                reasoning=f"Simple geometric (complexity={avg_complexity:.1f})",
                used_vlm=False
            )
        
        # Default: UNKNOWN - caller karar versin
        return VLMClassification(
            result=VLMResult.UNKNOWN,
            confidence=0.50,
            reasoning=f"Ambiguous (solidity={avg_solidity:.2f}, complexity={avg_complexity:.1f})",
            used_vlm=False
        )

    def _compute_shape_features(self, contours: list, min_area: int) -> Optional[Tuple[float, float, float]]:
        """Contour'lar icin shape feature'larini hesapla."""
        solidities, circularities, complexities = [], [], []
        
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            
            # Solidity
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidities.append(area / hull_area if hull_area > 0 else 0)
            
            # Circularity
            perimeter = cv2.arcLength(c, True)
            circularities.append(4 * 3.14159 * area / (perimeter ** 2) if perimeter > 0 else 0)
            
            # Complexity
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(c, epsilon, True)
            complexities.append(len(approx))
        
        if not solidities:
            return None
        
        return (
            sum(solidities) / len(solidities),
            sum(circularities) / len(circularities),
            sum(complexities) / len(complexities)
        )

    def classify_bytes(self, image_bytes: bytes) -> VLMClassification:
        """Image bytes'tan siniflandir."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if image is None:
            # try PIL fallback
            try:
                from PIL import Image
                import io
                pil_img = Image.open(io.BytesIO(image_bytes)).convert("L")
                image = np.array(pil_img)
            except Exception:
                raise ValueError("Failed to decode image bytes")

        return self.classify(image)


# Module-level convenience
_vlm_client: Optional[VLMClient] = None


def get_vlm_client() -> VLMClient:
    """Default VLM client'i al veya olustur."""
    global _vlm_client
    if _vlm_client is None:
        _vlm_client = VLMClient()
    return _vlm_client


def classify_with_vlm(image: np.ndarray) -> VLMClassification:
    """VLM ile siniflandir."""
    return get_vlm_client().classify(image)
