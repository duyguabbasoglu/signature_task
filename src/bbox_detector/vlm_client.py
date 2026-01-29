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
        self.endpoint = endpoint or os.getenv(
            "BBOX_LLM_ENDPOINT",
            "https://common-inference-apis.turkcelltech.ai/gpt-oss-120b/v1"
        )
        self.model = model or os.getenv("BBOX_LLM_MODEL", "gpt-oss-120b")
        self.api_key = api_key or os.getenv(
            "BBOX_LLM_API_KEY",
            "uavPCHhER6/EZnQFc2JafwjyqkcPE0oL6sowlCWsLGw="
        )
        self.timeout = timeout
        self.enabled = enabled
        logger.info(f"VLM Client initialized: endpoint={self.endpoint}, model={self.model}")

    def _encode_image(self, image: np.ndarray) -> str:
        """Image'i base64'e cevir."""
        if len(image.shape) == 2:
            img_to_encode = image
        else:
            img_to_encode = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        success, buffer = cv2.imencode('.png', img_to_encode)
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
                    {"type": "text", "text": self.USER_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"}
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

        try:
            response = requests.post(
                f"{self.endpoint}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout,
                verify=False
            )
            response.raise_for_status()
            
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip().upper()
            return content, None

        except requests.exceptions.Timeout:
            logger.warning("VLM request timed out")
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
            
            if "SIGNATURE" in response_text:
                return VLMClassification(
                    result=VLMResult.SIGNATURE,
                    confidence=0.90,
                    reasoning=reasoning,
                    used_vlm=True
                )
            elif "PUNCTUATION" in response_text:
                return VLMClassification(
                    result=VLMResult.PUNCTUATION,
                    confidence=0.90,
                    reasoning=reasoning,
                    used_vlm=True
                )
            else:
                logger.warning(f"Unexpected VLM response: {response_text}")
                return VLMClassification(
                    result=VLMResult.UNKNOWN,
                    confidence=0.50,
                    reasoning=f"Unclear response: {response_text}",
                    used_vlm=True
                )

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
