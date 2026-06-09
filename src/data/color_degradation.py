from __future__ import annotations

import io
import random
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image


Array = np.ndarray
VALID_DEGRADATION_PROFILES = {"balanced", "real_old_photo_heavy"}


@dataclass(frozen=True)
class ColorDegradationConfig:
    yellowing_prob: float = 0.80
    fading_prob: float = 0.70
    sepia_prob: float = 0.45
    blur_prob: float = 0.65
    noise_prob: float = 0.75
    jpeg_prob: float = 0.70
    gamma_prob: float = 0.55
    color_cast_prob: float = 0.65


def ensure_rgb_uint8(image: Image.Image | Array) -> Array:
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"))
    else:
        array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape HxWx3, got {array.shape}")
    return np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8))


class DegradationSimulator:
    """Synthetic old-photo quality degradation without cracks or missing regions."""

    def __init__(
        self,
        config: ColorDegradationConfig | None = None,
        seed: int | None = None,
        profile: str = "balanced",
    ) -> None:
        if profile not in VALID_DEGRADATION_PROFILES:
            raise ValueError(f"Unsupported degradation profile: {profile}")
        self.config = config or ColorDegradationConfig()
        self.seed = seed
        self.profile = profile

    @staticmethod
    def apply_yellowing(image: Array, rng: random.Random) -> tuple[Array, dict[str, float]]:
        strength = rng.uniform(0.08, 0.35)
        result = image.astype(np.float32)
        result[:, :, 0] *= 1.0 + 0.22 * strength
        result[:, :, 1] *= 1.0 + 0.08 * strength
        result[:, :, 2] *= 1.0 - 0.55 * strength
        return np.clip(result, 0, 255).astype(np.uint8), {"strength": strength}

    @staticmethod
    def apply_fading(image: Array, rng: random.Random) -> tuple[Array, dict[str, float]]:
        strength = rng.uniform(0.12, 0.65)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        result = cv2.addWeighted(image, 1.0 - strength, gray_rgb, strength, 0)
        return result, {"strength": strength}

    @staticmethod
    def apply_sepia(image: Array, rng: random.Random) -> tuple[Array, dict[str, float]]:
        strength = rng.uniform(0.10, 0.70)
        matrix = np.array(
            [
                [0.393, 0.769, 0.189],
                [0.349, 0.686, 0.168],
                [0.272, 0.534, 0.131],
            ],
            dtype=np.float32,
        )
        sepia = image.astype(np.float32).reshape(-1, 3) @ matrix.T
        sepia = np.clip(sepia.reshape(image.shape), 0, 255).astype(np.uint8)
        result = cv2.addWeighted(image, 1.0 - strength, sepia, strength, 0)
        return result, {"strength": strength}

    @staticmethod
    def apply_blur(image: Array, rng: random.Random) -> tuple[Array, dict[str, float]]:
        sigma = rng.uniform(0.5, 2.0)
        result = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
        return result, {"sigma": sigma}

    @staticmethod
    def apply_noise(
        image: Array,
        rng: random.Random,
        np_rng: np.random.Generator,
    ) -> tuple[Array, dict[str, Any]]:
        noise_type = rng.choice(["gaussian", "poisson"])
        source = image.astype(np.float32)
        if noise_type == "gaussian":
            sigma = rng.uniform(2.0, 14.0)
            result = source + np_rng.normal(0.0, sigma, source.shape)
            metadata: dict[str, Any] = {"type": noise_type, "sigma": sigma}
        else:
            peak = rng.uniform(20.0, 55.0)
            result = np_rng.poisson(np.clip(source / 255.0, 0.0, 1.0) * peak) / peak * 255.0
            metadata = {"type": noise_type, "peak": peak}
        return np.clip(result, 0, 255).astype(np.uint8), metadata

    @staticmethod
    def apply_jpeg(image: Array, rng: random.Random) -> tuple[Array, dict[str, int]]:
        quality = rng.randint(20, 60)
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        result = np.asarray(Image.open(buffer).convert("RGB"))
        return np.ascontiguousarray(result), {"quality": quality}

    @staticmethod
    def apply_gamma(image: Array, rng: random.Random) -> tuple[Array, dict[str, float]]:
        gamma = rng.uniform(0.7, 1.3)
        normalized = image.astype(np.float32) / 255.0
        result = np.power(np.clip(normalized, 0.0, 1.0), gamma) * 255.0
        return np.clip(result, 0, 255).astype(np.uint8), {"gamma": gamma}

    @staticmethod
    def apply_color_cast(image: Array, rng: random.Random) -> tuple[Array, dict[str, Any]]:
        channel = rng.randrange(3)
        shift = rng.uniform(-22.0, 22.0)
        result = image.astype(np.float32)
        result[:, :, channel] += shift
        return np.clip(result, 0, 255).astype(np.uint8), {"channel": channel, "shift": shift}

    @staticmethod
    def _apply_fading_strength(image: Array, strength: float) -> Array:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return cv2.addWeighted(image, 1.0 - strength, gray_rgb, strength, 0)

    @staticmethod
    def _apply_sepia_strength(image: Array, strength: float) -> Array:
        matrix = np.array(
            [
                [0.393, 0.769, 0.189],
                [0.349, 0.686, 0.168],
                [0.272, 0.534, 0.131],
            ],
            dtype=np.float32,
        )
        sepia = image.astype(np.float32).reshape(-1, 3) @ matrix.T
        sepia = np.clip(sepia.reshape(image.shape), 0, 255).astype(np.uint8)
        return cv2.addWeighted(image, 1.0 - strength, sepia, strength, 0)

    @staticmethod
    def _apply_warm_paper_tone(image: Array, strength: float) -> Array:
        paper_tone = np.full_like(image, [214, 184, 132], dtype=np.uint8)
        return cv2.addWeighted(image, 1.0 - strength, paper_tone, strength, 0)

    @staticmethod
    def _apply_low_contrast(image: Array, factor: float, lift: float) -> Array:
        source = image.astype(np.float32)
        mean = source.mean(axis=(0, 1), keepdims=True)
        result = mean + factor * (source - mean) + lift
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def _apply_gamma_value(image: Array, gamma: float) -> Array:
        normalized = image.astype(np.float32) / 255.0
        return np.clip(np.power(np.clip(normalized, 0.0, 1.0), gamma) * 255.0, 0, 255).astype(np.uint8)

    @staticmethod
    def _apply_gaussian_noise(image: Array, sigma: float, np_rng: np.random.Generator) -> Array:
        noise = np_rng.normal(0.0, sigma, image.shape)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    @staticmethod
    def _apply_jpeg_quality(image: Array, quality: int) -> Array:
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return np.ascontiguousarray(np.asarray(Image.open(buffer).convert("RGB")))

    def _apply_real_old_photo_heavy(
        self,
        image: Array,
        rng: random.Random,
        np_rng: np.random.Generator,
    ) -> tuple[Array, dict[str, Any]]:
        draw = rng.random()
        if draw < 0.35:
            subprofile = "strong_sepia"
        elif draw < 0.65:
            subprofile = "warm_near_grayscale"
        elif draw < 0.90:
            subprofile = "faded_old_color"
        elif draw < 0.95:
            subprofile = "mild_degradation"
        else:
            subprofile = "identity"

        result = image.copy()
        applied: dict[str, Any] = {}
        if subprofile == "strong_sepia":
            fading = rng.uniform(0.72, 0.92)
            sepia = rng.uniform(0.50, 0.82)
            warmth = rng.uniform(0.10, 0.24)
            contrast = rng.uniform(0.68, 0.86)
            result = self._apply_fading_strength(result, fading)
            result = self._apply_sepia_strength(result, sepia)
            result = self._apply_warm_paper_tone(result, warmth)
            result = self._apply_low_contrast(result, contrast, rng.uniform(2.0, 12.0))
            applied.update({"fading": fading, "sepia": sepia, "warm_paper": warmth, "contrast_factor": contrast})
        elif subprofile == "warm_near_grayscale":
            fading = rng.uniform(0.88, 1.0)
            warmth = rng.uniform(0.14, 0.32)
            contrast = rng.uniform(0.70, 0.90)
            result = self._apply_fading_strength(result, fading)
            result = self._apply_warm_paper_tone(result, warmth)
            result = self._apply_low_contrast(result, contrast, rng.uniform(0.0, 10.0))
            applied.update({"fading": fading, "warm_paper": warmth, "contrast_factor": contrast})
        elif subprofile == "faded_old_color":
            fading = rng.uniform(0.55, 0.80)
            warmth = rng.uniform(0.06, 0.18)
            contrast = rng.uniform(0.72, 0.90)
            result = self._apply_fading_strength(result, fading)
            result = self._apply_warm_paper_tone(result, warmth)
            result = self._apply_low_contrast(result, contrast, rng.uniform(0.0, 8.0))
            applied.update({"fading": fading, "warm_paper": warmth, "contrast_factor": contrast})
        elif subprofile == "mild_degradation":
            fading = rng.uniform(0.08, 0.28)
            warmth = rng.uniform(0.02, 0.08)
            result = self._apply_fading_strength(result, fading)
            result = self._apply_warm_paper_tone(result, warmth)
            applied.update({"fading": fading, "warm_paper": warmth})
        else:
            applied["identity"] = True

        if subprofile != "identity":
            gamma = rng.uniform(0.88, 1.16)
            result = self._apply_gamma_value(result, gamma)
            applied["gamma"] = gamma
            if rng.random() < 0.65:
                sigma = rng.uniform(0.3, 0.9)
                result = cv2.GaussianBlur(result, (0, 0), sigmaX=sigma, sigmaY=sigma)
                applied["blur_sigma"] = sigma
            if rng.random() < 0.70:
                noise_sigma = rng.uniform(2.0, 7.0)
                result = self._apply_gaussian_noise(result, noise_sigma, np_rng)
                applied["noise_sigma"] = noise_sigma
            if rng.random() < 0.70:
                jpeg_quality = rng.randint(38, 76)
                result = self._apply_jpeg_quality(result, jpeg_quality)
                applied["jpeg_quality"] = jpeg_quality
        return result, {"subprofile": subprofile, "applied": applied}

    def apply(
        self,
        image: Image.Image | Array,
        *,
        seed: int | None = None,
        return_metadata: bool = False,
    ) -> Array | tuple[Array, dict[str, Any]]:
        actual_seed = self.seed if seed is None else seed
        rng = random.Random(actual_seed)
        np_rng = np.random.default_rng(actual_seed)
        result = ensure_rgb_uint8(image)
        applied: dict[str, Any] = {}

        subprofile = None
        if self.profile == "real_old_photo_heavy":
            result, profile_metadata = self._apply_real_old_photo_heavy(result, rng, np_rng)
            subprofile = profile_metadata["subprofile"]
            applied = profile_metadata["applied"]
        else:
            stages = [
                ("yellowing", self.config.yellowing_prob, lambda value: self.apply_yellowing(value, rng)),
                ("fading", self.config.fading_prob, lambda value: self.apply_fading(value, rng)),
                ("sepia", self.config.sepia_prob, lambda value: self.apply_sepia(value, rng)),
                ("gamma", self.config.gamma_prob, lambda value: self.apply_gamma(value, rng)),
                ("color_cast", self.config.color_cast_prob, lambda value: self.apply_color_cast(value, rng)),
                ("blur", self.config.blur_prob, lambda value: self.apply_blur(value, rng)),
                ("noise", self.config.noise_prob, lambda value: self.apply_noise(value, rng, np_rng)),
                ("jpeg", self.config.jpeg_prob, lambda value: self.apply_jpeg(value, rng)),
            ]
            for name, probability, operation in stages:
                if rng.random() <= probability:
                    result, parameters = operation(result)
                    applied[name] = parameters

        metadata = {
            "seed": actual_seed,
            "profile": self.profile,
            "subprofile": subprofile,
            "config": asdict(self.config),
            "applied": applied,
        }
        return (result, metadata) if return_metadata else result
