"""Compatibility helpers for Krea 2 image-edit conditioning profiles.

The public ai-toolkit edit path uses clean reference tokens after the target,
with separate t=0 modulation.  The conradlocke Identity Edit adapters use the
older/private training contract exposed by their official ComfyUI nodes.  Keep
that behavior opt-in so existing edit LoRAs retain their original semantics.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F


AI_TOOLKIT_T0_PROFILE = "ai_toolkit_t0"
IDENTITY_EDIT_V12_PROFILE = "identity_edit_v12"

_PROFILE_ALIASES = {
    AI_TOOLKIT_T0_PROFILE: AI_TOOLKIT_T0_PROFILE,
    "default": AI_TOOLKIT_T0_PROFILE,
    IDENTITY_EDIT_V12_PROFILE: IDENTITY_EDIT_V12_PROFILE,
    "identity_edit_v1_2": IDENTITY_EDIT_V12_PROFILE,
    "conrad_identity_edit_v12": IDENTITY_EDIT_V12_PROFILE,
}


def resolve_edit_profile(model_kwargs: Mapping, is_edit: bool) -> str:
    """Return and validate the requested Krea 2 edit-conditioning profile."""

    requested = (
        str(model_kwargs.get("edit_profile", AI_TOOLKIT_T0_PROFILE)).strip().lower()
    )
    try:
        profile = _PROFILE_ALIASES[requested]
    except KeyError as exc:
        choices = ", ".join((AI_TOOLKIT_T0_PROFILE, IDENTITY_EDIT_V12_PROFILE))
        raise ValueError(
            f"Unknown Krea2 edit_profile {requested!r}; expected one of: {choices}"
        ) from exc

    if profile == IDENTITY_EDIT_V12_PROFILE:
        if not is_edit:
            raise ValueError(
                "Krea2 edit_profile='identity_edit_v12' requires model_kwargs.edit=true"
            )
        if bool(model_kwargs.get("kv_cache", False)):
            raise ValueError(
                "Krea2 edit_profile='identity_edit_v12' is incompatible with "
                "model_kwargs.kv_cache=true: its reference tokens use the current "
                "timestep and full joint attention"
            )
    return profile


def fit_reference_geometry(
    source_height: int,
    source_width: int,
    target_height: int,
    target_width: int,
) -> tuple[int, int, int, int, int, int]:
    """Compute Identity Edit v1.2's pixel-space FIT crop and output size.

    Returns ``(crop_y, crop_x, crop_height, crop_width, output_height,
    output_width)``.  Near-matched aspect ratios take the minimal center crop and
    fill the target.  Genuine mismatches fit inside, floor each dimension to a
    16-pixel transformer-grid boundary, and retain their aspect ratio.
    """

    if min(source_height, source_width, target_height, target_width) <= 0:
        raise ValueError("source and target dimensions must be positive")

    fit_scale = min(target_height / source_height, target_width / source_width)
    fitted_height = source_height * fit_scale
    fitted_width = source_width * fit_scale

    # The official v1.2 node uses an 8% tolerance to avoid one- or two-token
    # border gaps for sources whose aspect ratio is already close to the target.
    if fitted_height >= target_height * 0.92 and fitted_width >= target_width * 0.92:
        fill_scale = max(target_height / source_height, target_width / source_width)
        crop_height = min(source_height, int(round(target_height / fill_scale)))
        crop_width = min(source_width, int(round(target_width / fill_scale)))
        crop_y = (source_height - crop_height) // 2
        crop_x = (source_width - crop_width) // 2
        return (
            crop_y,
            crop_x,
            crop_height,
            crop_width,
            target_height,
            target_width,
        )

    snapped_target_height = max(16, target_height // 16 * 16)
    snapped_target_width = max(16, target_width // 16 * 16)
    output_height = min(max(16, int(fitted_height) // 16 * 16), snapped_target_height)
    output_width = min(max(16, int(fitted_width) // 16 * 16), snapped_target_width)
    return (0, 0, source_height, source_width, output_height, output_width)


def fit_reference_image(
    image: torch.Tensor, target_height: int, target_width: int
) -> torch.Tensor:
    """Apply v1.2 FIT to a ``(B,C,H,W)`` source before VAE encoding."""

    if image.dim() != 4:
        raise ValueError("FIT expects a (B,C,H,W) reference image tensor")
    source_height, source_width = image.shape[-2:]
    crop_y, crop_x, crop_h, crop_w, out_h, out_w = fit_reference_geometry(
        source_height, source_width, target_height, target_width
    )
    image = image[..., crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
    if image.shape[-2:] != (out_h, out_w):
        image = F.interpolate(
            image.float(),
            size=(out_h, out_w),
            mode="bicubic",
            antialias=True,
        )
    return image.clamp(0, 1)
