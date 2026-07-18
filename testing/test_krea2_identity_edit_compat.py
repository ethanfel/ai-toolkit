import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


# Import only Krea2's small source modules. Importing the public
# ``extensions_built_in.diffusion_models`` package eagerly imports every built-in
# architecture and makes these CPU-only unit tests depend on optional packages
# such as diffusers and transformers.
KREA2_SRC = (
    Path(__file__).resolve().parents[1]
    / "extensions_built_in"
    / "diffusion_models"
    / "krea2"
    / "src"
)
TEST_PACKAGE = "_aitk_test_krea2_src"


def _install_test_package():
    package = ModuleType(TEST_PACKAGE)
    package.__path__ = [str(KREA2_SRC)]
    sys.modules.setdefault(TEST_PACKAGE, package)

    if importlib.util.find_spec("diffusers") is None:
        diffusers = ModuleType("diffusers")
        diffusers.__path__ = []
        utils = ModuleType("diffusers.utils")
        utils.__path__ = []
        torch_utils = ModuleType("diffusers.utils.torch_utils")

        def randn_tensor(shape, generator=None, device=None, dtype=None):
            return torch.randn(shape, generator=generator, device=device, dtype=dtype)

        torch_utils.randn_tensor = randn_tensor
        sys.modules.setdefault("diffusers", diffusers)
        sys.modules.setdefault("diffusers.utils", utils)
        sys.modules.setdefault("diffusers.utils.torch_utils", torch_utils)


def _load_krea2_source(module_name):
    qualified_name = f"{TEST_PACKAGE}.{module_name}"
    path = KREA2_SRC / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(qualified_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


_install_test_package()
edit_compat = _load_krea2_source("edit_compat")
pipeline = _load_krea2_source("pipeline")
text_encoder = _load_krea2_source("text_encoder")

AI_TOOLKIT_T0_PROFILE = edit_compat.AI_TOOLKIT_T0_PROFILE
IDENTITY_EDIT_V12_PROFILE = edit_compat.IDENTITY_EDIT_V12_PROFILE
fit_reference_geometry = edit_compat.fit_reference_geometry
fit_reference_image = edit_compat.fit_reference_image
resolve_edit_profile = edit_compat.resolve_edit_profile
pack_ref_latents = pipeline.pack_ref_latents
predict_velocity = pipeline.predict_velocity
build_image_prompt = text_encoder.build_image_prompt


VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"


def test_edit_profile_defaults_to_ai_toolkit_t0():
    assert resolve_edit_profile({}, is_edit=False) == AI_TOOLKIT_T0_PROFILE
    assert resolve_edit_profile({"edit": True}, is_edit=True) == AI_TOOLKIT_T0_PROFILE


@pytest.mark.parametrize(
    "requested",
    ["identity_edit_v12", "identity_edit_v1_2", "conrad_identity_edit_v12"],
)
def test_identity_edit_v12_profile_is_opt_in_and_aliases_are_canonical(requested):
    model_kwargs = {"edit": True, "edit_profile": requested}
    assert resolve_edit_profile(model_kwargs, is_edit=True) == IDENTITY_EDIT_V12_PROFILE


def test_identity_edit_v12_profile_requires_edit_mode():
    with pytest.raises(ValueError, match="requires model_kwargs.edit=true"):
        resolve_edit_profile({"edit_profile": IDENTITY_EDIT_V12_PROFILE}, is_edit=False)


def test_identity_edit_v12_profile_rejects_kv_cache():
    with pytest.raises(ValueError, match="incompatible.*kv_cache=true"):
        resolve_edit_profile(
            {
                "edit": True,
                "edit_profile": IDENTITY_EDIT_V12_PROFILE,
                "kv_cache": True,
            },
            is_edit=True,
        )


def test_unknown_edit_profile_is_rejected():
    with pytest.raises(ValueError, match="Unknown Krea2 edit_profile"):
        resolve_edit_profile(
            {"edit": True, "edit_profile": "not-a-profile"}, is_edit=True
        )


def test_identity_edit_prompt_uses_bare_consecutive_vision_blocks():
    assert build_image_prompt(2, bare=True) == VISION_BLOCK * 2
    assert "Picture" not in build_image_prompt(3, bare=True)


def test_stock_prompt_keeps_numbered_picture_labels():
    assert build_image_prompt(2, bare=False) == (
        f"Picture 1: {VISION_BLOCK}Picture 2: {VISION_BLOCK}"
    )


def test_identity_edit_empty_instruction_still_grounds_the_same_images():
    # The instruction may be empty for the trained unconditional branch, but its
    # ordered reference-image placeholders must remain present.
    assert build_image_prompt(2, bare=True) + "" == VISION_BLOCK * 2
    assert build_image_prompt(0, bare=True) == ""


@pytest.mark.parametrize("bare", [False, True])
def test_image_prompt_rejects_negative_image_count(bare):
    with pytest.raises(ValueError, match="non-negative"):
        build_image_prompt(-1, bare=bare)


def test_fit_reference_geometry_floor_snaps_genuine_mismatch_to_16_pixels():
    # 1200x800 fit-inside a 1024 square is 1024x682.66..., then the width
    # floors to 672 so training and inference share the same transformer grid.
    assert fit_reference_geometry(1200, 800, 1024, 1024) == (
        0,
        0,
        1200,
        800,
        1024,
        672,
    )


def test_fit_reference_geometry_preserves_mismatched_source_without_crop():
    assert fit_reference_geometry(1024, 768, 1024, 1024) == (
        0,
        0,
        1024,
        768,
        1024,
        768,
    )


def test_fit_reference_geometry_near_match_uses_center_crop_and_full_grid():
    # A 1024x960 source is within the official 8% tolerance. The minimal
    # centered crop avoids a narrow unconditioned border in the target grid.
    assert fit_reference_geometry(1024, 960, 1024, 1024) == (
        32,
        0,
        960,
        960,
        1024,
        1024,
    )


def test_fit_reference_geometry_rejects_non_positive_dimensions():
    with pytest.raises(ValueError, match="must be positive"):
        fit_reference_geometry(0, 768, 1024, 1024)


def test_fit_reference_image_resizes_in_pixel_space_and_clamps_bicubic_overshoot():
    source = torch.tensor([[[[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]]])
    fitted = fit_reference_image(source, target_height=32, target_width=16)

    assert fitted.shape == (1, 1, 32, 16)
    assert fitted.min() >= 0
    assert fitted.max() <= 1


def test_fit_reference_positions_are_stride_one_and_centered():
    # Pixel-space 1024x768 becomes a VAE f8 latent of 128x96. With patch=2,
    # its 64x48 RoPE grid is centered in the target's 64x64 grid at x=8.
    ref = torch.zeros(1, 128, 96)
    tokens, pos, mask = pack_ref_latents(
        [[ref]],
        patch=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        target_hw=(128, 128),
        center_positions=True,
    )

    assert tokens.shape == (1, 64 * 48, 4)
    assert mask.all()
    torch.testing.assert_close(pos[0, 0], torch.tensor([1.0, 0.0, 8.0]))
    torch.testing.assert_close(pos[0, -1], torch.tensor([1.0, 63.0, 55.0]))

    grid = pos[0, :, 1:].reshape(64, 48, 2)
    torch.testing.assert_close(grid[:, 1:, 1] - grid[:, :-1, 1], torch.ones(64, 47))
    torch.testing.assert_close(grid[1:, :, 0] - grid[:-1, :, 0], torch.ones(63, 48))


def test_fit_reference_exact_match_has_zero_offset():
    ref = torch.zeros(1, 8, 8)
    _, pos, _ = pack_ref_latents(
        [[ref]],
        patch=2,
        device="cpu",
        dtype=torch.float32,
        target_hw=(8, 8),
        center_positions=True,
    )

    torch.testing.assert_close(pos[0, 0], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(pos[0, -1], torch.tensor([1.0, 3.0, 3.0]))


def test_fit_reference_multi_ref_frames_and_offsets_are_stable():
    vertical = torch.zeros(1, 4, 2)
    horizontal = torch.zeros(1, 2, 4)
    _, pos, mask = pack_ref_latents(
        [[vertical, horizontal]],
        patch=2,
        device="cpu",
        dtype=torch.float32,
        target_hw=(8, 8),
        center_positions=True,
    )

    assert mask.tolist() == [[True, True, True, True]]
    expected = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [2.0, 1.0, 1.0],
            [2.0, 1.0, 2.0],
        ]
    )
    torch.testing.assert_close(pos[0], expected)


class RecordingDenoiser:
    """Minimal callable matching the pipeline's SingleStreamDiT boundary."""

    def __init__(self):
        self.config = SimpleNamespace(patch=1, txtlayers=1)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        img = kwargs["img"]
        reflen = kwargs["reflen"]
        # Stock SingleStreamDiT returns only its target prefix. The Identity
        # compatibility path deliberately calls it with reflen=0 so all packed
        # image tokens share current-t modulation; predict_velocity then removes
        # the reference prefix from this output itself.
        if reflen:
            return img[:, : img.shape[1] - reflen]
        return img


def _prediction_inputs():
    target = torch.tensor([[[[10.0, 11.0], [12.0, 13.0]]]])
    ref_a = torch.tensor([[[20.0, 21.0]]])
    ref_b = torch.tensor([[[30.0]]])
    context = torch.zeros(1, 1, 1)
    text_mask = torch.ones(1, 1, dtype=torch.long)
    timestep = torch.tensor([0.625])
    return target, [[ref_a, ref_b]], context, text_mask, timestep


def test_identity_edit_packs_refs_before_target_and_slices_target_only():
    model = RecordingDenoiser()
    target, refs, context, text_mask, timestep = _prediction_inputs()

    prediction = predict_velocity(
        model,
        target,
        timestep,
        context,
        text_mask,
        ref_latents=refs,
        identity_edit_compat=True,
    )

    call = model.calls[0]
    assert call["reflen"] == 0
    assert call["isolate_refs"] is False
    assert call["img"][0, :, 0].tolist() == [20.0, 21.0, 30.0, 10.0, 11.0, 12.0, 13.0]
    # Position includes one text token followed by ref frames 1/2 and target frame 0.
    assert call["pos"][0, 1:, 0].tolist() == [1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0]
    torch.testing.assert_close(call["t"], timestep)
    torch.testing.assert_close(prediction, target)


def test_stock_pack_order_and_t0_reference_contract_are_unchanged():
    model = RecordingDenoiser()
    target, refs, context, text_mask, timestep = _prediction_inputs()

    prediction = predict_velocity(
        model,
        target,
        timestep,
        context,
        text_mask,
        ref_latents=refs,
        identity_edit_compat=False,
    )

    call = model.calls[0]
    assert call["reflen"] == 3
    assert call["img"][0, :, 0].tolist() == [10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 30.0]
    assert call["pos"][0, 1:, 0].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0]
    torch.testing.assert_close(prediction, target)
