"""Convert a "community" Krea 2 checkpoint (original krea `mmdit.py` / SingleStreamDiT weight
naming, e.g. `CalamitousFelicitousness/Krea-2-Base-Diffusers` or the bucket `raw.safetensors`)
into a layout that `diffusers.Krea2Transformer2DModel.from_pretrained` can load.

The community repos ship the transformer in the *original* parameter naming
(`blocks.0.attn.wq`, `mod.lin`, `last.up`/`last.down`, top-level `first/tmlp/tproj/...`) inside a
diffusers folder structure. diffusers' `Krea2Transformer2DModel` expects renamed keys
(`transformer_blocks.0.attn.to_q`, `scale_shift_table`, `img_in/time_embed/time_mod_proj/...`).
This script renames the transformer state dict (streaming, shard by shard), reshapes the per-block
modulation table, drops the two final-layer `up`/`down` weights that diffusers' `Krea2FinalLayer`
does not implement, writes a correct `config.json`, and copies/symlinks the remaining (already
diffusers-format) subfolders so the result is a complete pipeline directory.

NOTE on the dropped `last.up`/`last.down`: the original `LastLayer` adds a *purely linear*
residual `up(down(x))` on the pre-norm hidden state. diffusers' `Krea2FinalLayer` has no slot for
it and it cannot be folded into `linear(adaLN(norm(x)))`, so any diffusers-based Krea 2 (including
the official gated diffusers repo) necessarily omits it. The converted model is therefore identical
to what the official diffusers weights would be, but not bit-exact with krea's own `inference.py`.

Usage:
    python scripts/convert_krea2_community_to_diffusers.py \
        --src /media/p5/models/krea2-base-src \
        --dst /media/p5/models/krea2-base-diffusers
"""

import argparse
import json
import os
import re
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Other diffusers subfolders are already in correct format and are reused as-is.
SUBFOLDERS_TO_REUSE = ["vae", "text_encoder", "tokenizer", "scheduler"]
ROOT_FILES_TO_REUSE = ["model_index.json"]

# Architecture is fixed (12.9B Krea 2 Base/Turbo); all values verified against the original weight
# shapes and equal to diffusers' Krea2Transformer2DModel defaults.
DIFFUSERS_TRANSFORMER_CONFIG = {
    "_class_name": "Krea2Transformer2DModel",
    "_diffusers_version": "0.39.0.dev0",
    "in_channels": 64,
    "num_layers": 28,
    "attention_head_dim": 128,
    "num_attention_heads": 48,
    "num_key_value_heads": 12,
    "intermediate_size": 16384,
    "timestep_embed_dim": 256,
    "text_hidden_dim": 2560,
    "num_text_layers": 12,
    "text_num_attention_heads": 20,
    "text_num_key_value_heads": 20,
    "text_intermediate_size": 6912,
    "num_layerwise_text_blocks": 2,
    "num_refiner_text_blocks": 2,
    "axes_dims_rope": [32, 48, 48],
    "rope_theta": 1000.0,
    "norm_eps": 1e-5,
}

# Keys present in the original checkpoint that diffusers' Krea2FinalLayer cannot represent.
DROP_KEYS = {"last.up.weight", "last.down.weight"}


def rename_key(k: str) -> str:
    """Map an original Krea `SingleStreamDiT` parameter name to its diffusers equivalent.

    Order matters: the per-block / text-fusion-block submodule renames (`.attn.*`, `.mlp.*`,
    `.prenorm`/`.postnorm`) are applied to *any* depth, so they also rewrite the text-fusion
    blocks. Top-level module renames are anchored with `^`.
    """
    k = re.sub(r"^blocks\.", "transformer_blocks.", k)
    # attention submodules (also inside text-fusion blocks)
    k = k.replace(".attn.wq.", ".attn.to_q.").replace(".attn.wk.", ".attn.to_k.")
    k = k.replace(".attn.wv.", ".attn.to_v.").replace(".attn.wo.", ".attn.to_out.0.")
    k = k.replace(".attn.gate.", ".attn.to_gate.")
    k = k.replace(".attn.qknorm.qnorm.scale", ".attn.norm_q.weight")
    k = k.replace(".attn.qknorm.knorm.scale", ".attn.norm_k.weight")
    # feed-forward + norms
    k = k.replace(".mlp.", ".ff.")
    k = k.replace(".prenorm.scale", ".norm1.weight").replace(".postnorm.scale", ".norm2.weight")
    # per-block modulation table (reshape handled by caller)
    k = k.replace(".mod.lin", ".scale_shift_table")
    # top-level modules
    k = k.replace("first.", "img_in.")
    k = re.sub(r"^tmlp\.0\.", "time_embed.linear_1.", k)
    k = re.sub(r"^tmlp\.2\.", "time_embed.linear_2.", k)
    k = re.sub(r"^tproj\.1\.", "time_mod_proj.", k)
    k = re.sub(r"^txtmlp\.0\.scale$", "txt_in.norm.weight", k)
    k = re.sub(r"^txtmlp\.1\.", "txt_in.linear_1.", k)
    k = re.sub(r"^txtmlp\.3\.", "txt_in.linear_2.", k)
    k = k.replace("txtfusion.", "text_fusion.")
    # final layer
    k = k.replace("last.modulation.lin", "final_layer.scale_shift_table")
    k = k.replace("last.norm.scale", "final_layer.norm.weight")
    k = re.sub(r"^last\.linear\.", "final_layer.linear.", k)
    return k


def convert_transformer(src_tf: str, dst_tf: str):
    os.makedirs(dst_tf, exist_ok=True)
    index_path = os.path.join(src_tf, "diffusion_pytorch_model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        # single-file checkpoint
        shards = ["diffusion_pytorch_model.safetensors"]

    new_weight_map = {}
    total_size = 0
    n_renamed = 0
    n_dropped = 0
    n_reshaped = 0

    for shard in shards:
        src_path = os.path.join(src_tf, shard)
        out_tensors = {}
        with safe_open(src_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in DROP_KEYS:
                    n_dropped += 1
                    continue
                t = f.get_tensor(key)
                nk = rename_key(key)
                # per-block modulation table: flat (6*H,) -> (6, H). Final-layer table is already 2D.
                if nk.endswith("scale_shift_table") and t.ndim == 1:
                    t = t.reshape(6, -1).contiguous()
                    n_reshaped += 1
                out_tensors[nk] = t
                new_weight_map[nk] = shard
                total_size += t.numel() * t.element_size()
                n_renamed += 1
        save_file(out_tensors, os.path.join(dst_tf, shard), metadata={"format": "pt"})
        del out_tensors
        print(f"  shard {shard}: wrote {len(new_weight_map)} cumulative keys")

    # index.json (only meaningful when sharded; harmless otherwise)
    if len(shards) > 1:
        with open(os.path.join(dst_tf, "diffusion_pytorch_model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": total_size}, "weight_map": new_weight_map}, f, indent=2)

    with open(os.path.join(dst_tf, "config.json"), "w") as f:
        json.dump(DIFFUSERS_TRANSFORMER_CONFIG, f, indent=2)

    print(f"transformer: renamed {n_renamed}, reshaped {n_reshaped} tables, dropped {n_dropped} "
          f"(expected 2: last.up/last.down)")


def link_or_copy(src: str, dst: str, symlink: bool):
    if os.path.lexists(dst):
        return
    if symlink:
        os.symlink(os.path.abspath(src), dst)
    elif os.path.isdir(src):
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="downloaded community Krea2 diffusers-layout dir")
    ap.add_argument("--dst", required=True, help="output diffusers model dir")
    ap.add_argument("--copy", action="store_true",
                    help="copy reused subfolders instead of symlinking (default: symlink to save disk)")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    print(f"Converting transformer {args.src}/transformer -> {args.dst}/transformer")
    convert_transformer(os.path.join(args.src, "transformer"), os.path.join(args.dst, "transformer"))

    for sub in SUBFOLDERS_TO_REUSE:
        s = os.path.join(args.src, sub)
        if os.path.exists(s):
            link_or_copy(s, os.path.join(args.dst, sub), symlink=not args.copy)
            print(f"  reused subfolder: {sub}")
    for fn in ROOT_FILES_TO_REUSE:
        s = os.path.join(args.src, fn)
        if os.path.exists(s):
            link_or_copy(s, os.path.join(args.dst, fn), symlink=False)
            print(f"  reused file: {fn}")

    print(f"\nDone. Load with: Krea2Pipeline.from_pretrained('{args.dst}', torch_dtype=torch.bfloat16)")


if __name__ == "__main__":
    main()
