"""Subtle Edit: edit-LoRA text encode for MOSS-Image and Krea 2.

model = moss:
  Same as core ComfyUI's "Text Encode MOSS-Image Edit" (comfy_extras/nodes_moss_image.py), plus
  encode_res (square pixel budget of the Qwen3.5 vision input, AI-Toolkit
  model_kwargs.vlm_max_pixels = encode_res * encode_res) and ref_strength (core's image_strength).
  For krea2, ref_strength adds log(strength) to the DiT's attention on the vision tokens of the text.

model = krea2, edit_format = krea2edit:
  The layout of Conrad Locke's comfyui-krea2edit / krea2edit-trainer
  (https://github.com/lbouaraba/comfyui-krea2edit):
    - semantic path: instruction encoded together with the image(s) through Qwen3-VL, with the
      krea2edit template (Krea2EditGroundedEncode); encode_res = its grounding_px (longest side)
    - appearance path: VAE latents as clean reference tokens at RoPE frame 1, 2, sharing the
      target's timestep (core Krea 2 ref_latents_method "index"), h/w aligned with the target
  Matches Krea2EditModelPatch (fit) when the output size equals image_1's size.

Both models:
  ref_boost:    fidelity dial for image_1 (like ref_boost of the Krea 2 edit nodes): adds log(boost)
                to the generated image's attention logits on image_1's latent tokens. 1.0 = off.
  boost_mask:   optional MASK on image_1: the boost only applies to the tokens inside it.
  ref_position: per image, "aligned (trained)" = reference h/w on the output layout (training),
                "beside (right of target)" = reference shifted right of the canvas on the w axis,
                so the layout pin is gone (like ref_position of the Krea 2 edit nodes).
"""
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

import comfy.conds
import comfy.text_encoders.qwen_vl
import comfy.utils
import node_helpers

TAG = "[Subtle-Edit]"
MODELS = ["moss", "krea2"]
POSITIONS = ["aligned (trained)", "beside (right of target)"]

# edit_format -> (model, reference method, whether the text encoder also sees the references)
EDIT_FORMATS = {
    "ai-toolkit (kv_cache off)": ("moss", "index_timestep_zero", True),
    "ai-toolkit": ("moss", "index_timestep_zero_isolated", True),
    "Moss-Image-Edit": ("moss", "index_timestep_zero_text_start", False),
    "krea2edit": ("krea2", "index", True),
}

SE_KEY = "subtle_edit"          # conditioning / model kwarg carrying the settings
POS_KEY = "subtle_edit_ref_pos"  # transformer_options key for MOSS reference positions

KREA2EDIT_SYSTEM = ("Describe the image by detailing the color, shape, size, texture, quantity, text, "
                    "spatial relationships of the objects and background:")


def _krea2edit_template(nimg: int) -> str:
    vis = "<|vision_start|><|image_pad|><|vision_end|>" * nimg
    return ("<|im_start|>system\n" + KREA2EDIT_SYSTEM + "<|im_end|>\n<|im_start|>user\n"
            + vis + "{}<|im_end|>\n<|im_start|>assistant\n")


def _fit_area(image: torch.Tensor, max_pixels: int) -> torch.Tensor:
    """Downscale (never upscale) a (B, H, W, C) image to fit max_pixels, keeping aspect."""
    h, w = image.shape[1], image.shape[2]
    scale = min(1.0, math.sqrt(max_pixels / float(h * w)))
    nw, nh = max(round(w * scale), 1), max(round(h * scale), 1)
    if (nh, nw) == (h, w):
        return image
    return comfy.utils.common_upscale(image.movedim(-1, 1), nw, nh, "area", "disabled").movedim(1, -1)


def _fit_longest(image: torch.Tensor, longest: int) -> torch.Tensor:
    """krea2edit grounding: cap the longest side (area downscale, never up), RGB only."""
    h, w = image.shape[1], image.shape[2]
    if longest and max(h, w) > longest:
        s = longest / float(max(h, w))
        image = comfy.utils.common_upscale(image.movedim(-1, 1), round(w * s), round(h * s), "area", "disabled").movedim(1, -1)
    return image[:, :, :, :3]


def _krea2_vision_bias(tokens, weight: float) -> Optional[torch.Tensor]:
    """Per-token logit bias over Krea 2's conditioning sequence: `weight` on the vision tokens.
    Krea2TEModel drops everything up to and including the second <|im_start|> + "user\n";
    each image entry expands to (grid_h * grid_w) / 4 tokens (Qwen3-VL merges 2x2 patches of 16 px)."""
    toks = next(iter(tokens.values()))[0]
    end, seen = -1, 0
    for i, t in enumerate(toks):
        v = t[0]
        if not isinstance(v, dict) and not torch.is_tensor(v) and v == 151644 and seen < 2:
            end, seen = i, seen + 1
    if end >= 0 and end + 2 < len(toks) and toks[end + 1][0] == 872 and toks[end + 2][0] == 198:
        end += 3
    bias = []  # type: List[float]
    for t in toks[max(end, 0):]:
        if isinstance(t[0], dict):
            _, grid = comfy.text_encoders.qwen_vl.process_qwen2vl_images(t[0]["data"], patch_size=16)
            bias += [weight] * (int(grid[0, 1] * grid[0, 2]) // 4)
        else:
            bias.append(0.0)
    return torch.tensor(bias)


def _is_beside(mode: str) -> bool:
    return str(mode).startswith("beside")


def _settings(se, n_refs: int) -> Tuple[Tuple[float, ...], Tuple[str, ...], Optional[torch.Tensor]]:
    boosts = tuple(float(b) for b in list(se.get("boosts", ()))[:n_refs])
    boosts += (1.0,) * max(0, n_refs - len(boosts))
    pos = tuple(str(p) for p in list(se.get("positions", ()))[:n_refs])
    pos += (POSITIONS[0],) * max(0, n_refs - len(pos))
    return boosts, pos, se.get("mask", None)


def _boost_blocks(boosts, lens, first_col: int, mask: Optional[torch.Tensor]) -> List[tuple]:
    """(start column, length, boost, token indices or None) per reference."""
    blocks, col = [], first_col
    for i, (b, n) in enumerate(zip(boosts, lens)):
        idx = None
        if i == 0 and mask is not None:
            if mask.numel() == n:
                idx = mask.reshape(-1).nonzero().flatten()
            else:
                print("{} boost_mask ignored: {} tokens, image_1 has {}".format(TAG, mask.numel(), n))
        blocks.append((col, n, b, idx))
        col += n
    return blocks


def _bias_mask(lq: int, lk: int, rows: Tuple[int, int], blocks: List[tuple], device, dtype) -> torch.Tensor:
    """(1, 1, lq, lk) additive mask: log(boost) on rows x each reference's (masked) columns."""
    bias = torch.zeros((1, 1, lq, lk), device=device, dtype=dtype)
    for start, n, b, idx in blocks:
        if b == 1.0:
            continue
        v = math.log(max(b, 1e-4))
        if idx is None:
            bias[:, :, rows[0]:rows[1], start:start + n] = v
        elif idx.numel() > 0:
            bias[:, :, rows[0]:rows[1], start + idx.to(device)] = v
    return bias


def _add_mask(mask: Optional[torch.Tensor], bias: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return bias
    if mask.ndim == 2:
        mask = mask[:, None, None]
    elif mask.ndim == 3:
        mask = mask[:, None]
    return mask.to(device=bias.device, dtype=bias.dtype) + bias


# ---------------------------------------------------------------------------
# Model-side support (ref_boost, boost_mask, ref_position)
#
# Core has no hooks for these, so small wrappers are installed at import. They do nothing
# unless a conditioning carries Subtle Edit settings that differ from the defaults.
#
# MOSS:   sequence [image | text | refs] (kv_cache: ref keys precomputed in extra_conds and
#         appended at the end) -> refs are the LAST key columns. Boost via the attention
#         override, positions via the model's rope() (also used for the precomputed keys).
# Krea 2: sequence [text | target | refs] -> boost via attn1_patch, positions via post_input.
# Boost rows = generated image tokens only (target -> reference attention).
# ---------------------------------------------------------------------------

def _install_moss() -> bool:
    try:
        import comfy.model_base as model_base
        import comfy.ldm.moss_image.model as moss_model
        from comfy.ldm.flux.math import rope as flux_rope
        base_cls, dit_cls = model_base.MOSSImage, moss_model.MOSSImage
    except Exception as e:
        print("{} MOSS support disabled: {}".format(TAG, e))
        return False
    if getattr(dit_cls, "_subtle_edit_installed", False):
        return True
    orig_extra_conds, orig_forward, orig_rope = base_cls.extra_conds, dit_cls.forward, dit_cls.rope

    def rope(self, h, w, text_len, device, transformer_options={}, ref_shapes=(), f_offset=0):
        pe = orig_rope(self, h, w, text_len, device, transformer_options, ref_shapes, f_offset)
        ctx = transformer_options.get(POS_KEY, None) or getattr(self, "_subtle_edit_pos_ctx", None)
        if not ctx or len(ref_shapes) == 0:
            return pe
        positions, target_w = ctx
        # target w positions are centered: -(W-1)/2 .. (W-1)/2; "beside" starts one token right of it
        x0 = (float(target_w) - 1.0) / 2.0 + 1.0
        start = h * w + text_len
        for i, (rh, rw) in enumerate(ref_shapes):
            n = rh * rw
            if i < len(positions) and _is_beside(positions[i]):
                ids = torch.zeros(1, n, 4, device=device, dtype=torch.float32)
                ids[0, :, 0] = i + 1
                ids[0, :, 2] = (torch.arange(rh, device=device, dtype=torch.float32) - (rh - 1) / 2).repeat_interleave(rw)
                ids[0, :, 3] = torch.arange(rw, device=device, dtype=torch.float32).repeat(rh) + x0
                ref_pe = torch.cat([flux_rope(ids[..., a], d, 10000) for a, d in enumerate(self.rope_axes_dim)], dim=-3)
                pe[:, start:start + n] = ref_pe.unsqueeze(1).transpose(1, 2).to(pe.dtype)
                x0 += rw
            start += n
        return pe

    def extra_conds(self, **kwargs):
        se, refs = kwargs.get(SE_KEY, None), kwargs.get("reference_latents", None)
        if not se or not refs:
            return orig_extra_conds(self, **kwargs)
        boosts, positions, mask = _settings(se, len(refs))
        noise = kwargs.get("noise", None)
        beside = any(_is_beside(p) for p in positions) and noise is not None
        dm = self.diffusion_model
        if beside:  # the kv_cache path precomputes the reference keys in here
            dm._subtle_edit_pos_ctx = (positions, int(noise.shape[-1]))
        try:
            out = orig_extra_conds(self, **kwargs)
        finally:
            if beside:
                del dm._subtle_edit_pos_ctx
        lens = tuple(int(r.shape[-2] * r.shape[-1]) for r in refs)
        out[SE_KEY] = comfy.conds.CONDConstant({"boosts": boosts, "positions": positions, "mask": mask, "lens": lens})
        return out

    def make_override(blocks, n_ref, hw, text_len, prev):
        expected_k = hw + text_len + n_ref
        cache = {}  # type: Dict[tuple, torch.Tensor]

        def override(func, *args, **kwargs):
            q, k = args[0], args[1]
            if (not kwargs.get("skip_reshape", False)) and q.ndim == 3 and k.shape[1] == expected_k and q.shape[1] >= hw:
                lq, lk = q.shape[1], k.shape[1]
                key = (lq, lk, q.device, q.dtype)
                if key not in cache:
                    shifted = [(s - expected_k + lk, n, b, idx) for s, n, b, idx in blocks]
                    cache[key] = _bias_mask(lq, lk, (0, hw), shifted, q.device, q.dtype)
                kwargs["mask"] = _add_mask(kwargs.get("mask", None), cache[key])
            if prev is not None:
                return prev(func, *args, **kwargs)
            return func(*args, **kwargs)
        return override

    def forward(self, x, timestep, context, transformer_options={}, **kwargs):
        se = kwargs.pop(SE_KEY, None)
        if se:
            lens = tuple(se["lens"])
            boosts, positions, mask = _settings(se, len(lens))
            transformer_options = transformer_options.copy()
            if any(_is_beside(p) for p in positions):
                transformer_options[POS_KEY] = (positions, int(x.shape[-1]))
            if any(b != 1.0 for b in boosts):
                hw, txt = int(x.shape[-2] * x.shape[-1]), int(context.shape[1])
                blocks = _boost_blocks(boosts, lens, hw + txt, mask)
                transformer_options["optimized_attention_override"] = make_override(
                    blocks, sum(lens), hw, txt, transformer_options.get("optimized_attention_override", None))
        return orig_forward(self, x, timestep, context, transformer_options, **kwargs)

    base_cls.extra_conds = extra_conds
    dit_cls.forward = forward
    dit_cls.rope = rope
    dit_cls._subtle_edit_installed = True
    return True


def _install_krea2() -> bool:
    try:
        import comfy.model_base as model_base
        import comfy.ldm.krea2.model as krea2_model
        base_cls, dit_cls = model_base.Krea2, krea2_model.SingleStreamDiT
    except Exception as e:
        print("{} Krea 2 support disabled: {}".format(TAG, e))
        return False
    if getattr(dit_cls, "_subtle_edit_installed", False):
        return True
    orig_extra_conds, orig_forward = base_cls.extra_conds, dit_cls.forward

    def extra_conds(self, **kwargs):
        out = orig_extra_conds(self, **kwargs)
        se, refs = kwargs.get(SE_KEY, None), kwargs.get("reference_latents", None)
        if se and refs:
            boosts, positions, mask = _settings(se, len(refs))
            out[SE_KEY] = comfy.conds.CONDConstant({"boosts": boosts, "positions": positions, "mask": mask,
                                                     "text_bias": se.get("text_bias", None)})
        return out

    def grid(lat, patch) -> Tuple[int, int]:
        return int(math.ceil(lat.shape[-2] / patch)), int(math.ceil(lat.shape[-1] / patch))

    def forward(self, x, timesteps, context, attention_mask=None, ref_latents=None, transformer_options={}, **kwargs):
        se = kwargs.pop(SE_KEY, None)
        if se and ref_latents and kwargs.get("ref_latents_method", self.default_ref_method) is not None:
            patch = self.patch
            boosts, positions, mask = _settings(se, len(ref_latents))
            txt = int(context.shape[1])
            th, tw = grid(x, patch)
            img = th * tw
            lens = [a * b for a, b in (grid(r, patch) for r in ref_latents)]
            transformer_options = transformer_options.copy()
            patches = dict(transformer_options.get("patches", {}))

            if any(_is_beside(p) for p in positions):
                def post_input(args):
                    ids = args["img_ids"].clone()
                    start, x0 = img, float(tw)
                    for n, p in zip(lens, positions):
                        if _is_beside(p):
                            blk = ids[:, start:start + n, 2]
                            blk = blk - blk.min() + x0
                            ids[:, start:start + n, 2] = blk
                            x0 = float(blk.max()) + 1.0
                        start += n
                    return {"img": args["img"], "txt": args["txt"], "img_ids": ids, "txt_ids": args["txt_ids"]}
                patches["post_input"] = list(patches.get("post_input", [])) + [post_input]

            text_bias = se.get("text_bias", None)
            if text_bias is not None and text_bias.numel() != txt:
                print("{} ref_strength ignored: conditioning has {} tokens, bias {}".format(TAG, txt, text_bias.numel()))
                text_bias = None
            boosted = any(b != 1.0 for b in boosts)
            if boosted or text_bias is not None:
                expected = txt + img + sum(lens)
                blocks = _boost_blocks(boosts, lens, txt + img, mask) if boosted else []
                cache = {}  # type: Dict[tuple, torch.Tensor]

                def attn1_patch(q, k, v, pe=None, attn_mask=None, extra_options=None):
                    lq, lk = q.shape[2], k.shape[2]   # q, k: (B, H, L, D)
                    if lk != expected or lq != lk:
                        return {}
                    key = (lq, lk, q.device, q.dtype)
                    if key not in cache:
                        bias = _bias_mask(lq, lk, (txt, txt + img), blocks, q.device, q.dtype)
                        if text_bias is not None:  # ref_strength: all rows, vision-token columns of the text
                            bias[:, :, :, :txt] += text_bias.to(q.device, q.dtype)
                        cache[key] = bias
                    return {"attn_mask": _add_mask(attn_mask, cache[key])}
                patches["attn1_patch"] = list(patches.get("attn1_patch", [])) + [attn1_patch]

            transformer_options["patches"] = patches
        return orig_forward(self, x, timesteps, context, attention_mask, ref_latents, transformer_options, **kwargs)

    base_cls.extra_conds = extra_conds
    dit_cls.forward = forward
    dit_cls._subtle_edit_installed = True
    return True


MODEL_SUPPORT = {"moss": _install_moss(), "krea2": _install_krea2()}


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def _mask_tokens(mask: torch.Tensor, grid_hw: Tuple[int, int], image: torch.Tensor) -> Optional[torch.Tensor]:
    """MASK (B, H, W) on image_1 -> bool per reference token (row-major), token >50% inside."""
    m = mask
    if m.ndim == 2:
        m = m[None]
    m = m[:1].float()
    ih, iw = image.shape[1], image.shape[2]
    if abs(m.shape[-1] / float(m.shape[-2]) - iw / float(ih)) > 0.02 * (iw / float(ih)):
        print("{} boost_mask aspect {}x{} differs from image_1 {}x{}".format(TAG, m.shape[-1], m.shape[-2], iw, ih))
    t = F.interpolate(m[None], size=grid_hw, mode="area")[0, 0] > 0.5
    if not bool(t.any()):
        print("{} boost_mask is empty: boost applies to the whole image_1".format(TAG))
        return None
    return t.reshape(-1).cpu()


class SubtleEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "model": (MODELS, {"default": "moss", "tooltip": "Base model the edit LoRA was trained on."}),
                "edit_format": (list(EDIT_FORMATS), {
                    "default": "ai-toolkit (kv_cache off)",
                    "tooltip": "Layout the edit LoRA was trained with. moss: ai-toolkit (kv_cache off), ai-toolkit, "
                               "Moss-Image-Edit. krea2: krea2edit.",
                }),
                "encode_res": ("INT", {
                    "default": 512, "min": 64, "max": 2048, "step": 32,
                    "tooltip": "Resolution the text encoder sees each image at. Must match the LoRA's training. "
                               "moss (ai-toolkit): edge of a square pixel budget, 384 = stock (vlm_max_pixels 147456), "
                               "512 = 262144, 768 = 589824. krea2edit: longest side (grounding_px), 768 default.",
                }),
                "ref_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 100.0, "step": 0.05,
                    "tooltip": "Attention weight of the images the text encoder sees (meaning, not pixels). "
                               "0 leaves them out, 1.0 = unchanged (off), above 1 emphasizes them.",
                }),
                "ref_boost": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 20.0, "step": 0.01,
                    "tooltip": "Fidelity dial for image_1: multiplies the generated image's attention to its "
                               "latent tokens (only inside boost_mask, if connected). 1.0 = off, >1 sticks closer "
                               "to its appearance, <1 loosens. Optimal value is LoRA-specific.",
                }),
                "ref_position": (POSITIONS, {
                    "default": POSITIONS[0],
                    "tooltip": "Where image_1 sits in position space. aligned (trained) = pinned onto the output "
                               "layout, how edit LoRAs are trained (use for normal edits). beside = placed right of "
                               "the canvas: appearance/identity still flows, the layout pin is gone (restaging).",
                }),
                "ref_position_2": (POSITIONS, {
                    "default": POSITIONS[0],
                    "tooltip": "The same choice for image_2. No effect without image_2.",
                }),
                "vae": ("VAE", {"tooltip": "Encodes the images as reference latents for the edit LoRA."}),
                "image_1": ("IMAGE", {"tooltip": "The image to edit (Picture 1 / frame 1)."}),
            },
            "optional": {
                "image_2": ("IMAGE", {"tooltip": "Optional extra reference (Picture 2 / frame 2), no boost."}),
                "boost_mask": ("MASK", {"tooltip": "Optional region on image_1 (same aspect) the boost is limited "
                                                   "to, e.g. the face. Not connected = whole image_1."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "conditioning/subtle_edit"
    DESCRIPTION = (
        "Subtle Edit: edit-LoRA encode for MOSS-Image and Krea 2 (krea2edit by Locke) with selectable vision "
        "resolution, a boost for image_1 (optionally masked) and reference placement. Use it for the negative "
        "too, with the same images and settings. References are VAE-encoded at the size they come in: scale "
        "them first and generate at image_1's size."
    )

    def encode(self, clip, prompt, model, edit_format, encode_res, ref_strength, ref_boost, ref_position,
               ref_position_2, vae, image_1, image_2=None, boost_mask=None):
        fmt_model, method, vision = EDIT_FORMATS[edit_format]
        if fmt_model != model:
            raise ValueError("Subtle Edit: edit_format '{}' is for {}, but model is set to {}.".format(
                edit_format, fmt_model, model))
        images = [im for im in (image_1, image_2) if im is not None]
        boosts = [float(ref_boost)] + [1.0] * (len(images) - 1)
        positions = [ref_position, ref_position_2][:len(images)]

        text_bias = None
        if model == "krea2":
            conditioning, text_bias = self._encode_krea2(clip, prompt, images, int(encode_res), ref_strength)
        else:
            conditioning = self._encode_moss(clip, prompt, images, vision, vae, int(encode_res), ref_strength)

        if vae is None:
            return (conditioning,)
        refs = [vae.encode(im[:, :, :, :3]) for im in images]
        values = {"reference_latents": refs, "reference_latents_method": method}

        boosted = boosts[0] != 1.0
        placed = any(_is_beside(p) for p in positions)
        if boosted or placed or text_bias is not None:
            if not MODEL_SUPPORT.get(model, False):
                print("{} ref_boost/ref_position ignored: model support could not be installed".format(TAG))
            mask = None
            if boosted and boost_mask is not None:
                lh, lw = refs[0].shape[-2], refs[0].shape[-1]
                grid_hw = (lh, lw) if model == "moss" else (int(math.ceil(lh / 2)), int(math.ceil(lw / 2)))
                mask = _mask_tokens(boost_mask, grid_hw, image_1)
            values[SE_KEY] = {"boosts": tuple(boosts), "positions": tuple(positions), "mask": mask,
                              "text_bias": text_bias}
        return (node_helpers.conditioning_set_values(conditioning, values),)

    @staticmethod
    def _encode_krea2(clip, prompt, images, longest, ref_strength):
        imgs = [_fit_longest(im, longest) for im in images] if ref_strength > 0 else []
        tokens = clip.tokenize(prompt, images=imgs, llama_template=_krea2edit_template(len(imgs)))
        text_bias = None
        if len(imgs) > 0 and ref_strength != 1.0:
            text_bias = _krea2_vision_bias(tokens, math.log(ref_strength))
        return clip.encode_from_tokens_scheduled(tokens), text_bias

    @staticmethod
    def _encode_moss(clip, prompt, images, vision, vae, edge, ref_strength):
        vl = []  # type: List[torch.Tensor]
        if (vision or vae is None) and ref_strength > 0:
            vl = [_fit_area(im, edge * edge) for im in images]
            prompt = "".join(
                "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1) for i in range(len(vl))
            ) + prompt
        tokens = clip.tokenize(prompt, images=vl)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if len(vl) > 0 and ref_strength != 1.0:
            # log(strength) on each image's vision tokens scales their share of attention;
            # Qwen3.5 merges 2x2 patches of 16 px into one token
            weight = math.log(ref_strength)
            bias = []  # type: List[float]
            for t in next(iter(tokens.values()))[0]:
                if isinstance(t[0], dict):
                    _, grid = comfy.text_encoders.qwen_vl.process_qwen2vl_images(t[0]["data"], patch_size=16)
                    bias += [weight] * (int(grid[0, 1] * grid[0, 2]) // 4)
                else:
                    bias.append(0.0)
            conditioning = node_helpers.conditioning_set_values(
                conditioning, {"context_attention_bias": torch.tensor([bias])})
        return conditioning


NODE_CLASS_MAPPINGS = {
    "SubtleEdit": SubtleEdit,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SubtleEdit": "Subtle Edit",
}
