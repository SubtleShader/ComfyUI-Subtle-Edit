# ComfyUI-Subtle-Edit

One node: **Subtle Edit** (`conditioning/subtle_edit`). Edit-LoRA conditioning for MOSS-Image and Krea 2.

| Input | What it does |
|---|---|
| `model` | `moss` or `krea2` |
| `edit_format` | moss: `ai-toolkit (kv_cache off)`, `ai-toolkit`, `Moss-Image-Edit`. krea2: `krea2edit` |
| `encode_res` | Resolution the text encoder sees the images at, must match training. moss: edge of a square pixel budget (AI-Toolkit `vlm_max_pixels` = edge², 384 stock, 512, 768). krea2edit: longest side (Locke's `grounding_px`, 768) |
| `ref_strength` | Attention weight of the images the text encoder sees (meaning, not pixels). 1.0 = off, 0 = images left out of the text encoder. moss: core's `image_strength`. krea2: bias on the DiT's attention to the vision tokens |
| `ref_boost` | Fidelity dial for `image_1`: multiplies the generated image's attention to its latent tokens. 1.0 = off, range 0-20. Optimal value is LoRA-specific |
| `ref_position`, `ref_position_2` | Where `image_1` / `image_2` sit in position space. `aligned (trained)` = on the output layout (normal edits). `beside (right of target)` = right of the canvas: appearance and identity still flow, the layout pin is gone (restaging) |
| `vae`, `image_1`, `image_2` | `image_1` = image to edit, `image_2` = optional extra reference without boost |
| `boost_mask` | Optional MASK on `image_1` (same aspect): the boost only applies to the tokens inside it (e.g. the face). Not connected = whole `image_1` |

Use the node for the negative too, with the same images and settings.

## krea2edit

The layout of [comfyui-krea2edit](https://github.com/lbouaraba/comfyui-krea2edit) by Conrad Locke:
the instruction is encoded together with the image(s) through Qwen3-VL with the krea2edit template
(as `Krea2EditGroundedEncode`), and the images go in as clean reference latents at RoPE frame 1, 2
through core Krea 2's native references (`ref_latents_method` `index`). This gives the same result as
`Krea2EditModelPatch` (`fit`) when the output has the same size as `image_1`, so generate at the size
of the scaled input (the included workflow does). `ref_boost` boosts `image_1` (Locke's `ref_boost_a`
in two-image workflows, `ref_boost` with one image), `boost_mask` is Locke's `ref_boost_mask`, and
`ref_position` / `ref_position_2` are Locke's `ref_position_a` / `ref_position` (scene first, person second).
`bag of features` is not included.

## Model-side internals

Core has no hooks for `ref_boost`, `boost_mask` and `ref_position`, so the pack wraps `extra_conds` and the
diffusion model `forward` of MOSS-Image and Krea 2 (and MOSS `rope`) at import:
- boost: `log(boost)` on the generated image's attention to `image_1`'s (masked) latent tokens
  (MOSS: attention override, Krea 2: `attn1_patch`)
- beside: the reference's w positions start one token right of the canvas (MOSS: `rope`, also for the
  precomputed kv_cache keys; Krea 2: `post_input`)

They do nothing at default settings. For MOSS, `ref_position` is untrained: test it. A ComfyUI update that
changes these internals can disable them; the console then prints `[Subtle-Edit] ... disabled/ignored`.

## Install

Copy the `ComfyUI-Subtle-Edit` folder to `ComfyUI/custom_nodes/` and restart. Remove older versions
(`ComfyUI-MOSS-Edit-Res`, `ComfyUI-Subtle-Moss-Edit`) first. Needs a ComfyUI with native MOSS-Image
and Krea 2 support (tested against master `0decb39`, 8 Nov 2026). No extra dependencies.

## Workflows

`workflow/subtle_edit_moss.json` and `workflow/subtle_edit_krea2.json`: pick your own model, LoRA,
text encoder (with vision weights) and VAE files.
