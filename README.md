# ComfyUI-Subtle-Edit

**Subtle Edit** is one ComfyUI node for editing images. You give it a picture and an instruction like
"Change the jacket to red leather", and the model changes only what you asked for.

It needs an editing-capable model. That can be either:

- an **edit LoRA** loaded on top of a normal base model, or
- an **edit model**: a checkpoint where the edit functionality is already built-in. Then you don't need a LoRA at all.

It works with two model families:

- **MOSS-Image** with edits trained in AI-Toolkit (or the Moss-Image-Edit LoRA)
- **Krea 2** with the krea2edit LoRAs by Conrad Locke

You find the node under **conditioning → subtle_edit → Subtle Edit**.

---

## Installation

Open a command prompt (cmd), go to your ComfyUI `custom_nodes` folder and clone the repository.
Change the path to where your ComfyUI is:

```bat
cd /d C:\ComfyUI\custom_nodes
git clone https://github.com/SubtleShader/ComfyUI-Subtle-Edit
```

Then restart ComfyUI.

**Update** to the newest version:

```bat
cd /d C:\ComfyUI\custom_nodes\ComfyUI-Subtle-Edit
git pull
```

Then restart ComfyUI.

Without git: download the zip, unzip it and copy the `ComfyUI-Subtle-Edit` folder into `ComfyUI/custom_nodes/`.

You need a recent ComfyUI that supports MOSS-Image and Krea 2. Nothing else has to be installed.

---

## Quick start

The easiest way is to open one of the ready-made workflows from the `workflow` folder
(drag the file into ComfyUI):

- `subtle_edit_moss.json` for MOSS-Image
- `subtle_edit_krea2.json` for Krea 2

Then:

1. Select your own files in the loader nodes: model, text encoder and VAE.
   - **With an edit LoRA:** pick your base model and select the LoRA in the LoRA loader.
   - **With an edit model:** pick the edit model and bypass the LoRA loader (select it and press Ctrl+B) or delete it.
   - The text encoder must be a version **with vision weights** (the official ComfyUI files have them).
2. Load the picture you want to edit.
3. Type what you want to change into the prompt of the top Subtle Edit node.
4. Press **Run**.

The workflow makes the output the same size as your picture (about 1 megapixel). Keep it that way:
the edit works best when the output and the input picture have the same size.

---

## The settings

You only need the first four for normal use. Leave everything else at its default until you
want to experiment.

### Basic settings

| Setting | What to choose |
|---|---|
| **prompt** | What to change, e.g. *"Make it night"* or *"Change her hair to blonde"*. |
| **model** | `moss` for MOSS-Image, `krea2` for Krea 2. |
| **edit_format** | How your edit LoRA or edit model was trained. Ask the author if unsure. MOSS: `ai-toolkit (kv_cache off)`, `ai-toolkit` or `Moss-Image-Edit`. Krea 2: `krea2edit`. It must fit the selected model, otherwise you get an error. |
| **encode_res** | How big the text encoder "looks" at your picture. Must match how the edit LoRA or model was trained. Typical: **512** for MOSS LoRAs trained at 512, **384** for standard AI-Toolkit MOSS LoRAs, **768** for krea2edit. |

### Optional settings

| Setting | What it does | Default |
|---|---|---|
| **ref_strength** | How much the model pays attention to *what is in* the picture (people, objects, scene). Higher = follows the picture's content more, lower = follows the prompt more. 0 = the text side ignores the picture. | 1.0 (off) |
| **ref_boost** | How closely the result keeps the *look* of your picture (face, colors, details). Higher = closer to the original, lower = more freedom to change. Too high can block the edit. Try 1.2 to 1.5 for MOSS. Locke's Krea 2 example uses 4. | 1.0 (off) |
| **ref_position** | `aligned (trained)` = the result keeps the layout of your picture. Use this for normal edits. `beside (right of target)` = the model still uses the person or object from your picture, but is free to change pose and framing. Experimental. | aligned |
| **ref_position_2** | The same choice for the second picture. | aligned |

A value of **1.0** for ref_strength and ref_boost means "no change". The node then behaves exactly as without these settings.

### Inputs (the connection dots)

| Input | What to connect |
|---|---|
| **clip** | The text encoder from your Load CLIP node. |
| **vae** | The VAE from your Load VAE node. |
| **image_1** | The picture you want to edit. |
| **image_2** | *Optional.* A second picture, e.g. a person or a piece of clothing you want to bring in. Mention it in the prompt ("the person from picture 2"). |
| **boost_mask** | *Optional.* A mask on image_1. ref_boost then only works inside the mask, e.g. only on the face, so the face stays close to the original while the rest can change. Draw it with ComfyUI's mask editor (right-click on Load Image → Open in MaskEditor) and connect the MASK output. |

---

## Positive and negative prompt

Use **two** Subtle Edit nodes, like in the included workflows:

- the top one with your instruction → **positive**
- the bottom one with an empty prompt → **negative**

Connect the **same picture(s)** and use the **same settings** on both. Otherwise the result can look wrong.

(With Krea 2 Turbo at CFG 1 the negative is not used, but it does not hurt to keep it.)

---

## Examples

**Normal edit (change color, clothes, background, lighting)**
Everything on default. Only set model, edit_format and encode_res.

**The face changes too much**
Raise ref_boost a bit (e.g. 1.3), optionally with a boost_mask on the face.

**The edit is too weak, the picture hardly changes**
Lower ref_boost below 1.0 or set it back to 1.0. With an edit LoRA, also check the LoRA strength in the LoRA loader.

**Put a person into a new scene (Krea 2)**
image_1 = the scene, image_2 = the person. ref_position = aligned, ref_position_2 = beside.

---

## Troubleshooting

- **Error "edit_format ... is for krea2, but model is set to moss"**: model and edit_format do not fit together. Change one of them.
- **Results look blurry or shifted**: make sure the output size is the same as your input picture (the included workflows do this automatically).
- **A message starting with `[Subtle-Edit]` in the console** saying something is "disabled" or "ignored": a ComfyUI update changed something the node needs. The basic editing still works; only ref_boost, boost_mask, ref_position or ref_strength (Krea 2) may not. Please report it.
- **Old workflows show wrong values after an update**: delete the Subtle Edit nodes and add them again.

---

## Credits

- The Krea 2 mode follows [comfyui-krea2edit](https://github.com/lbouaraba/comfyui-krea2edit) by Conrad Locke and gives the same results as his nodes (when the output has the same size as image_1).
- MOSS-Image by OpenMOSS, Krea 2 by Krea, AI-Toolkit by ostris.
