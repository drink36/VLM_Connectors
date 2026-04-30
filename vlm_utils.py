"""
Shared VLM utilities used by caption_eval.py and perturb_eval.py.

  - MODEL_IDS          : canonical HuggingFace model IDs
  - numeric_suffix()   : sort key for shard files
  - build_image_index(): stem → path mapping over an image directory
  - load_vlm()         : load processor + model for a given model family
  - build_vlm_inputs() : build tokenizer/processor inputs for captioning
  - decode_outputs()   : strip prompt tokens and decode generated ids
  - resolve_connector(): locate the connector module inside a VLM
"""
import re
from pathlib import Path

import torch
import torch.nn as nn
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Idefics2ForConditionalGeneration,
    LlavaForConditionalGeneration,
)
from transformers.image_utils import load_image

MODEL_IDS = {
    "llava":     "llava-hf/llava-1.5-7b-hf",
    "idefics2":  "HuggingFaceM4/idefics2-8b",
    "qwen2.5vl": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen3.5":   "Qwen/Qwen3.5-9B",
}

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def numeric_suffix(path: Path) -> int:
    """Sort key: extract the trailing integer from a shard filename."""
    match = re.search(r"_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else 10**18


def build_image_index(image_dir: str) -> dict[str, str]:
    """Return {stem: full_path} for every image under image_dir."""
    index: dict[str, str] = {}
    for p in Path(image_dir).rglob("*"):
        if p.suffix.lower() in _IMAGE_EXTS and p.stem not in index:
            index[p.stem] = str(p)
    return index


def load_vlm(model_name: str, img_size: int, device: torch.device):
    """Load processor and model for model_name. Returns (processor, model)."""
    model_id = MODEL_IDS[model_name]
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    if model_name == "idefics2" and hasattr(processor, "image_processor"):
        processor.image_processor.size = {"longest_edge": img_size, "shortest_edge": img_size}
        if hasattr(processor.image_processor, "do_image_splitting"):
            processor.image_processor.do_image_splitting = False

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    if model_name == "llava":
        model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)
    elif model_name == "idefics2":
        model = Idefics2ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True
        )
    return processor, model.to(device).eval()


def build_vlm_inputs(
    processor,
    image_paths: list[str],
    prompt: str,
    model_name: str,
    img_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build tokenized inputs for captioning. Resizes images to img_size if > 0."""
    images = [load_image(p) for p in image_paths]
    if img_size and img_size > 0:
        images = [im.resize((img_size, img_size)) for im in images]

    if model_name in ("llava", "idefics2"):
        template = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(template, add_generation_prompt=True)
        img_arg = images if model_name == "llava" else [[im] for im in images]
        inputs = processor(images=img_arg, text=[text] * len(images), padding=True, return_tensors="pt")
        return {k: v.to(device) for k, v in inputs.items()}

    from qwen_vl_utils import process_vision_info
    templates = [
        [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": prompt}]}]
        for im in images
    ]
    tmpl_kwargs = {"enable_thinking": False} if model_name == "qwen3.5" else {}
    texts = [
        processor.apply_chat_template(t, tokenize=False, add_generation_prompt=True, **tmpl_kwargs)
        for t in templates
    ]
    image_inputs, video_inputs = process_vision_info(templates)
    inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


def decode_outputs(
    processor,
    out_ids: torch.Tensor,
    input_ids: torch.Tensor,
    model_name: str,
) -> list[str]:
    """Strip prompt tokens from generated ids and decode to strings."""
    if model_name in ("qwen2.5vl", "qwen3.5"):
        pad_id = processor.tokenizer.pad_token_id
        results = []
        for i, row in enumerate(input_ids):
            plen = int((row != pad_id).sum().item()) if pad_id is not None else input_ids.shape[1]
            txt = processor.batch_decode(
                out_ids[i, plen:].unsqueeze(0), skip_special_tokens=True
            )[0]
            results.append(txt.strip())
        if model_name == "qwen3.5":
            results = [re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL).strip() for t in results]
        return results

    gen = out_ids[:, input_ids.shape[1]:]
    return [c.strip() for c in processor.batch_decode(gen, skip_special_tokens=True)]


def resolve_connector(model: nn.Module, model_name: str) -> nn.Module:
    """Locate and return the connector/projector module inside a VLM."""
    mods = dict(model.named_modules())
    if model_name == "llava":
        mod = mods.get("multi_modal_projector") or mods.get("model.multi_modal_projector")
    elif model_name == "idefics2":
        mod = mods.get("model.connector") or mods.get("connector")
    else:
        mod = mods.get("visual.merger") or mods.get("model.visual.merger")
    if mod is None:
        raise RuntimeError(f"connector module not found for model_name={model_name!r}")
    return mod
