import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

MODELCONFIGS = {
    "qwen2.5vl": {"model_id": "Qwen/Qwen2.5-VL-7B-Instruct", "processor_id": "Qwen/Qwen2.5-VL-7B-Instruct"},
    "qwen3.5": {"model_id": "Qwen/Qwen3.5-9B", "processor_id": "Qwen/Qwen3.5-9B"},
    "llava": {"model_id": "llava-hf/llava-1.5-7b-hf", "processor_id": "llava-hf/llava-1.5-7b-hf"},
    "idefics2": {"model_id": "HuggingFaceM4/idefics2-8b", "processor_id": "HuggingFaceM4/idefics2-8b"},
}

QWEN_FAMILIES = {"qwen2.5vl", "qwen3.5"}


def parse_args():
    p = argparse.ArgumentParser("Extract pre/post vectors for multiple VLM families")
    p.add_argument("--root", type=str, required=True, help="folder containing images or subfolders of images")
    p.add_argument("--out_dir", type=str, default="data/vector")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_items", type=int, default=0, help="0 = no limit")
    p.add_argument("--save_every", type=int, default=400, help="flush shard every N samples")
    p.add_argument("--models", type=str, default="all", help="comma-separated model families or 'all'")
    p.add_argument("--skip_existing", action="store_true", help="skip outputs that already have manifest.pt")

    p.add_argument("--prompt", type=str, default="What do you see in this image?")
    p.add_argument("--img_size", type=int, default=336, help="resize images to square")
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--no_caption", action="store_true", help="skip caption generation and save empty caps")

    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    return p.parse_args()


class FolderImageDataset(Dataset):
    def __init__(self, root: str):
        self.root = Path(root)
        paths = sorted(self.root.glob("*.jpg")) + sorted(self.root.glob("*.png"))
        self.image_paths = sorted(paths)
        if not self.image_paths:
            raise FileNotFoundError(f"No .jpg or .png found in {self.root}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        p = self.image_paths[idx]
        key = p.stem
        try:
            img = Image.open(p).convert("RGB")
            return {"key": key, "image": img}
        except Exception:
            return None


def collate_keep_pil(batch: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    batch = [b for b in batch if b is not None]
    return {
        "keys": [b["key"] for b in batch],
        "images": [b["image"] for b in batch],
    }


def _dtype_from_str(s: str):
    if s == "bf16":
        return torch.bfloat16
    if s == "fp16":
        return torch.float16
    return torch.float32


def _from_pretrained_with_dtype(model_cls, model_id: str, dtype, trust_remote_code: bool = True):
    # Newer transformers prefers dtype=..., some versions still require torch_dtype=...
    try:
        return model_cls.from_pretrained(model_id, dtype=dtype, trust_remote_code=trust_remote_code)
    except TypeError:
        return model_cls.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=trust_remote_code)


def load_model_and_processor(args,model_family: str, hf_model: str, processor_id: str, dtype, device: str):
    if model_family in QWEN_FAMILIES:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(processor_id, trust_remote_code=True)
        model = _from_pretrained_with_dtype(AutoModelForImageTextToText, hf_model, dtype, trust_remote_code=True)

    elif model_family == "llava":
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        
        
        processor = AutoProcessor.from_pretrained(processor_id, trust_remote_code=True)
        model = _from_pretrained_with_dtype(LlavaForConditionalGeneration, hf_model, dtype, trust_remote_code=True)

    elif model_family == "idefics2":
        from transformers import AutoModelForImageTextToText, AutoProcessor

        
        processor = AutoProcessor.from_pretrained(processor_id, trust_remote_code=True)
        if hasattr(processor, "image_processor"):
            processor.image_processor.size = {"longest_edge": args.img_size, "shortest_edge": args.img_size}
            if hasattr(processor.image_processor, "do_image_splitting"):
                processor.image_processor.do_image_splitting = False
        model = _from_pretrained_with_dtype(AutoModelForImageTextToText, hf_model, dtype, trust_remote_code=True)

    elif model_family == "chameleon":
        from transformers import ChameleonForConditionalGeneration, ChameleonProcessor

        
        processor = ChameleonProcessor.from_pretrained(processor_id, trust_remote_code=True)
        model = _from_pretrained_with_dtype(ChameleonForConditionalGeneration, hf_model, dtype, trust_remote_code=True)

    else:
        raise ValueError(f"Unsupported model_family: {model_family}")

    model = model.to(device)
    model.eval()
    return processor, model


def resolve_hook_module(model, model_family: str):
    mods = dict(model.named_modules())
    if model_family in QWEN_FAMILIES:
        return mods.get("visual.merger") or mods.get("model.visual.merger")
    if model_family == "llava":
        return mods.get("multi_modal_projector") or mods.get("model.multi_modal_projector")
    if model_family == "idefics2":
        return mods.get("model.connector") or mods.get("connector")
    return None


def _resize_images(images: List[Image.Image], img_size: int) -> List[Image.Image]:
    if img_size is not None and img_size > 0:
        return [im.resize((img_size, img_size)) for im in images]
    return images


def _build_inputs_qwen(processor, images: List[Image.Image], prompt: str, img_size: int, device: str):
    from qwen_vl_utils import process_vision_info

    images = _resize_images(images, img_size)
    templates = []
    for im in images:
        templates.append([
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": im},
                    {"type": "text", "text": prompt},
                ],
            }
        ])

    texts = [processor.apply_chat_template(t, tokenize=False, add_generation_prompt=True) for t in templates]
    image_inputs, video_inputs = process_vision_info(templates)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in inputs.items()}


def _build_inputs_llava(processor, images: List[Image.Image], prompt: str, img_size: int, device: str):
    images = _resize_images(images, img_size)
    templates = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        for _ in images
    ]
    prompts = [processor.apply_chat_template(t, add_generation_prompt=True) for t in templates]
    inputs = processor(images=images, text=prompts, padding=True, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


def _build_inputs_idefics2(processor, images: List[Image.Image], prompt: str, img_size: int, device: str):
    images = _resize_images(images, img_size)
    prompts = [prompt + "<image>" for _ in images]
    inputs = processor(text=prompts, images=[[im] for im in images], padding=True, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


def build_inputs(processor, model_family: str, images: List[Image.Image], prompt: str, img_size: int, device: str):
    if model_family in QWEN_FAMILIES:
        return _build_inputs_qwen(processor, images, prompt, img_size, device)
    if model_family == "llava":
        return _build_inputs_llava(processor, images, prompt, img_size, device)
    if model_family == "idefics2":
        return _build_inputs_idefics2(processor, images, prompt, img_size, device)
    raise ValueError(f"Unsupported model_family in batched build_inputs: {model_family}")


def _decode_generated(processor, out_ids, input_ids, no_caption: bool):
    if no_caption:
        bsz = out_ids.size(0)
        return [""] * bsz
    gen_only = out_ids[:, input_ids.shape[1] :]
    captions = processor.batch_decode(gen_only, skip_special_tokens=True)
    return [c.strip() for c in captions]


def _extract_chameleon_batch(model, processor, keys: List[str], images: List[Image.Image], prompt: str, device: str, max_new_tokens: int, no_caption: bool):
    pre_tokens: List[torch.Tensor] = []
    post_tokens: List[torch.Tensor] = []
    captions: List[str] = []

    for im in images:
        inputs = processor(images=im, text=prompt + "<image>", return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            quant, _, _ = model.model.vqmodel.encode(inputs["pixel_values"])
            bpe_tokens = model.model.get_image_tokens(inputs["pixel_values"])
            if no_caption:
                caption = ""
            else:
                out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
                gen_ids = out_ids[:, inputs["input_ids"].shape[1] :]
                caption = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

        pre_tokens.append(quant.detach().float().cpu())
        post_tokens.append(bpe_tokens.detach().float().cpu())
        captions.append(caption)

    return keys, pre_tokens, post_tokens, captions


@torch.inference_mode()
def extract_vectors(
    model,
    processor,
    dataloader,
    model_family: str,
    device: str,
    out_dir: str,
    max_items: Optional[int],
    save_every: int,
    prompt: str,
    img_size: int,
    max_new_tokens: int,
    no_caption: bool,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seen = 0
    shard_count = 0

    shard_keys: List[str] = []
    shard_pre: List[torch.Tensor] = []
    shard_post: List[torch.Tensor] = []
    shard_caps: List[str] = []

    def flush_shard(tag_seen: int):
        nonlocal shard_count, shard_keys, shard_pre, shard_post, shard_caps
        if not shard_keys:
            return
        pre_mat = torch.cat(shard_pre, dim=0)
        post_mat = torch.cat(shard_post, dim=0)
        torch.save({"keys": shard_keys, "vecs": pre_mat, "caps": shard_caps}, out_dir / f"pre_vectors_{tag_seen}.pt")
        torch.save({"keys": shard_keys, "vecs": post_mat, "caps": shard_caps}, out_dir / f"post_vectors_{tag_seen}.pt")
        print(f"[shard] saved {len(shard_keys)} samples at seen={tag_seen} | pre={tuple(pre_mat.shape)} post={tuple(post_mat.shape)}")
        shard_count += 1
        shard_keys = []
        shard_pre = []
        shard_post = []
        shard_caps = []

    for batch in tqdm(dataloader, desc="Extract"):
        keys = batch["keys"]
        images = batch["images"]
        if not keys:
            continue

        if max_items is not None and seen >= max_items:
            break

        if max_items is not None and seen + len(keys) > max_items:
            keep = max_items - seen
            keys = keys[:keep]
            images = images[:keep]

        if model_family == "chameleon":
            b_keys, b_pre_list, b_post_list, b_caps = _extract_chameleon_batch(
                model=model,
                processor=processor,
                keys=keys,
                images=images,
                prompt=prompt,
                device=device,
                max_new_tokens=max_new_tokens,
                no_caption=no_caption,
            )
            for j, k in enumerate(b_keys):
                shard_keys.append(k)
                shard_pre.append(b_pre_list[j])
                shard_post.append(b_post_list[j])
                shard_caps.append(b_caps[j])
                seen += 1

        else:
            inputs = build_inputs(processor, model_family, images, prompt, img_size, device)

            got = {"ok": False}
            pre_buf: List[torch.Tensor] = []
            post_buf: List[torch.Tensor] = []

            def hook_connector(module, inp, out):
                if got["ok"]:
                    return
                x = inp[0]
                y = out
                bsz = len(keys)
                if x.dim() == 2:
                    x = x.view(bsz, -1, x.size(-1))
                if y.dim() == 2:
                    y = y.view(bsz, -1, y.size(-1))
                pre_buf.append(x.detach().float().cpu())
                post_buf.append(y.detach().float().cpu())
                got["ok"] = True

            hook_module = resolve_hook_module(model, model_family)
            if hook_module is None:
                raise RuntimeError(f"Hook module not found for model_family={model_family}")

            h = hook_module.register_forward_hook(hook_connector)
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
            )
            h.remove()

            if not got["ok"]:
                raise RuntimeError(f"Hook did not fire for model_family={model_family}")

            captions = _decode_generated(processor, out_ids, inputs["input_ids"], no_caption=no_caption)
            pre_tokens = pre_buf[0]
            post_tokens = post_buf[0]

            for j, k in enumerate(keys):
                shard_keys.append(k)
                shard_pre.append(pre_tokens[j : j + 1])
                shard_post.append(post_tokens[j : j + 1])
                shard_caps.append(captions[j])
                seen += 1

        if save_every > 0 and len(shard_keys) >= save_every:
            flush_shard(seen)

    flush_shard(seen)
    torch.save({"total_seen": seen, "shards": shard_count, "model_family": model_family}, out_dir / "manifest.pt")
    print(f"Done. total_seen={seen}, shards={shard_count}, out_dir={out_dir}")


def main():
    args = parse_args()
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    dtype = _dtype_from_str(args.dtype)

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root path not found: {root}")

    model_families = list(MODELCONFIGS.keys()) if args.models == "all" else [m.strip() for m in args.models.split(",") if m.strip()]
    invalid = [m for m in model_families if m not in MODELCONFIGS]
    if invalid:
        raise ValueError(f"Unsupported models: {invalid}. Supported: {list(MODELCONFIGS.keys())}")

    image_folders: List[Path] = []
    root_imgs = sorted(root.glob("*.jpg")) + sorted(root.glob("*.png"))
    if root_imgs:
        image_folders.append(root)
    else:
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            imgs = sorted(sub.glob("*.jpg")) + sorted(sub.glob("*.png"))
            if imgs:
                image_folders.append(sub)

    if not image_folders:
        raise FileNotFoundError(f"No image folders with .jpg/.png found under {root}")

    print(f"Found {len(image_folders)} image folder(s) under {root}")
    max_items = None if args.max_items == 0 else args.max_items

    for model_family in model_families:
        hf_model = MODELCONFIGS[model_family]["model_id"]
        processor_id = MODELCONFIGS[model_family]["processor_id"]
        print(f"\nLoading model_family={model_family} model={hf_model}")
        processor, model = load_model_and_processor(args, model_family, hf_model, processor_id, dtype=dtype, device=device)

        for folder in image_folders:
            run_out_dir = Path(args.out_dir) / model_family / folder.name
            manifest = run_out_dir / "manifest.pt"
            if args.skip_existing and manifest.exists():
                print(f"Skipping existing run: {run_out_dir}")
                continue

            print(f"Extracting folder={folder} -> out={run_out_dir}")
            ds = FolderImageDataset(str(folder))
            dl = DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_keep_pil,
                pin_memory=(device == "cuda"),
                persistent_workers=(args.num_workers > 0),
            )

            extract_vectors(
                model=model,
                processor=processor,
                dataloader=dl,
                model_family=model_family,
                device=device,
                out_dir=str(run_out_dir),
                save_every=args.save_every,
                max_items=max_items,
                prompt=args.prompt,
                img_size=args.img_size,
                max_new_tokens=args.max_new_tokens,
                no_caption=args.no_caption,
            )


if __name__ == "__main__":
    main()
