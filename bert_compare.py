import os
from pathlib import Path

import pandas as pd
from bert_score import score


def image_to_ref_txt(image_path: str) -> str:
    return str(Path(image_path).with_suffix(".txt"))


def read_reference(txt_path: str) -> str:
    if not os.path.exists(txt_path):
        return ""
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


csv_path = "caption_compare_out/qwen2.5/caption_original_vs_recon.csv"
out_path = "caption_compare_out/qwen2.5/caption_original_vs_recon_with_bertscore.csv"
# csv_path = "caption_compare_out/llava/caption_original_vs_recon.csv"

# out_path = "caption_compare_out/llava/caption_original_vs_recon_with_bertscore.csv"
out_dir = Path(out_path).parent
if not out_dir.exists():
    out_dir.mkdir(parents=True, exist_ok=True)

# 依你的欄位順序，假設至少有：
# sample_id,key,image_path,caption_original,caption_recon,reproj_mse,reproj_cosine
df = pd.read_csv(csv_path)

# 找 reference txt
df["reference_path"] = df["image_path"].astype(str).apply(image_to_ref_txt)
df["reference_caption"] = df["reference_path"].apply(read_reference)

# 清理空值
for col in ["caption_original", "caption_recon", "reference_caption"]:
    df[col] = df[col].fillna("").astype(str).str.strip()

# 只保留 reference 不為空的資料來算 reference-based metrics
mask_ref = df["reference_caption"] != ""

# print first few column of caption_original, caption_recon, reference_caption for sanity check
print(df[["caption_original", "caption_recon", "reference_caption"]].head())
# 1) recon vs original
P1, R1, F1_1 = score(
    cands=df["caption_recon"].tolist(),
    refs=df["caption_original"].tolist(),
    lang="en",
    batch_size=16,
    rescale_with_baseline=True,
)

df["bs_recon_orig_p"] = P1.cpu().numpy()
df["bs_recon_orig_r"] = R1.cpu().numpy()
df["bs_recon_orig_f1"] = F1_1.cpu().numpy()

# 2) original vs reference
if mask_ref.any():
    P2, R2, F1_2 = score(
        cands=df.loc[mask_ref, "caption_original"].tolist(),
        refs=df.loc[mask_ref, "reference_caption"].tolist(),
        lang="en",
        batch_size=16,
        rescale_with_baseline=True,
    )
    df.loc[mask_ref, "bs_orig_ref_p"] = P2.cpu().numpy()
    df.loc[mask_ref, "bs_orig_ref_r"] = R2.cpu().numpy()
    df.loc[mask_ref, "bs_orig_ref_f1"] = F1_2.cpu().numpy()

# 3) recon vs reference
if mask_ref.any():
    P3, R3, F1_3 = score(
        cands=df.loc[mask_ref, "caption_recon"].tolist(),
        refs=df.loc[mask_ref, "reference_caption"].tolist(),
        lang="en",
        batch_size=16,
        rescale_with_baseline=True,
    )
    df.loc[mask_ref, "bs_recon_ref_p"] = P3.cpu().numpy()
    df.loc[mask_ref, "bs_recon_ref_r"] = R3.cpu().numpy()
    df.loc[mask_ref, "bs_recon_ref_f1"] = F1_3.cpu().numpy()

print("Mean BERTScore F1")
print("recon vs original:", df["bs_recon_orig_f1"].mean())

if mask_ref.any():
    print("original vs reference:", df.loc[mask_ref, "bs_orig_ref_f1"].mean())
    print("recon vs reference:", df.loc[mask_ref, "bs_recon_ref_f1"].mean())
    print(
        "delta (recon_ref - orig_ref):",
        (df.loc[mask_ref, "bs_recon_ref_f1"] - df.loc[mask_ref, "bs_orig_ref_f1"]).mean(),
    )

df.to_csv(out_path, index=False)
print(f"saved to {out_path}")