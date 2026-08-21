import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



CONCH_REPO_DIR  = r""
CHECKPOINT_PATH = r""

H5AD_DIR        = Path(r"")
WSI_DIR         = Path(r"")
OUTPUT_DIR      = Path(r"")

TILE_SIZE       = 256
TUMOR_THRESHOLD = 0.5
BATCH_SIZE      = 64
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

PROMPT_TEMPLATES = [
    "a histopathology image of {}.",
    "an H&E stained tissue section showing {}.",
    "a photomicrograph of {}.",
    "a pathology slide depicting {}.",
    "histologic appearance of {}.",
    "a microscopy image of {}.",
]

CLASSNAME_SYNONYMS = {
    "tumor": [
        "renal cell carcinoma",
        "kidney cancer cells",
        "clear cell renal cell carcinoma",
        "malignant renal epithelial neoplasm",
        "renal tumor cells with clear cytoplasm",
        "neoplastic renal cells",
        "papillary renal cell carcinoma",
        "chromophobe renal cell carcinoma",
        "carcinoma of the kidney",
    ],
    "non_tumor": [
        "normal kidney parenchyma",
        "normal renal tubules and glomeruli",
        "benign renal tissue",
        "renal stroma and blood vessels",
        "fibrotic tissue",
        "inflammatory infiltrate",
        "adipose tissue",
        "necrotic tissue",
        "hemorrhage and edema",
    ],
}

def load_conch(repo_dir, ckpt_path, device):
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer

    model, preprocess = create_model_from_pretrained(
        "conch_ViT-B-16", checkpoint_path=ckpt_path,
    )
    model = model.to(device).eval()
    tokenizer = get_tokenizer()

    if hasattr(model, "logit_scale"):
        logit_scale = model.logit_scale.exp().item()
    else:
        logit_scale = 100.0
    print(f"[INFO] CONCH loaded on {device}, logit_scale = {logit_scale:.1f}")

    return model, preprocess, tokenizer, logit_scale


def safe_tokenize(tokenizer, texts: list) -> torch.Tensor:
    tokens = tokenizer(
        texts,
        max_length=127,
        add_special_tokens=True,
        return_token_type_ids=False,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    tokens = F.pad(tokens["input_ids"], (0, 1), value=tokenizer.pad_token_id)
    return tokens


def build_text_embeddings(model, tokenizer, templates, classname_synonyms, device):
    class_embs = {}
    with torch.inference_mode():
        for cls, synonyms in classname_synonyms.items():
            embs = []
            for syn in synonyms:
                for tmpl in templates:
                    tok = safe_tokenize(tokenizer, [tmpl.format(syn)]).to(device)
                    e = model.encode_text(tok)
                    e = F.normalize(e, dim=-1)
                    embs.append(e)
            stacked = torch.cat(embs, dim=0)
            mean = F.normalize(stacked.mean(0, keepdim=True), dim=-1)
            class_embs[cls] = mean.squeeze(0).cpu()
            print(f"  [TEXT] '{cls}': {len(embs)} prompts")
    sim = (class_embs["tumor"] @ class_embs["non_tumor"]).item()
    print(f"  [DIAG] cos(tumor, non_tumor) = {sim:.4f}")
    return class_embs


def encode_patches_from_wsi(model, preprocess, wsi_path, coords,
                            tile_size=256, batch_size=64, device="cuda"):

    try:
        import openslide
        slide = openslide.OpenSlide(str(wsi_path))
    except Exception:
        import tiffslide
        slide = tiffslide.TiffSlide(str(wsi_path))

    all_embs = []

    with torch.inference_mode():
        for i in tqdm(range(0, len(coords), batch_size),
                      desc=f"  Encoding {Path(wsi_path).stem}"):
            batch_coords = coords[i:i + batch_size]
            imgs = []
            for (x, y) in batch_coords:
                region = slide.read_region(
                    (int(x), int(y)), 0, (tile_size, tile_size)
                ).convert("RGB")
                imgs.append(preprocess(region))

            batch = torch.stack(imgs).to(device)

            embs = model.encode_image(batch, proj_contrast=True, normalize=True)
            all_embs.append(embs.cpu())

    try:
        slide.close()
    except Exception:
        pass

    return torch.cat(all_embs, dim=0)  # (N, D)


def zeroshot_classify(image_embs, class_embs, logit_scale):
    text_mat = torch.stack([class_embs["tumor"], class_embs["non_tumor"]])
    logits = image_embs @ text_mat.T
    logits = logits * logit_scale
    probs = torch.softmax(logits, dim=-1)
    return probs[:, 0].numpy()


def get_coords_from_adata(adata):
    if "x" in adata.obs.columns and "y" in adata.obs.columns:
        return adata.obs[["x", "y"]].values.astype(float)
    if adata.obsm is not None and "spatial" in adata.obsm:
        return adata.obsm["spatial"].astype(float)
    raise KeyError("not find：need obs['x','y'] or obsm['spatial']")


def find_wsi_path(h5ad_path, wsi_dir):
    wsi_stem = h5ad_path.stem.split("__")[0]
    candidates = list(wsi_dir.glob(f"{wsi_stem}.*"))
    if not candidates:
        return None
    return candidates[0]


def quick_test_one_slide(h5ad_path, model, preprocess, tokenizer,
                         logit_scale, class_embs, wsi_dir,
                         device="cuda", save_fig=True):
    h5ad_path = Path(h5ad_path)
    print(f"\n{'='*60}")
    print(f"  Quick Test: {h5ad_path.name}")
    print(f"{'='*60}")

    adata = ad.read_h5ad(h5ad_path)
    coords = get_coords_from_adata(adata)
    wsi_path = find_wsi_path(h5ad_path, wsi_dir)
    if wsi_path is None:
        print(f"  [ERROR] can not find WSI")
        return

    print(f"  WSI: {wsi_path.name}, Patches: {len(coords)}")

    image_embs = encode_patches_from_wsi(
        model, preprocess, wsi_path, coords,
        tile_size=TILE_SIZE, batch_size=BATCH_SIZE, device=device,
    )

    tumor_probs = zeroshot_classify(image_embs, class_embs, logit_scale)

    print(f"\n  Probability Distribution:")
    print(f"    min   = {tumor_probs.min():.4f}")
    print(f"    5%    = {np.percentile(tumor_probs, 5):.4f}")
    print(f"    25%   = {np.percentile(tumor_probs, 25):.4f}")
    print(f"    50%   = {np.median(tumor_probs):.4f}")
    print(f"    75%   = {np.percentile(tumor_probs, 75):.4f}")
    print(f"    95%   = {np.percentile(tumor_probs, 95):.4f}")
    print(f"    max   = {tumor_probs.max():.4f}")
    print(f"    mean  = {tumor_probs.mean():.4f}")
    print(f"    std   = {tumor_probs.std():.4f}")

    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        n = (tumor_probs >= thr).sum()
        print(f"    threshold={thr:.1f} → {n}/{len(tumor_probs)} tumor ({n/len(tumor_probs):.1%})")

    if save_fig:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(tumor_probs, bins=80, color="#4C72B0", edgecolor="white", alpha=0.85)
        ax.axvline(0.5, color="red", ls="--", lw=1.5, label="threshold=0.5")
        ax.set_xlabel("P(tumor)", fontsize=12)
        ax.set_ylabel("# Patches", fontsize=12)
        ax.set_title(f"{h5ad_path.stem.split('__')[0]} — Tumor Probability", fontsize=13)
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        fig_path = OUTPUT_DIR / f"{h5ad_path.stem.split('__')[0]}__prob_hist.png"
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)
        print(f"\n  Histogram saved → {fig_path}")

    return tumor_probs, adata


def process_all(model, preprocess, class_embs, logit_scale, device="cuda"):
    h5ad_files = sorted(H5AD_DIR.glob("*.h5ad"))
    print(f"\n[INFO] Found {len(h5ad_files)} h5ad files")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for h5ad_path in h5ad_files:
        try:
            adata = ad.read_h5ad(h5ad_path)
            if adata.n_obs == 0:
                print(f"\n  [SKIP] {h5ad_path.name}: empty")
                continue

            coords = get_coords_from_adata(adata)
            wsi_path = find_wsi_path(h5ad_path, WSI_DIR)
            if wsi_path is None:
                print(f"\n  [ERROR] {h5ad_path.name}: WSI not found")
                continue

            print(f"\n--- {h5ad_path.name} ({adata.n_obs} patches, WSI: {wsi_path.name}) ---")

            image_embs = encode_patches_from_wsi(
                model, preprocess, wsi_path, coords,
                tile_size=TILE_SIZE, batch_size=BATCH_SIZE, device=device,
            )

            tumor_probs = zeroshot_classify(image_embs, class_embs, logit_scale)
            tumor_labels = (tumor_probs >= TUMOR_THRESHOLD).astype(int)

            adata.obs["tumor_prob"] = tumor_probs
            adata.obs["is_tumor"]   = tumor_labels

            n_tumor = tumor_labels.sum()
            n_total = len(tumor_labels)
            print(f"  → {n_tumor}/{n_total} tumor ({n_tumor/n_total:.1%}), "
                  f"prob range [{tumor_probs.min():.3f}, {tumor_probs.max():.3f}]")

            out_path = OUTPUT_DIR / h5ad_path.name
            adata.write_h5ad(out_path)

            all_results.append({
                "slide": h5ad_path.stem.split("__")[0],
                "h5ad_file": h5ad_path.name,
                "n_total": n_total,
                "n_tumor": int(n_tumor),
                "tumor_fraction": n_tumor / n_total if n_total else 0,
                "mean_prob": tumor_probs.mean(),
                "median_prob": np.median(tumor_probs),
                "min_prob": tumor_probs.min(),
                "max_prob": tumor_probs.max(),
            })

        except Exception as e:
            print(f"  [ERROR] {h5ad_path.name}: {e}")
            import traceback
            traceback.print_exc()

    if all_results:
        df = pd.DataFrame(all_results)
        report = OUTPUT_DIR / "zeroshot_tumor_summary.csv"
        df.to_csv(report, index=False)
        print(f"\n{'='*60}")
        print(f"Summary → {report}")
        print(f"Slides: {len(df)}, "
              f"Patches: {df['n_total'].sum()}, "
              f"Tumor: {df['n_tumor'].sum()} "
              f"({df['n_tumor'].sum()/df['n_total'].sum():.1%})")


def main():
    print("=" * 70)
    print("  CONCH Zero-Shot RCC Tumor Patch Filtering")
    print("  (encode from WSI + logit_scale)")
    print("=" * 70)

    model, preprocess, tokenizer, logit_scale = load_conch(
        CONCH_REPO_DIR, CHECKPOINT_PATH, DEVICE
    )

    print("\n[Step 1] Building text embeddings...")
    class_embs = build_text_embeddings(
        model, tokenizer, PROMPT_TEMPLATES, CLASSNAME_SYNONYMS, DEVICE
    )

    print("\n[Step 2] Quick test on first slide...")
    h5ad_files = sorted(H5AD_DIR.glob("*.h5ad"))
    if h5ad_files:
        quick_test_one_slide(
            h5ad_files[0], model, preprocess, tokenizer,
            logit_scale, class_embs, WSI_DIR, device=DEVICE,
        )

    print("\n[Step 3] Processing all slides...")
    process_all(model, preprocess, class_embs, logit_scale, DEVICE)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
