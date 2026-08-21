import os
from pathlib import Path

import numpy as np
import lazyslide as zs
import pandas as pd
from wsidata import open_wsi
from huggingface_hub import login
import matplotlib.pyplot as plt

UNI_WEIGHTS = r"D:\pyfile\checkpoint\pytorch_model_UNI.bin"

# 2) WSI 路径与输出路径
WSI_DIR = Path(r"")
OUT_DIR = Path(r"")
OUT_THUMBNAIL_DIR = Path(r"_thumbnail")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 3) patch & 模型参数
TILE_SIZE = 256
TILE_MPP  = 0.5
TILE_KEY  = "tiles"
MODEL_NAME = "uni"


def save_patching_thumbnail(wsi, slide_stem, out_dir, tile_key="tiles"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zs.pl.tissue(wsi)
    fig = plt.gcf()
    out_path = out_dir / f"{slide_stem}__tiles_overview.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved tiling overview to: {out_path}")

def process_one_slide(
    slide_path: Path,
    tile_size: int = TILE_SIZE,
    mpp: float = TILE_MPP,
    model_name: str = MODEL_NAME,
    tile_key: str = TILE_KEY,
):
    print(f"\n=== Processing {slide_path} ===")

    wsi = open_wsi(os.path.join(WSI_DIR, slide_path))
    print(wsi)

    zs.pp.find_tissues(wsi)

    zs.pp.tile_tissues(wsi, tile_size, mpp=mpp, key_added=tile_key)
    print(f"Tiling done. Tiles table: {wsi[tile_key].shape[0]} patches")

    save_patching_thumbnail(wsi, slide_path.split('.')[0], OUT_THUMBNAIL_DIR)

    features = zs.tl.feature_extraction(
        wsi,
        model=model_name,
        model_path=UNI_WEIGHTS,
        tile_key=tile_key,
        key_added=f"{model_name}_{tile_key}",
        batch_size=128,
        num_workers=0,
        device="cuda",
        amp=False,
        return_features=True,
    )


    print(f"Features shape from {model_name}: {features.shape}")


    adata = wsi.fetch.features_anndata(model_name)
    h5ad_out = OUT_DIR / f"{slide_path.split('.')[0]}__{model_name}_{tile_key}.h5ad"
    adata.write_h5ad(h5ad_out)
    print(f"Saved tile features (AnnData) to: {h5ad_out}")



if __name__ == "__main__":
    slides = [p for p in os.listdir(r'')]

    print(f"Found {len(slides)} slides in {WSI_DIR}")
    for slide_path in slides:
        process_one_slide(slide_path)
