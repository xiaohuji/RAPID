from pathlib import Path
import anndata as ad
import matplotlib.pyplot as plt


H5AD_DIR = Path(r"")
FIG_DIR  = H5AD_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_spatial_heatmap(
    h5ad_path: Path,
    tile_size: int = 256,
    save: bool = True,
    cmap: str = "RdBu_r",
):
    adata = ad.read_h5ad(h5ad_path)
    if "tumor_prob" not in adata.obs.columns:
        print(f"[SKIP] {h5ad_path.name}: no 'tumor_prob' column")
        return

    if "x" in adata.obs.columns and "y" in adata.obs.columns:
        x = adata.obs["x"].values.astype(float)
        y = adata.obs["y"].values.astype(float)
    elif "spatial" in adata.obsm:
        x = adata.obsm["spatial"][:, 0].astype(float)
        y = adata.obsm["spatial"][:, 1].astype(float)
    else:
        print(f"[SKIP] {h5ad_path.name}: no spatial coordinates found")
        return

    probs = adata.obs["tumor_prob"].values
    slide_name = h5ad_path.stem.split("__")[0]

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(
        x, y,
        c=probs,
        cmap=cmap,
        vmin=0, vmax=1,
        s=2, marker="s",
        edgecolors="none",
        alpha=0.8,
    )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(f"{slide_name} — Zero-Shot Tumor Heatmap", fontsize=13)
    ax.set_xlabel("X (px)", fontsize=11)
    ax.set_ylabel("Y (px)", fontsize=11)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("P(tumor)", fontsize=11)

    plt.tight_layout()
    if save:
        out = FIG_DIR / f"{slide_name}_tumor_heatmap.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  Saved heatmap → {out}")
    plt.close(fig)

def main():
    print("=" * 60)
    print("  Post-Classification Visualization & QC")
    print("=" * 60)

    h5ad_files = sorted(H5AD_DIR.glob("*.h5ad"))
    print(f"Found {len(h5ad_files)} h5ad files\n")

    for f in h5ad_files:
        plot_spatial_heatmap(f)


    print("\n[DONE] All visualizations and subsets generated.")


if __name__ == "__main__":
    main()

