import os
import shutil
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm

CONCH_DIR = Path(r"")
UNI_DIR   = Path(r"")
OUTPUT_DIR = Path(r"")

CONCH_MODEL = "conch"
UNI_MODEL   = "uni"


def get_slide_stem(h5ad_name, model_name):
    return h5ad_name.replace(f"__{model_name}_tiles.h5ad", "")


def transfer_tumor_labels():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    conch_files = sorted(CONCH_DIR.glob("*.h5ad"))
    print(f"Found {len(conch_files)} CONCH h5ad files")
    print(f"Output → {OUTPUT_DIR}\n")

    success, skip, fail = 0, 0, 0

    for conch_path in tqdm(conch_files, desc="Transferring"):
        slide_stem = get_slide_stem(conch_path.name, CONCH_MODEL)
        uni_src    = UNI_DIR / f"{slide_stem}__{UNI_MODEL}_tiles.h5ad"
        uni_dst    = OUTPUT_DIR / f"{slide_stem}__{UNI_MODEL}_tiles.h5ad"

        if not uni_src.exists():
            print(f"  [SKIP] UNI not found: {uni_src.name}")
            skip += 1
            continue

        try:
            with h5py.File(conch_path, 'r') as fc:
                if 'obs' not in fc or 'is_tumor' not in fc['obs']:
                    print(f"  [SKIP] {conch_path.name}: no is_tumor")
                    skip += 1
                    continue

                conch_is_tumor   = fc['obs']['is_tumor'][:]
                conch_tumor_prob = fc['obs']['tumor_prob'][:] if 'tumor_prob' in fc['obs'] else None
                conch_n = fc['X'].shape[0]

                conch_coords = None
                if 'x' in fc['obs'] and 'y' in fc['obs']:
                    conch_coords = np.stack([fc['obs']['x'][:], fc['obs']['y'][:]], axis=1)

            with h5py.File(uni_src, 'r') as fu:
                uni_n = fu['X'].shape[0]
                uni_coords = None
                if 'obs' in fu and 'x' in fu['obs'] and 'y' in fu['obs']:
                    uni_coords = np.stack([fu['obs']['x'][:], fu['obs']['y'][:]], axis=1)

            shutil.copy2(uni_src, uni_dst)

            if conch_n == uni_n:
                need_coord_match = False
                if conch_coords is not None and uni_coords is not None:
                    if not np.allclose(conch_coords, uni_coords, atol=1.0):
                        need_coord_match = True

                if need_coord_match:
                    is_tumor, tumor_prob = _match_by_coords(
                        conch_coords, uni_coords, conch_is_tumor, conch_tumor_prob
                    )
                else:
                    is_tumor = conch_is_tumor
                    tumor_prob = conch_tumor_prob

            else:
                print(f"  [INFO] {slide_stem}: CONCH={conch_n}, UNI={uni_n}, coord matching")
                if conch_coords is None or uni_coords is None:
                    print(f"  [FAIL] {slide_stem}: no coords")
                    os.remove(uni_dst)
                    fail += 1
                    continue

                is_tumor, tumor_prob = _match_by_coords(
                    conch_coords, uni_coords, conch_is_tumor, conch_tumor_prob
                )

            with h5py.File(uni_dst, 'a') as fu:
                obs = fu['obs']
                for name, data in [('is_tumor', is_tumor), ('tumor_prob', tumor_prob)]:
                    if data is not None:
                        if name in obs:
                            del obs[name]
                        obs.create_dataset(name, data=data)

            n_tumor = (is_tumor == 1).sum()
            success += 1

        except Exception as e:
            print(f"  [FAIL] {slide_stem}: {e}")
            if uni_dst.exists():
                os.remove(uni_dst)
            fail += 1

    print(f"\nDone: {success} success, {skip} skipped, {fail} failed")
    print(f"Output files in: {OUTPUT_DIR}")


def _match_by_coords(conch_coords, uni_coords, conch_is_tumor, conch_tumor_prob):
    conch_lookup = {(round(x), round(y)): i for i, (x, y) in enumerate(conch_coords)}

    is_tumor   = np.zeros(len(uni_coords), dtype=conch_is_tumor.dtype)
    tumor_prob = np.zeros(len(uni_coords), dtype=np.float32)
    matched = 0

    for j, (x, y) in enumerate(uni_coords):
        key = (round(x), round(y))
        if key in conch_lookup:
            ci = conch_lookup[key]
            is_tumor[j] = conch_is_tumor[ci]
            if conch_tumor_prob is not None:
                tumor_prob[j] = conch_tumor_prob[ci]
            matched += 1

    print(f"    Matched {matched}/{len(uni_coords)} patches")
    return is_tumor, tumor_prob if conch_tumor_prob is not None else None


if __name__ == "__main__":
    transfer_tumor_labels()