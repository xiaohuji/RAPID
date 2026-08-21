import os
import pickle
import h5py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def _to_numpy_feature(obj):
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        return obj
    if torch.is_tensor(obj):
        return obj.detach().cpu().numpy()
    if isinstance(obj, (list, tuple)):
        return np.asarray(obj)
    if isinstance(obj, dict):
        for k in ["feat", "feature", "embed", "embedding", "x"]:
            if k in obj:
                return _to_numpy_feature(obj[k])
        for _, v in obj.items():
            arr = _to_numpy_feature(v)
            if arr is not None:
                return arr
    return None


def load_vector_feature(feature_root, pid, feat_path=None, expected_dim=1024):
    candidates = []

    if feat_path is not None and str(feat_path).strip() != "":
        candidates.append(str(feat_path))

    if feature_root is not None and str(feature_root).strip() != "":
        for ext in [".pkl", ".npy", ".pt", ".pth"]:
            candidates.append(os.path.join(feature_root, f"{pid}{ext}"))

    for path in candidates:
        if not os.path.exists(path):
            continue

        if path.endswith(".pkl"):
            with open(path, "rb") as f:
                obj = pickle.load(f)
            arr = _to_numpy_feature(obj)
        elif path.endswith(".npy"):
            arr = np.load(path, allow_pickle=True)
        elif path.endswith(".pt") or path.endswith(".pth"):
            obj = torch.load(path, map_location="cpu")
            arr = _to_numpy_feature(obj)
        else:
            arr = None

        if arr is None:
            continue

        arr = np.asarray(arr).squeeze()
        if arr.ndim != 1:
            arr = arr.reshape(-1)

        if expected_dim is not None and arr.shape[0] != expected_dim:
            raise ValueError(f"Feature dim mismatch for {path}: got {arr.shape[0]}, expected {expected_dim}")

        return torch.tensor(arr, dtype=torch.float32)

    return torch.zeros(expected_dim, dtype=torch.float32)

class RCCDatasetWSIDistill(torch.utils.data.Dataset):
    """
    统一 student 数据集：
    - source_type = 0 : WSI-only
    - source_type = 1 : WSI + CT
    - source_type = 2 : WSI + MRI
    - source_type = 3 : WSI + CT + MRI
    """

    def __init__(
        self,
        args,
        pid_col_name,
        feature_path_pd,
        wsi_feature_root,
        ct_feature_root=None,
        mri_feature_root=None,
        model_name="uni",
    ):
        super().__init__()
        self.args = args
        self.pid_col_name = pid_col_name
        self.df = feature_path_pd.copy().reset_index(drop=True)
        self.df[pid_col_name] = self.df[pid_col_name].astype(str)

        self.wsi_feature_root = wsi_feature_root
        self.ct_feature_root = ct_feature_root
        self.mri_feature_root = mri_feature_root
        self.model_name = model_name

    def __len__(self):
        return len(self.df)

    def _load_wsi_feat(self, row):
        img_name = row["wsi_img_name"] if "wsi_img_name" in row else row["img_name"]
        feat_name = str(img_name).split(".")[0]
        h5ad_path = os.path.join(
            self.wsi_feature_root,
            f"{feat_name}__{self.model_name}_tiles.h5ad"
        )
        with h5py.File(h5ad_path, "r") as f:
            feat = f["X"][:]
        return torch.tensor(feat, dtype=torch.float32)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        pid = str(row[self.pid_col_name])
        label = int(row[self.args.label_col])

        has_ct = int(row.get("has_ct", 0))
        has_mri = int(row.get("has_mri", 0))
        source_type = int(row.get("source_type", 0))

        wsi_feat = self._load_wsi_feat(row)

        ct_feat_path = row.get("ct_feat_path", None)
        mri_feat_path = row.get("mri_feat_path", None)

        if has_ct:
            ct_feat = load_vector_feature(
                self.ct_feature_root,
                pid,
                feat_path=ct_feat_path,
                expected_dim=self.args.ct_pair_dim,
            )
        else:
            ct_feat = torch.zeros(self.args.ct_pair_dim, dtype=torch.float32)

        if has_mri:
            mri_feat = load_vector_feature(
                self.mri_feature_root,
                pid,
                feat_path=mri_feat_path,
                expected_dim=self.args.mri_pair_dim,
            )
        else:
            mri_feat = torch.zeros(self.args.mri_pair_dim, dtype=torch.float32)

        return {
            "pid": pid,
            "wsi_feat": wsi_feat,
            "ct_feat": ct_feat,
            "mri_feat": mri_feat,
            "has_ct": torch.tensor(has_ct, dtype=torch.long),
            "has_mri": torch.tensor(has_mri, dtype=torch.long),
            "source_type": torch.tensor(source_type, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }


class RCCDatasetWSIDistillTumor(torch.utils.data.Dataset):
    """
    统一 student 数据集：
    - source_type = 0 : WSI-only
    - source_type = 1 : WSI + CT
    - source_type = 2 : WSI + MRI
    - source_type = 3 : WSI + CT + MRI
    """

    def __init__(
        self,
        args,
        pid_col_name,
        feature_path_pd,
        wsi_feature_root,
        ct_feature_root=None,
        mri_feature_root=None,
        model_name="uni",
    ):
        super().__init__()
        self.args = args
        self.pid_col_name = pid_col_name
        self.df = feature_path_pd.copy().reset_index(drop=True)
        self.df[pid_col_name] = self.df[pid_col_name].astype(str)

        self.wsi_feature_root = wsi_feature_root
        self.ct_feature_root = ct_feature_root
        self.mri_feature_root = mri_feature_root
        self.model_name = model_name

    def __len__(self):
        return len(self.df)

    def _load_wsi_feat(self, row):
        img_name = row["wsi_img_name"] if "wsi_img_name" in row else row["img_name"]
        feat_name = str(img_name).split(".")[0]
        h5ad_path = os.path.join(
            self.wsi_feature_root,
            f"{feat_name}__{self.model_name}_tiles.h5ad"
        )
        with h5py.File(h5ad_path, 'r') as f:
            feat = f['X'][:]

            # 读取 is_tumor 标记，只保留肿瘤 patch
            if 'obs' in f and 'is_tumor' in f['obs']:
                is_tumor = f['obs']['is_tumor'][:]
                tumor_mask = (is_tumor == 1)

                if tumor_mask.sum() > 0:
                    feat = feat[tumor_mask]
                else:
                    # 该 slide 没有肿瘤 patch，保留全部（避免空特征）
                    print(f"[WARN] {feat_name}: no tumor patches, using all patches")
        return torch.tensor(feat, dtype=torch.float32)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        pid = str(row[self.pid_col_name])
        wsi_img_name = row["wsi_img_name"]
        label = int(row[self.args.label_col])

        has_ct = int(row.get("has_ct", 0))
        has_mri = int(row.get("has_mri", 0))
        source_type = int(row.get("source_type", 0))

        wsi_feat = self._load_wsi_feat(row)

        ct_feat_path = row.get("ct_feat_path", None)
        mri_feat_path = row.get("mri_feat_path", None)

        if has_ct:
            ct_feat = load_vector_feature(
                self.ct_feature_root,
                pid,
                feat_path=ct_feat_path,
                expected_dim=self.args.ct_pair_dim,
            )
        else:
            ct_feat = torch.zeros(self.args.ct_pair_dim, dtype=torch.float32)

        if has_mri:
            mri_feat = load_vector_feature(
                self.mri_feature_root,
                pid,
                feat_path=mri_feat_path,
                expected_dim=self.args.mri_pair_dim,
            )
        else:
            mri_feat = torch.zeros(self.args.mri_pair_dim, dtype=torch.float32)

        return {
            "pid": pid,
            "wsi_img_name": wsi_img_name,
            "wsi_feat": wsi_feat,
            "ct_feat": ct_feat,
            "mri_feat": mri_feat,
            "has_ct": torch.tensor(has_ct, dtype=torch.long),
            "has_mri": torch.tensor(has_mri, dtype=torch.long),
            "source_type": torch.tensor(source_type, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }