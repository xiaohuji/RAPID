import os
import h5py
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk

from monai.transforms import Compose, ScaleIntensityRangePercentilesd, Lambdad


class RCCDataset3DINOBox(Dataset):
    """
    输出:
      feat: FloatTensor, shape = (1, 112, 112, 64)   # (C,H,W,D) -> 3DINO 需要
      label: LongTensor
    """
    def __init__(self, args, pid_col_name, pid, feature_path_pd, feature_root,
                 box_hw=112, box_d=64,
                 foreground=0,
                 empty_mask_policy="error"):
        super().__init__()
        self.args = args
        self.pid = pid
        self.pid_col_name = pid_col_name
        self.feature_path_pd = feature_path_pd
        self.feature_path_pd[pid_col_name] = self.feature_path_pd[pid_col_name].astype(str)
        self.feature_root = feature_root

        self.box_hw = int(box_hw)   # 112
        self.box_d = int(box_d)     # 64
        self.foreground = foreground
        self.empty_mask_policy = empty_mask_policy

        # ✅ 3DINO 仓库同款预处理：NaN->mean + 分位数缩放到[-1,1]并clip
        # train3d.py: lower=0.05, upper=99.95, b_min=-1, b_max=1, clip=True :contentReference[oaicite:3]{index=3}
        self.preproc = Compose([
            Lambdad(keys=["image"], func=lambda x: torch.nan_to_num(x, torch.nanmean(x).item())),
            ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=0.05, upper=99.95,
                b_min=-1.0, b_max=1.0,
                clip=True
            ),
        ])

    def __len__(self):
        return len(self.pid)

    @staticmethod
    def _mask_center_zyx(mask_zyx: np.ndarray, foreground=0):
        coords = np.where(mask_zyx > foreground)
        if coords[0].size == 0:
            return None
        zs, ys, xs = coords
        cz = (int(zs.min()) + int(zs.max())) // 2
        cy = (int(ys.min()) + int(ys.max())) // 2
        cx = (int(xs.min()) + int(xs.max())) // 2
        return cz, cy, cx

    @staticmethod
    def _crop_or_pad_3d_zyx_torch(vol_zyx: torch.Tensor,
                                 cz: int, cy: int, cx: int,
                                 out_d: int, out_h: int, out_w: int,
                                 pad_value: float = -1.0) -> torch.Tensor:
        """
        vol_zyx: (Z,Y,X)
        return:  (out_d,out_h,out_w) in ZYX order
        """
        Z, Y, X = vol_zyx.shape
        hz, hy, hx = out_d // 2, out_h // 2, out_w // 2

        z0, z1 = cz - hz, cz - hz + out_d
        y0, y1 = cy - hy, cy - hy + out_h
        x0, x1 = cx - hx, cx - hx + out_w

        out = torch.full((out_d, out_h, out_w), pad_value, dtype=vol_zyx.dtype)

        src_z0, src_z1 = max(0, z0), min(Z, z1)
        src_y0, src_y1 = max(0, y0), min(Y, y1)
        src_x0, src_x1 = max(0, x0), min(X, x1)

        dst_z0, dst_z1 = src_z0 - z0, (src_z0 - z0) + (src_z1 - src_z0)
        dst_y0, dst_y1 = src_y0 - y0, (src_y0 - y0) + (src_y1 - src_y0)
        dst_x0, dst_x1 = src_x0 - x0, (src_x0 - x0) + (src_x1 - src_x0)

        out[dst_z0:dst_z1, dst_y0:dst_y1, dst_x0:dst_x1] = \
            vol_zyx[src_z0:src_z1, src_y0:src_y1, src_x0:src_x1]
        return out

    def __getitem__(self, index):
        pid = self.pid[index]
        feat_name = self.feature_path_pd['img_name'].values[index]

        img_path  = os.path.join(self.feature_root, feat_name, 'image.nii.gz')
        mask_path = os.path.join(self.feature_root, feat_name, 'mask.nii.gz')

        img  = sitk.ReadImage(img_path)
        mask = sitk.ReadImage(mask_path)

        img_zyx  = sitk.GetArrayFromImage(img).astype(np.float32)   # (Z,Y,X)
        mask_zyx = sitk.GetArrayFromImage(mask)                     # (Z,Y,X)

        center = self._mask_center_zyx(mask_zyx, foreground=self.foreground)
        if center is None:
            if self.empty_mask_policy == "all":
                Z, Y, X = img_zyx.shape
                center = (Z // 2, Y // 2, X // 2)
            else:
                raise ValueError(f"Empty mask for pid={pid}, feat_name={feat_name}")

        cz, cy, cx = center

        # 1) 先做 3DINO 同款强度预处理（在“整幅图”上算分位数，更贴近原仓库流程）:contentReference[oaicite:4]{index=4}
        img_t = torch.from_numpy(img_zyx).unsqueeze(0)  # (1,Z,Y,X) 作为 channel-first
        d = self.preproc({"image": img_t})
        vol_zyx = d["image"].squeeze(0)                 # (Z,Y,X)，已是 [-1,1]

        # 2) 肿瘤中心裁剪 112×112×64；pad 用 -1（和 3DINO 下界一致）:contentReference[oaicite:5]{index=5}
        box_zyx = self._crop_or_pad_3d_zyx_torch(
            vol_zyx, cz, cy, cx,
            out_d=self.box_d, out_h=self.box_hw, out_w=self.box_hw,
            pad_value=-1.0
        )  # (64,112,112)

        # 3) 转成 3DINO 输入顺序 (C,H,W,D)；PatchEmbed3d 期望 (B,C,H,W,D) :contentReference[oaicite:6]{index=6}
        feat = box_zyx.permute(1, 2, 0).contiguous().unsqueeze(0)  # (1,112,112,64)

        label = torch.tensor(self.feature_path_pd['label'].values[index], dtype=torch.long)
        return {"pid": pid, "feat": feat, "label": label}


class RCCDataset3DINOFeat(Dataset):
    def __init__(self, args, pid_col_name, pid, feature_path_pd, feature_root):
        super().__init__()
        self.args = args
        self.pid = [str(x) for x in pid]
        self.pid_col_name = pid_col_name
        self.feature_path_pd = feature_path_pd.copy()
        self.feature_path_pd[pid_col_name] = self.feature_path_pd[pid_col_name].astype(str)
        self.feature_root = feature_root

        self.label_dict = dict(
            zip(
                self.feature_path_pd[pid_col_name].tolist(),
                self.feature_path_pd["label"].tolist()
            )
        )

    def __len__(self):
        return len(self.pid)

    def __getitem__(self, index):
        pid = self.pid[index]
        pkl_path = os.path.join(self.feature_root, f"{pid}.pkl")

        with open(pkl_path, "rb") as f:
            embed = pickle.load(f)

        feat = embed
        label = self.label_dict[pid]

        return {"pid": pid, "feat": feat, "label": label}


class RCCDatasetWSIMROnline(Dataset):
    def __init__(
        self,
        args,
        pid_col_name,
        pid,
        feature_path_pd,
        wsi_feature_root,
        img_feature_root,
        box_hw=112,
        box_d=64,
        model_name='uni'
    ):
        super().__init__()
        self.args = args
        self.pid = [str(x) for x in pid]
        self.pid_col_name = pid_col_name
        self.feature_path_pd = feature_path_pd.copy().reset_index(drop=True)
        self.feature_path_pd[pid_col_name] = self.feature_path_pd[pid_col_name].astype(str)

        self.wsi_feature_root = wsi_feature_root
        self.img_feature_root = img_feature_root
        self.box_hw = box_hw
        self.box_d = box_d
        self.model_name = model_name

        self.preproc = Compose([
            Lambdad(keys=["image"], func=lambda x: torch.nan_to_num(x, torch.nanmean(x).item())),
            ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=0.05, upper=99.95,
                b_min=-1.0, b_max=1.0,
                clip=True
            ),
        ])

    def __len__(self):
        return len(self.pid)

    def _load_wsi_feat(self, index):
        """
        这里直接复用你 RCCDatasetWSITumor 里原本读 wsi_feat 的逻辑
        """
        feat_name = str(self.feature_path_pd['wsi_img_name'].values[index]).split('.')[0]

        wsi_path = os.path.join(
            self.wsi_feature_root,
            f"{feat_name}__{self.model_name}_tiles.h5ad"
        )
        with h5py.File(wsi_path, 'r') as f:
            wsi_feat = f['X'][:]

            # 读取 is_tumor 标记，只保留肿瘤 patch
            if 'obs' in f and 'is_tumor' in f['obs']:
                is_tumor = f['obs']['is_tumor'][:]
                tumor_mask = (is_tumor == 1)

                if tumor_mask.sum() > 0:
                    wsi_feat = wsi_feat[tumor_mask]
                else:
                    # 该 slide 没有肿瘤 patch，保留全部（避免空特征）
                    print(f"[WARN] {feat_name}: no tumor patches, using all patches")

        wsi_feat = torch.tensor(wsi_feat, dtype=torch.float32)
        return wsi_feat

    def _load_img(self, index):
        row = self.feature_path_pd.iloc[index]
        feat_name = row["img_name"]

        img_path = os.path.join(self.img_feature_root, feat_name, "image.nii.gz")
        mask_path = os.path.join(self.img_feature_root, feat_name, "mask.nii.gz")

        img = sitk.ReadImage(img_path)
        mask = sitk.ReadImage(mask_path)

        img_zyx = sitk.GetArrayFromImage(img).astype(np.float32)
        mask_zyx = sitk.GetArrayFromImage(mask)

        center = RCCDataset3DINOBox._mask_center_zyx(mask_zyx, foreground=0)
        if center is None:
            raise ValueError(f"Empty mask for pid={row[self.pid_col_name]}, feat_name={feat_name}")

        cz, cy, cx = center

        # ---------- image 预处理 + 裁剪 ----------
        img_t = torch.from_numpy(img_zyx).unsqueeze(0)
        d = self.preproc({"image": img_t})
        vol_zyx = d["image"].squeeze(0)

        box_zyx = RCCDataset3DINOBox._crop_or_pad_3d_zyx_torch(
            vol_zyx, cz, cy, cx,
            out_d=self.box_d, out_h=self.box_hw, out_w=self.box_hw,
            pad_value=-1.0
        )
        img = box_zyx.permute(1, 2, 0).contiguous().unsqueeze(0)  # (1,112,112,64)

        # ---------- mask 裁剪（同样的中心和尺寸） ----------
        mask_t = torch.from_numpy(mask_zyx.astype(np.float32))  # (D,H,W)
        mask_box_zyx = RCCDataset3DINOBox._crop_or_pad_3d_zyx_torch(
            mask_t, cz, cy, cx,
            out_d=self.box_d, out_h=self.box_hw, out_w=self.box_hw,
            pad_value=0.0  # mask 用 0 填充
        )
        mask = mask_box_zyx.permute(1, 2, 0).contiguous().unsqueeze(0)  # (1,112,112,64)
        mask = (mask > 0.5).float()  # 确保二值化

        return img, mask

    def __getitem__(self, index):
        row = self.feature_path_pd.iloc[index]
        pid = str(row[self.pid_col_name])

        wsi_feat = self._load_wsi_feat(index)
        mr_img, mr_mask = self._load_img(index)
        label = torch.tensor(int(row["label"]), dtype=torch.long)

        return {
            "pid": pid,
            "wsi_feat": wsi_feat,
            "mr_img": mr_img,
            "mr_mask": mr_mask,
            "label": label
        }



class RCCDatasetWSICTOnline(Dataset):
    """
    在线加载 CT 原图（Merlin 预处理）+ WSI 特征，用于可视化。

    CT 预处理流程（与 RCCDatasetMerlin 一致）:
      1) 方向统一 → RAS
      2) 重采样 → (1.5, 1.5, 3.0) mm
      3) HU clip [-1000, 1000] → [0, 1]
      4) 体积中心裁剪 + pad → 224 x 224 x 160

    WSI 特征从 .h5ad 文件读取（只保留肿瘤 patch）。
    """

    def __init__(
        self,
        args,
        pid_col_name,
        pid,
        feature_path_pd,
        wsi_feature_root,
        img_feature_root,        # CT 原图根目录（包含 image.nii.gz）
        out_hw=224,
        out_d=160,
        target_spacing=(1.5, 1.5, 3.0),
        model_name='uni',
    ):
        super().__init__()
        self.args = args
        self.pid = [str(x) for x in pid]
        self.pid_col_name = pid_col_name
        self.feature_path_pd = feature_path_pd.copy().reset_index(drop=True)
        self.feature_path_pd[pid_col_name] = self.feature_path_pd[pid_col_name].astype(str)

        self.wsi_feature_root = wsi_feature_root
        self.img_feature_root = img_feature_root

        self.out_hw = int(out_hw)
        self.out_d = int(out_d)
        self.target_spacing = tuple(float(x) for x in target_spacing)
        self.model_name = model_name

    def __len__(self):
        return len(self.pid)

    # ---------- CT 预处理辅助 ----------

    @staticmethod
    def _reorient_to_ras(img: sitk.Image) -> sitk.Image:
        return sitk.DICOMOrient(img, "RAS")

    @staticmethod
    def _resample_sitk(img: sitk.Image, out_spacing, is_label=False) -> sitk.Image:
        in_spacing = np.array(img.GetSpacing(), dtype=np.float64)
        in_size = np.array(img.GetSize(), dtype=np.int64)
        out_spacing = np.array(out_spacing, dtype=np.float64)
        out_size = np.round(in_size * (in_spacing / out_spacing)).astype(np.int64)
        out_size = np.maximum(out_size, 1)

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(tuple(float(x) for x in out_spacing))
        resampler.SetSize([int(x) for x in out_size])
        resampler.SetOutputDirection(img.GetDirection())
        resampler.SetOutputOrigin(img.GetOrigin())
        resampler.SetTransform(sitk.Transform())

        if is_label:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetDefaultPixelValue(0)
        else:
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(-1000.0)

        return resampler.Execute(img)

    @staticmethod
    def _hu_clip_and_scale_01(vol_zyx: np.ndarray) -> np.ndarray:
        vol_zyx = np.nan_to_num(vol_zyx, nan=-1000.0, posinf=1000.0, neginf=-1000.0).astype(np.float32)
        vol_zyx = np.clip(vol_zyx, -1000.0, 1000.0)
        vol_zyx = (vol_zyx + 1000.0) / 2000.0
        return vol_zyx.astype(np.float32)

    @staticmethod
    def _crop_or_pad_3d_zyx_torch(vol_zyx, out_d, out_h, out_w,
                                   center_zyx=None, pad_value=0.0):
        Z, Y, X = vol_zyx.shape
        if center_zyx is None:
            cz, cy, cx = Z // 2, Y // 2, X // 2
        else:
            cz, cy, cx = center_zyx

        hz, hy, hx = out_d // 2, out_h // 2, out_w // 2
        z0, z1 = cz - hz, cz - hz + out_d
        y0, y1 = cy - hy, cy - hy + out_h
        x0, x1 = cx - hx, cx - hx + out_w

        out = torch.full((out_d, out_h, out_w), pad_value, dtype=vol_zyx.dtype)

        src_z0, src_z1 = max(0, z0), min(Z, z1)
        src_y0, src_y1 = max(0, y0), min(Y, y1)
        src_x0, src_x1 = max(0, x0), min(X, x1)

        dst_z0 = src_z0 - z0
        dst_z1 = dst_z0 + (src_z1 - src_z0)
        dst_y0 = src_y0 - y0
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        dst_x0 = src_x0 - x0
        dst_x1 = dst_x0 + (src_x1 - src_x0)

        out[dst_z0:dst_z1, dst_y0:dst_y1, dst_x0:dst_x1] = \
            vol_zyx[src_z0:src_z1, src_y0:src_y1, src_x0:src_x1]
        return out

    # ---------- 数据加载 ----------

    def _load_wsi_feat(self, index):
        feat_name = str(self.feature_path_pd['wsi_img_name'].values[index]).split('.')[0]
        wsi_path = os.path.join(
            self.wsi_feature_root,
            f"{feat_name}__{self.model_name}_tiles.h5ad"
        )
        with h5py.File(wsi_path, 'r') as f:
            wsi_feat = f['X'][:]
            if 'obs' in f and 'is_tumor' in f['obs']:
                is_tumor = f['obs']['is_tumor'][:]
                tumor_mask = (is_tumor == 1)
                if tumor_mask.sum() > 0:
                    wsi_feat = wsi_feat[tumor_mask]
                else:
                    print(f"[WARN] {feat_name}: no tumor patches, using all patches")
        return torch.tensor(wsi_feat, dtype=torch.float32)

    def _load_ct(self, index):
        row = self.feature_path_pd.iloc[index]
        feat_name = row["img_name"]

        img_path = os.path.join(self.img_feature_root, feat_name, "image.nii.gz")

        # 1) 读图 + RAS
        img = sitk.ReadImage(img_path)
        img = self._reorient_to_ras(img)

        # 2) 重采样
        img = self._resample_sitk(img, self.target_spacing, is_label=False)

        # 3) HU clip → [0,1]
        img_zyx = sitk.GetArrayFromImage(img).astype(np.float32)
        img_zyx = self._hu_clip_and_scale_01(img_zyx)

        vol_zyx = torch.from_numpy(img_zyx)

        # 4) 体积中心裁剪/pad
        vol_zyx = self._crop_or_pad_3d_zyx_torch(
            vol_zyx, self.out_d, self.out_hw, self.out_hw,
            center_zyx=None, pad_value=0.0
        )
        ct_img = vol_zyx.permute(1, 2, 0).contiguous().unsqueeze(0)  # (1, H, W, D)

        return ct_img

    def __getitem__(self, index):
        row = self.feature_path_pd.iloc[index]
        pid = str(row[self.pid_col_name])

        wsi_feat = self._load_wsi_feat(index)
        ct_img = self._load_ct(index)
        label = torch.tensor(int(row["label"]), dtype=torch.long)

        return {
            "pid": pid,
            "wsi_feat": wsi_feat,
            "ct_img": ct_img,       # (1, 224, 224, 160)
            "label": label,
        }