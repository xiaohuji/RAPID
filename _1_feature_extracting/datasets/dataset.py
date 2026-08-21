import os
import numpy as np
import torch
import SimpleITK as sitk
from torch.utils.data import Dataset
import pickle

# class RCCDataset(Dataset):
#     def __init__(self, args, pid_col_name, pid, feature_path_pd, feature_root,
#                  patch_size=224, foreground=0,
#                  ct_window=(-125, 225),
#                  slice_stride=1,
#                  max_slices=None,
#                  empty_mask_policy="error",   # "error" / "all"
#                  out_channels=1               # ✅ 1(推荐) 或 3(复制成RGB)
#                  ):
#         super().__init__()
#         self.args = args
#         self.pid = pid
#         self.pid_col_name = pid_col_name
#         self.feature_path_pd = feature_path_pd
#         self.feature_path_pd[pid_col_name] = self.feature_path_pd[pid_col_name].astype(str)
#         self.feature_root = feature_root
#
#         self.patch_size = int(patch_size)
#         self.foreground = foreground
#         self.ct_window = ct_window
#         self.slice_stride = max(1, int(slice_stride))
#         self.max_slices = max_slices
#         self.empty_mask_policy = empty_mask_policy
#         self.out_channels = int(out_channels)
#
#         if self.out_channels not in (1, 3):
#             raise ValueError("out_channels must be 1 or 3")
#
#     def __len__(self):
#         return len(self.pid)
#
#     def _normalize_slice(self, arr2d: np.ndarray) -> torch.Tensor:
#         """(H,W) numpy -> (C,H,W) torch.float32 in [0,1]"""
#         w_min, w_max = self.ct_window
#         arr2d = np.clip(arr2d, w_min, w_max)
#         arr2d = (arr2d - w_min) / (w_max - w_min + 1e-8)
#         t = torch.from_numpy(arr2d).float().unsqueeze(0)  # (1,H,W)
#         if self.out_channels == 3:
#             t = t.repeat(3, 1, 1)  # (3,H,W) 伪RGB
#         return t
#
#     @staticmethod
#     def _crop_or_pad_2d(arr2d: np.ndarray, cx: int, cy: int, patch: int) -> np.ndarray:
#         H, W = arr2d.shape
#         half = patch // 2
#         x0 = cx - half
#         x1 = x0 + patch
#         y0 = cy - half
#         y1 = y0 + patch
#
#         out = np.zeros((patch, patch), dtype=arr2d.dtype)
#
#         src_x0 = max(0, x0); src_x1 = min(W, x1)
#         src_y0 = max(0, y0); src_y1 = min(H, y1)
#
#         dst_x0 = src_x0 - x0; dst_x1 = dst_x0 + (src_x1 - src_x0)
#         dst_y0 = src_y0 - y0; dst_y1 = dst_y0 + (src_y1 - src_y0)
#
#         out[dst_y0:dst_y1, dst_x0:dst_x1] = arr2d[src_y0:src_y1, src_x0:src_x1]
#         return out
#
#     @staticmethod
#     def _mask_center_xy(mask_zyx: np.ndarray, foreground=0):
#         """mask_zyx: (S,H,W) -> 返回 (cx,cy)"""
#         coords = np.where(mask_zyx > foreground)  # (z_idx, y_idx, x_idx)
#         if coords[0].size == 0:
#             return None
#         ys = coords[1]
#         xs = coords[2]
#         x_min, x_max = int(xs.min()), int(xs.max())
#         y_min, y_max = int(ys.min()), int(ys.max())
#         cx = (x_min + x_max) // 2
#         cy = (y_min + y_max) // 2
#         return cx, cy
#
#     def __getitem__(self, index):
#         pid = self.pid[index]
#         feat_name = self.feature_path_pd['img_name'].values[index]
#
#         img_path = os.path.join(self.feature_root, feat_name, 'image.nii.gz')
#         mask_path = os.path.join(self.feature_root, feat_name, 'mask.nii.gz')
#
#         img = sitk.ReadImage(img_path)
#         mask = sitk.ReadImage(mask_path)
#
#         img_zyx, mask_zyx, z_range = self.crop_img_to_mask_slices(img, mask, foreground=self.foreground)
#
#         if img_zyx is None:
#             if self.empty_mask_policy == "all":
#                 img_zyx = sitk.GetArrayFromImage(img)
#                 mask_zyx = sitk.GetArrayFromImage(mask)
#                 z_range = (0, img_zyx.shape[0] - 1)
#             else:
#                 raise ValueError(f"Empty mask for pid={pid}, feat_name={feat_name}")
#
#         # z 抽样
#         img_zyx = img_zyx[::self.slice_stride]
#         mask_zyx = mask_zyx[::self.slice_stride] if mask_zyx is not None else None
#
#         # 限制最大slice数（均匀采样）
#         if self.max_slices is not None and img_zyx.shape[0] > self.max_slices:
#             idxs = np.linspace(0, img_zyx.shape[0] - 1, self.max_slices).round().astype(int)
#             img_zyx = img_zyx[idxs]
#             if mask_zyx is not None:
#                 mask_zyx = mask_zyx[idxs]
#
#         if mask_zyx is None:
#             raise ValueError("mask_zyx is None but tumor-centered crop requires mask.")
#         center = self._mask_center_xy(mask_zyx, foreground=self.foreground)
#         if center is None:
#             raise ValueError(f"Mask has no foreground after cropping for pid={pid}, feat_name={feat_name}")
#         cx, cy = center
#
#         # ✅ 一次性得到 224×224×n（表现为 n 张 224×224 的序列）
#         patches = []
#         for z in range(img_zyx.shape[0]):
#             patch2d = self._crop_or_pad_2d(img_zyx[z], cx=cx, cy=cy, patch=self.patch_size)
#             patches.append(self._normalize_slice(patch2d))  # (C,224,224)
#
#         feat = torch.stack(patches, dim=0)  # [n, C, 224, 224]
#
#         label = self.feature_path_pd['label'].values[index]
#         label = torch.tensor(label, dtype=torch.long)
#
#         return {
#             'pid': pid,
#             'feat': feat,          # ✅ [n, C, 224, 224]  n可变
#             'label': label,
#         }
#
#     def crop_img_to_mask_slices(self, img: sitk.Image, mask: sitk.Image, foreground=0):
#         if img.GetSize() != mask.GetSize():
#             raise ValueError(f"Size mismatch: img={img.GetSize()}, mask={mask.GetSize()}")
#
#         mask_arr = sitk.GetArrayFromImage(mask)  # (z,y,x)
#         exists = np.any(mask_arr > foreground, axis=(1, 2))
#         if not np.any(exists):
#             return None, None, None
#
#         z_idx = np.where(exists)[0]
#         z_min, z_max = int(z_idx.min()), int(z_idx.max())
#
#         size = list(img.GetSize())  # (x,y,z)
#         roi_index = [0, 0, z_min]
#         roi_size  = [size[0], size[1], z_max - z_min + 1]
#
#         img_roi = sitk.RegionOfInterest(img, size=roi_size, index=roi_index)
#         mask_roi = sitk.RegionOfInterest(mask, size=roi_size, index=roi_index)
#
#         img_zyx = sitk.GetArrayFromImage(img_roi)    # (n,H,W)
#         mask_zyx = sitk.GetArrayFromImage(mask_roi)  # (n,H,W)
#
#         return img_zyx, mask_zyx, (z_min, z_max)
class RCCDataset(Dataset):
    def __init__(self, args, pid_col_name, pid, feature_path_pd, feature_root,
                 patch_size=224, foreground=0,
                 ct_window=(-125, 225),
                 slice_stride=1,
                 max_slices=None,
                 empty_mask_policy="error",
                 out_channels=3,              # ✅ 固定用3通道喂预训练resnet
                 use_imagenet_norm=True):     # ✅ 预训练resnet建议 True
        super().__init__()
        self.args = args
        self.pid = pid
        self.pid_col_name = pid_col_name
        self.feature_path_pd = feature_path_pd
        self.feature_path_pd[pid_col_name] = self.feature_path_pd[pid_col_name].astype(str)
        self.feature_root = feature_root

        self.patch_size = int(patch_size)
        self.foreground = foreground
        self.ct_window = ct_window
        self.slice_stride = max(1, int(slice_stride))
        self.max_slices = max_slices
        self.empty_mask_policy = empty_mask_policy

        self.out_channels = int(out_channels)
        if self.out_channels != 3:
            raise ValueError("For pretrained ResNet RGB input, please set out_channels=3")

        self.use_imagenet_norm = bool(use_imagenet_norm)

        # ImageNet mean/std（torchvision 预训练模型默认）
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.imagenet_std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.pid)

    def _normalize_slice(self, arr2d: np.ndarray) -> torch.Tensor:
        """
        (H,W) -> (3,H,W)
        1) CT窗裁剪 -> [0,1]
        2) 灰度复制成RGB: (3,H,W)
        3) 可选 ImageNet mean/std 标准化
        """
        w_min, w_max = self.ct_window
        arr2d = np.clip(arr2d, w_min, w_max)
        arr2d = (arr2d - w_min) / (w_max - w_min + 1e-8)   # -> [0,1]

        t = torch.from_numpy(arr2d).float().unsqueeze(0)    # (1,H,W)
        t = t.repeat(3, 1, 1)                               # (3,H,W)  等价 convert('RGB')

        if self.use_imagenet_norm:
            # 注意：mean/std tensor 在CPU，后续to(device)时会一起搬过去
            t = (t - self.imagenet_mean) / self.imagenet_std

        return t

    @staticmethod
    def _crop_or_pad_2d(arr2d: np.ndarray, cx: int, cy: int, patch: int) -> np.ndarray:
        H, W = arr2d.shape
        half = patch // 2
        x0 = cx - half; x1 = x0 + patch
        y0 = cy - half; y1 = y0 + patch

        out = np.zeros((patch, patch), dtype=arr2d.dtype)

        src_x0 = max(0, x0); src_x1 = min(W, x1)
        src_y0 = max(0, y0); src_y1 = min(H, y1)

        dst_x0 = src_x0 - x0; dst_x1 = dst_x0 + (src_x1 - src_x0)
        dst_y0 = src_y0 - y0; dst_y1 = dst_y0 + (src_y1 - src_y0)

        out[dst_y0:dst_y1, dst_x0:dst_x1] = arr2d[src_y0:src_y1, src_x0:src_x1]
        return out

    @staticmethod
    def _mask_center_xy(mask_zyx: np.ndarray, foreground=0):
        coords = np.where(mask_zyx > foreground)
        if coords[0].size == 0:
            return None
        ys = coords[1]; xs = coords[2]
        cx = (int(xs.min()) + int(xs.max())) // 2
        cy = (int(ys.min()) + int(ys.max())) // 2
        return cx, cy

    def __getitem__(self, index):
        pid = self.pid[index]
        feat_name = self.feature_path_pd['img_name'].values[index]

        img_path = os.path.join(self.feature_root, feat_name, 'image.nii.gz')
        mask_path = os.path.join(self.feature_root, feat_name, 'mask.nii.gz')

        img = sitk.ReadImage(img_path)
        mask = sitk.ReadImage(mask_path)

        img_zyx, mask_zyx, _ = self.crop_img_to_mask_slices(img, mask, foreground=self.foreground)

        if img_zyx is None:
            if self.empty_mask_policy == "all":
                img_zyx = sitk.GetArrayFromImage(img)
                mask_zyx = sitk.GetArrayFromImage(mask)
            else:
                raise ValueError(f"Empty mask for pid={pid}, feat_name={feat_name}")

        img_zyx = img_zyx[::self.slice_stride]
        mask_zyx = mask_zyx[::self.slice_stride] if mask_zyx is not None else None

        if self.max_slices is not None and img_zyx.shape[0] > self.max_slices:
            idxs = np.linspace(0, img_zyx.shape[0] - 1, self.max_slices).round().astype(int)
            img_zyx = img_zyx[idxs]
            if mask_zyx is not None:
                mask_zyx = mask_zyx[idxs]

        if mask_zyx is None:
            raise ValueError("mask_zyx is None but tumor-centered crop requires mask.")
        center = self._mask_center_xy(mask_zyx, foreground=self.foreground)
        if center is None:
            raise ValueError(f"Mask has no foreground after cropping for pid={pid}, feat_name={feat_name}")
        cx, cy = center

        patches = []
        for z in range(img_zyx.shape[0]):
            patch2d = self._crop_or_pad_2d(img_zyx[z], cx=cx, cy=cy, patch=self.patch_size)
            patches.append(self._normalize_slice(patch2d))   # (3,224,224)

        feat = torch.stack(patches, dim=0)  # ✅ [n, 3, 224, 224]

        label = self.feature_path_pd['label'].values[index]
        label = torch.tensor(label, dtype=torch.long)

        return {"pid": pid, "feat": feat, "label": label}

    def crop_img_to_mask_slices(self, img: sitk.Image, mask: sitk.Image, foreground=0):
        if img.GetSize() != mask.GetSize():
            raise ValueError(f"Size mismatch: img={img.GetSize()}, mask={mask.GetSize()}")

        mask_arr = sitk.GetArrayFromImage(mask)  # (z,y,x)
        exists = np.any(mask_arr > foreground, axis=(1, 2))
        if not np.any(exists):
            return None, None, None

        z_idx = np.where(exists)[0]
        z_min, z_max = int(z_idx.min()), int(z_idx.max())

        size = list(img.GetSize())  # (x,y,z)
        roi_index = [0, 0, z_min]
        roi_size  = [size[0], size[1], z_max - z_min + 1]

        img_roi = sitk.RegionOfInterest(img, size=roi_size, index=roi_index)
        mask_roi = sitk.RegionOfInterest(mask, size=roi_size, index=roi_index)

        img_zyx = sitk.GetArrayFromImage(img_roi)
        mask_zyx = sitk.GetArrayFromImage(mask_roi)
        return img_zyx, mask_zyx, (z_min, z_max)



import os
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
    def __init__(self, pid_col_name, pid, feature_path_pd, feature_root,
                 box_hw=112, box_d=64,
                 foreground=0,
                 empty_mask_policy="error"):
        super().__init__()
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