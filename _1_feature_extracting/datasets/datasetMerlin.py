import os
import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import torch.nn.functional as F

class RCCDatasetMerlin(Dataset):
    """
    严格贴近 Merlin 论文的 CT 预处理:
      1) 方向统一为:
         first axis: left -> right
         second axis: posterior -> anterior
         third axis: inferior -> superior
         在 SimpleITK 里可用 RAS 近似实现
      2) 重采样到 (1.5, 1.5, 3.0) mm
      3) HU 裁剪到 [-1000, 1000]，再映射到 [0, 1]
      4) pad + center crop 到 224 x 224 x 160

    输出:
      feat: FloatTensor, shape = (1, 224, 224, 160)   # (C,H,W,D)
      label: LongTensor
    """

    def __init__(
        self,
        args,
        pid_col_name,
        pid,
        feature_path_pd,
        feature_root,
        out_hw=224,
        out_d=160,
        target_spacing=(1.5, 1.5, 3.0),
        crop_mode="volume_center",   # "volume_center" = 严格按论文；"mask_center" = 你的任务定制版
        foreground=0,
        empty_mask_policy="error",
    ):
        super().__init__()
        self.args = args
        self.pid = pid
        self.pid_col_name = pid_col_name
        self.feature_path_pd = feature_path_pd.copy()
        self.feature_path_pd[pid_col_name] = self.feature_path_pd[pid_col_name].astype(str)
        self.feature_root = feature_root

        self.out_hw = int(out_hw)   # 224
        self.out_d = int(out_d)     # 160
        self.target_spacing = tuple(float(x) for x in target_spacing)

        self.crop_mode = crop_mode
        self.foreground = foreground
        self.empty_mask_policy = empty_mask_policy

    def __len__(self):
        return len(self.pid)

    @staticmethod
    def _reorient_to_ras(img: sitk.Image) -> sitk.Image:
        """
        论文要求:
          first axis: left -> right
          second axis: posterior -> anterior
          third axis: inferior -> superior

        用 SimpleITK 时，可将图像统一到 RAS。
        """
        return sitk.DICOMOrient(img, "RAS")

    @staticmethod
    def _resample_sitk(img: sitk.Image, out_spacing, is_label=False) -> sitk.Image:
        in_spacing = np.array(img.GetSpacing(), dtype=np.float64)   # (sx, sy, sz)
        in_size = np.array(img.GetSize(), dtype=np.int64)           # (nx, ny, nz)
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
            # 论文是先 resample，再 HU clip；因此这里用 -1000 作为 CT 默认背景更合理
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(-1000.0)

        return resampler.Execute(img)

    @staticmethod
    def _hu_clip_and_scale_01(vol_zyx: np.ndarray) -> np.ndarray:
        """
        论文要求:
          HU [-1000, 1000] -> [0, 1]
        """
        vol_zyx = np.nan_to_num(
            vol_zyx,
            nan=-1000.0,
            posinf=1000.0,
            neginf=-1000.0
        ).astype(np.float32)

        vol_zyx = np.clip(vol_zyx, -1000.0, 1000.0)
        vol_zyx = (vol_zyx + 1000.0) / 2000.0
        return vol_zyx.astype(np.float32)

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
    def _crop_or_pad_3d_zyx_torch(
        vol_zyx: torch.Tensor,
        out_d: int,
        out_h: int,
        out_w: int,
        center_zyx=None,
        pad_value: float = 0.0,
    ) -> torch.Tensor:
        """
        vol_zyx: (Z, Y, X)
        return:  (out_d, out_h, out_w) in ZYX order
        """
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

    def __getitem__(self, index):
        pid = self.pid[index]
        feat_name = self.feature_path_pd['img_name'].values[index]

        img_path = os.path.join(self.feature_root, feat_name, 'image.nii.gz')
        mask_path = os.path.join(self.feature_root, feat_name, 'mask.nii.gz')

        # -------------------------
        # 1) 读图 + 方向统一
        # -------------------------
        img = sitk.ReadImage(img_path)
        img = self._reorient_to_ras(img)

        # -------------------------
        # 2) 重采样到 1.5 x 1.5 x 3.0 mm
        # -------------------------
        img = self._resample_sitk(
            img,
            out_spacing=self.target_spacing,
            is_label=False
        )

        # -------------------------
        # 3) HU clip 到 [-1000,1000] 并映射到 [0,1]
        # -------------------------
        img_zyx = sitk.GetArrayFromImage(img).astype(np.float32)  # (Z,Y,X)
        img_zyx = self._hu_clip_and_scale_01(img_zyx)
        vol_zyx = torch.from_numpy(img_zyx)  # (Z,Y,X)

        # -------------------------
        # 4) pad + crop 到 224 x 224 x 160
        #    严格论文: 用 volume_center
        #    如果你想保留肿瘤中心，可用 mask_center，但那就不是论文原始设置
        # -------------------------
        center_zyx = None

        if self.crop_mode == "mask_center":
            mask = sitk.ReadImage(mask_path)
            mask = self._reorient_to_ras(mask)
            mask = self._resample_sitk(
                mask,
                out_spacing=self.target_spacing,
                is_label=True
            )
            mask_zyx = sitk.GetArrayFromImage(mask)

            center_zyx = self._mask_center_zyx(mask_zyx, foreground=self.foreground)
            if center_zyx is None:
                if self.empty_mask_policy == "all":
                    Z, Y, X = vol_zyx.shape
                    center_zyx = (Z // 2, Y // 2, X // 2)
                else:
                    raise ValueError(f"Empty mask for pid={pid}, feat_name={feat_name}")

        vol_zyx = self._crop_or_pad_3d_zyx_torch(
            vol_zyx,
            out_d=self.out_d,
            out_h=self.out_hw,
            out_w=self.out_hw,
            center_zyx=center_zyx,   # None -> volume center
            pad_value=0.0            # 论文未明说 pad 常数；这里取归一化后的背景下界
        )  # (160,224,224)

        # -------------------------
        # 5) 转成模型输入顺序 (C,H,W,D)
        # -------------------------
        feat = vol_zyx.permute(1, 2, 0).contiguous().unsqueeze(0)  # (1,224,224,160)

        label = self.feature_path_pd['label'].values[index]


        return {
            "pid": pid,
            "feat": feat,
            "label": label
        }



class RCCDatasetMerlinMaskCenter(Dataset):
    """
    自定义 ROI 版 Merlin 预处理:

      1) 方向统一为:
         first axis: left -> right
         second axis: posterior -> anterior
         third axis: inferior -> superior
         用 SimpleITK 的 RAS 近似实现

      2) 重采样到 (1.5, 1.5, 3.0) mm

      3) HU 裁剪到 [-1000, 1000]，再映射到 [0, 1]

      4) 使用 mask 中心裁出 112 x 112 x 80 的 box
         注意这里 box 的顺序是:
           (H, W, D) = (112, 112, 80)
         对应内部 ZYX 为:
           (D, H, W) = (80, 112, 112)

      5) 再把这个 box resize 到 224 x 224 x 160

    输出:
      feat: FloatTensor, shape = (1, 224, 224, 160)   # (C,H,W,D)
      label: LongTensor
    """

    def __init__(
        self,
        pid_col_name,
        pid,
        feature_path_pd,
        feature_root,
        crop_hw=224,
        crop_d=160,
        out_hw=224,
        out_d=160,
        target_spacing=(1.5, 1.5, 3.0),
        foreground=0,
        empty_mask_policy="error",
    ):
        super().__init__()
        self.pid = pid
        self.pid_col_name = pid_col_name
        self.feature_path_pd = feature_path_pd.copy()
        self.feature_path_pd[pid_col_name] = self.feature_path_pd[pid_col_name].astype(str)
        self.feature_root = feature_root

        self.crop_hw = int(crop_hw)   # 112
        self.crop_d = int(crop_d)     # 80
        self.out_hw = int(out_hw)     # 224
        self.out_d = int(out_d)       # 160

        self.target_spacing = tuple(float(x) for x in target_spacing)
        self.foreground = foreground
        self.empty_mask_policy = empty_mask_policy

    def __len__(self):
        return len(self.pid)

    @staticmethod
    def _reorient_to_ras(img: sitk.Image) -> sitk.Image:
        """
        统一方向:
          first axis: left -> right
          second axis: posterior -> anterior
          third axis: inferior -> superior
        """
        return sitk.DICOMOrient(img, "RAS")

    @staticmethod
    def _resample_sitk(img: sitk.Image, out_spacing, is_label=False) -> sitk.Image:
        """
        按目标 spacing 重采样.
        out_spacing: (sx, sy, sz) = (x, y, z)
        """
        in_spacing = np.array(img.GetSpacing(), dtype=np.float64)   # (sx, sy, sz)
        in_size = np.array(img.GetSize(), dtype=np.int64)           # (nx, ny, nz)
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
        """
        HU [-1000, 1000] -> [0, 1]
        """
        vol_zyx = np.nan_to_num(
            vol_zyx,
            nan=-1000.0,
            posinf=1000.0,
            neginf=-1000.0
        ).astype(np.float32)

        vol_zyx = np.clip(vol_zyx, -1000.0, 1000.0)
        vol_zyx = (vol_zyx + 1000.0) / 2000.0
        return vol_zyx.astype(np.float32)

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
    def _crop_or_pad_3d_zyx_torch(
        vol_zyx: torch.Tensor,
        out_d: int,
        out_h: int,
        out_w: int,
        center_zyx,
        pad_value: float = 0.0,
    ) -> torch.Tensor:
        """
        vol_zyx: (Z, Y, X)
        return:  (out_d, out_h, out_w) in ZYX order
        """
        Z, Y, X = vol_zyx.shape
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

    @staticmethod
    def _resize_zyx_torch(
        vol_zyx: torch.Tensor,
        out_d: int,
        out_h: int,
        out_w: int,
    ) -> torch.Tensor:
        """
        把 (Z,Y,X) resize 到 (out_d,out_h,out_w)
        使用 trilinear，更适合连续 CT 强度
        """
        # [Z,Y,X] -> [N,C,D,H,W]
        x = vol_zyx.unsqueeze(0).unsqueeze(0)   # (1,1,Z,Y,X)
        x = F.interpolate(
            x,
            size=(out_d, out_h, out_w),
            mode="trilinear",
            align_corners=False,
        )
        x = x.squeeze(0).squeeze(0)  # (out_d,out_h,out_w)
        return x

    def __getitem__(self, index):
        pid = self.pid[index]
        feat_name = self.feature_path_pd['img_name'].values[index]

        img_path = os.path.join(self.feature_root, feat_name, 'image.nii.gz')
        mask_path = os.path.join(self.feature_root, feat_name, 'mask.nii.gz')

        # -------------------------
        # 1) 读图 + 方向统一
        # -------------------------
        img = sitk.ReadImage(img_path)
        mask = sitk.ReadImage(mask_path)

        img = self._reorient_to_ras(img)
        mask = self._reorient_to_ras(mask)

        # -------------------------
        # 2) 重采样到 1.5 x 1.5 x 3.0 mm
        # -------------------------
        img = self._resample_sitk(
            img,
            out_spacing=self.target_spacing,
            is_label=False
        )
        mask = self._resample_sitk(
            mask,
            out_spacing=self.target_spacing,
            is_label=True
        )

        # -------------------------
        # 3) HU clip 到 [-1000,1000] 并映射到 [0,1]
        # -------------------------
        img_zyx = sitk.GetArrayFromImage(img).astype(np.float32)   # (Z,Y,X)
        img_zyx = self._hu_clip_and_scale_01(img_zyx)
        vol_zyx = torch.from_numpy(img_zyx)                        # (Z,Y,X)

        mask_zyx = sitk.GetArrayFromImage(mask)
        center_zyx = self._mask_center_zyx(mask_zyx, foreground=self.foreground)

        if center_zyx is None:
            if self.empty_mask_policy == "all":
                Z, Y, X = vol_zyx.shape
                center_zyx = (Z // 2, Y // 2, X // 2)
            else:
                raise ValueError(f"Empty mask for pid={pid}, feat_name={feat_name}")

        # -------------------------
        # 4) 以 mask 中心裁 112 x 112 x 80 的 box
        #    ZYX 内部顺序对应:
        #      (80, 112, 112)
        # -------------------------
        roi_zyx = self._crop_or_pad_3d_zyx_torch(
            vol_zyx,
            out_d=self.crop_d,
            out_h=self.crop_hw,
            out_w=self.crop_hw,
            center_zyx=center_zyx,
            pad_value=0.0
        )  # (80,112,112)

        # -------------------------
        # 5) 再 resize 到 224 x 224 x 160
        #    ZYX: (80,112,112) -> (160,224,224)
        # -------------------------
        roi_zyx = self._resize_zyx_torch(
            roi_zyx,
            out_d=self.out_d,
            out_h=self.out_hw,
            out_w=self.out_hw,
        )  # (160,224,224)

        # -------------------------
        # 6) 转成模型输入顺序 (C,H,W,D)
        # -------------------------
        feat = roi_zyx.permute(1, 2, 0).contiguous().unsqueeze(0)  # (1,224,224,160)

        label = torch.tensor(
            self.feature_path_pd['label'].values[index],
            dtype=torch.long
        )

        return {
            "pid": pid,
            "feat": feat,
            "label": label
        }