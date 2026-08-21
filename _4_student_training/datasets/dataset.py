import os
from torch.utils.data import Dataset
import h5py
import pickle
import torch

class RCCDataset(Dataset):
    def __init__(self, args, pid_col_name, pid, feature_path_pd, wsi_feature_root, model_name='uni'):
        super().__init__()
        self.args = args
        self.pid = pid
        self.pid_col_name = pid_col_name
        self.feature_path_pd = feature_path_pd
        self.feature_path_pd[pid_col_name] = self.feature_path_pd[pid_col_name].astype(str)
        self.wsi_feature_root = wsi_feature_root
        self.model_name = model_name


    def __len__(self):
        return len(self.pid)

    def __getitem__(self, index):
        pid = self.pid[index]
        feat_name = self.feature_path_pd['img_name'].values[index].split('.')[0]
        h5ad_path = os.path.join(self.wsi_feature_root, '{}__{}_tiles.h5ad'.format(feat_name, self.model_name))

        try:
            with h5py.File(h5ad_path, 'r') as f:
                # 直接读取 dataset 到 numpy 数组，读取完后文件自动关闭
                feat = f['X'][:]
        except Exception as e:
            print(f"Error reading {h5ad_path}: {e}")
            # 如果读取失败，做一个 fallback 或者 raise error
            # feat = np.zeros((1, 1024)) # 示例
            raise e

        # feat = ad.read_h5ad(h5ad_path).X
        label = self.feature_path_pd['label'].values[index]

        return {
            'pid': pid,
            'wsi_feat': feat,
            'label': label,
        }



class RCCDatasetWSIPair(Dataset):
    def __init__(self, args, pid, feature_path_pd, pid_col_name, wsi_feature_root, pair_feature_root, model_name='uni'):
        super().__init__()
        self.args = args
        self.pid = pid
        self.pid_col_name = pid_col_name
        self.feature_path_pd = feature_path_pd
        self.feature_path_pd[pid_col_name] = self.feature_path_pd[pid_col_name].astype(str)
        self.wsi_feature_root = wsi_feature_root
        self.pair_feature_root = pair_feature_root
        self.model_name = model_name

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
        label = self.label_dict[pid]
        feat_name = str(self.feature_path_pd['wsi_img_name'].values[index]).split('.')[0]

        # WSI feature
        wsi_path = os.path.join(
            self.wsi_feature_root,
            f"{feat_name}__{self.model_name}_tiles.h5ad"
        )
        with h5py.File(wsi_path, "r") as f:
            wsi_feat = f["X"][:]   # [N, 1024]

        # MRI feature
        mr_path = os.path.join(self.pair_feature_root, f"{pid}.pkl")
        with open(mr_path, "rb") as f:
            pair_feat = pickle.load(f)   # expected [1024]

        wsi_feat = torch.tensor(wsi_feat, dtype=torch.float32)
        pair_feat = torch.tensor(pair_feat, dtype=torch.float32)

        return {
            "pid": pid,
            "wsi_feat": wsi_feat,
            "pair_feat": pair_feat,
            "label": label,
        }


class RCCDatasetWSITumor(Dataset):
    def __init__(self, args, pid, feature_path_pd, pid_col_name, wsi_feature_root, pair_feature_root, model_name='uni'):
        super().__init__()
        self.args = args
        self.pid = pid
        self.pid_col_name = pid_col_name
        self.feature_path_pd = feature_path_pd
        self.feature_path_pd[pid_col_name] = self.feature_path_pd[pid_col_name].astype(str)
        self.wsi_feature_root = wsi_feature_root
        self.pair_feature_root = pair_feature_root
        self.model_name = model_name

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
        label = self.label_dict[pid]
        feat_name = str(self.feature_path_pd['wsi_img_name'].values[index]).split('.')[0]

        # WSI feature
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

        # MRI feature
        mr_path = os.path.join(self.pair_feature_root, f"{pid}.pkl")
        with open(mr_path, "rb") as f:
            pair_feat = pickle.load(f)   # expected [1024]

        wsi_feat = torch.tensor(wsi_feat, dtype=torch.float32)
        pair_feat = torch.tensor(pair_feat, dtype=torch.float32)

        return {
            "pid": pid,
            "wsi_feat": wsi_feat,
            "pair_feat": pair_feat,
            "label": label,
        }

