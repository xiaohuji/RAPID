import sys
import pickle
import os
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import torch
from datasets.datasetMerlin import RCCDatasetMerlinMaskCenter
from merlin import Merlin
torch.backends.cuda.matmul.allow_tf32 = True

def validate(dataloader, model, save_dir, device, mode, key):
    os.makedirs(save_dir, exist_ok=True)
    print(f"{mode} cohort_{key}")
    model.eval()

    bar = tqdm(dataloader, file=sys.stdout)

    with torch.no_grad():
        for step, batch in enumerate(bar):
            # ---- data ----
            pid = batch["pid"][0]
            print(pid)
            x = batch["feat"].to(device).float()

            # ---- forward ----
            embed = model(x)

            # 移到 CPU，便于保存
            embed_cpu = embed.detach().cpu()
            save_path = os.path.join(save_dir, f"{pid}.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(embed_cpu[0].numpy(), f)

def collate_3dino(batch):
    pids = [b["pid"] for b in batch]
    feats = torch.stack([b["feat"] for b in batch], dim=0)   # (B,1,112,112,64)
    labels = torch.stack([b["label"] for b in batch], dim=0) # (B,)
    return {"pid": pids, "feat": feats, "label": labels}


if __name__ == '__main__':
    device = rf"cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    datasets_val = ['', '']

    test_dataloader = {}
    pid_col_name = 'pid'
    label_col_name = 'label'
    img_path_col = 'img_path'

    train_data_info = pd.read_csv(r'')
    ct_feature_root = r''
    train_dataset = RCCDatasetMerlinMaskCenter(pid_col_name=pid_col_name,
                              pid=train_data_info[pid_col_name].values,
                              feature_path_pd=train_data_info,
                              feature_root=ct_feature_root,
                              crop_hw=224, crop_d=160,
                              out_hw=224, out_d=160,
                              target_spacing=(1.5, 1.5, 3.0))
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   drop_last=False)

    # Creating model_construct
    model = Merlin(ImageEmbedding=True).to(device)
    save_dir = r''
    validate(train_dataloader, model, save_dir, device, 'val', '')








