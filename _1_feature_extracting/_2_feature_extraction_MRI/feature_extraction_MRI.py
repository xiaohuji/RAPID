import sys
import pickle
import os
import pandas as pd
from tqdm import tqdm
import torch
from datasets.dataset import RCCDataset3DINOBox
from dinov2.eval.setup import build_model_for_eval
from dinov2.configs import load_and_merge_config_3d
torch.backends.cuda.matmul.allow_tf32 = True


def criterion():
    torch.nn.BCEWithLogitsLoss()


def validate(dataloader, model, save_dir, device, mode, key):
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

    pid_col_name = 'pid'
    label_col_name = 'label'

    train_data_info = pd.read_csv(r'D:\pyfile\RCC_Classify\rcc_code_CTMR\data_summary\patient_info\301rcc_MR_with_imgname.csv')
    train_data_info = train_data_info[train_data_info[label_col_name].isin([0, 1])]
    train_data_info.drop_duplicates(subset=['pid'], keep='first',
                              inplace=True)
    train_data_info = train_data_info.reset_index(
        drop=True)

    train_val_wsi_feature_root = r'D:\pyfile\RCC_Classify\dataset\MR_resample\080860_noW\h301'

    train_dataset = RCCDataset3DINOBox(pid_col_name=pid_col_name,
                              pid=train_data_info[pid_col_name].values,
                              feature_path_pd=train_data_info,
                              feature_root=train_val_wsi_feature_root,
                                       box_d=64)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   drop_last=False,
                                                   collate_fn=collate_3dino)

    # Creating model_construct
    config_file = 'train/vit3d_highres'
    pretrained_weights = r'./3dino_vit_weights.pth'  # adjust this to local path

    cfg = load_and_merge_config_3d(config_file)
    backbone_model = build_model_for_eval(cfg, pretrained_weights)
    model = backbone_model.to(device)
    save_dir = r''
    validate(train_dataloader, model, save_dir, device, 'val','')









