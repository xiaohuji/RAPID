import sys
import argparse
import time
import os
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import seaborn as sns
import random
from tqdm import tqdm
import torch
import torch.nn as nn

from sklearn.metrics import roc_curve, auc, confusion_matrix
from model_teacher.wsi_pair.acmil_wsi_pair import Teacher
from _5_teacher_eval.datasets.dataset import RCCDatasetWSITumor
from _5_teacher_eval.config.acmil_mri_config import get_parser

from sklearn.model_selection import train_test_split
torch.backends.cuda.matmul.allow_tf32 = True

def get_param_groups(model, wd=1e-4,
                     lr_wsi_backbone=2e-5,
                     lr_wsi_head=5e-5,
                     lr_new=1e-4):
    decay_wsi_backbone, no_decay_wsi_backbone = [], []
    decay_wsi_head, no_decay_wsi_head = [], []
    decay_new, no_decay_new = [], []

    def split_params(module, decay_list, no_decay_list):
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1 or name.endswith("bias") or "norm" in name.lower():
                no_decay_list.append(p)
            else:
                decay_list.append(p)

    # WSI backbone
    split_params(model.dimreduction, decay_wsi_backbone, no_decay_wsi_backbone)
    split_params(model.attention, decay_wsi_backbone, no_decay_wsi_backbone)

    # WSI heads
    split_params(model.classifier, decay_wsi_head, no_decay_wsi_head)
    split_params(model.wsi_head, decay_wsi_head, no_decay_wsi_head)

    # New pair/fusion modules
    split_params(model.pair_proj, decay_new, no_decay_new)
    split_params(model.pair_head, decay_new, no_decay_new)
    split_params(model.patch_key, decay_new, no_decay_new)
    split_params(model.pair_to_token, decay_new, no_decay_new)
    split_params(model.fusion_head, decay_new, no_decay_new)

    if hasattr(model, "cross_scale") and model.cross_scale.requires_grad:
        no_decay_new.append(model.cross_scale)

    param_groups = [
        {"params": decay_wsi_backbone, "lr": lr_wsi_backbone, "weight_decay": wd},
        {"params": no_decay_wsi_backbone, "lr": lr_wsi_backbone, "weight_decay": 0.0},

        {"params": decay_wsi_head, "lr": lr_wsi_head, "weight_decay": wd},
        {"params": no_decay_wsi_head, "lr": lr_wsi_head, "weight_decay": 0.0},

        {"params": decay_new, "lr": lr_new, "weight_decay": wd},
        {"params": no_decay_new, "lr": lr_new, "weight_decay": 0.0},
    ]
    return param_groups


def build_optimizer_and_scheduler(model, args):
    param_groups = get_param_groups(
        model,
        wd=args.weight_decay,
        lr_wsi_backbone=args.lr_wsi_backbone,
        lr_wsi_head=args.lr_wsi_head,
        lr_new=args.lr_new,
    )

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.999)
    )

    warmup_epochs = max(int(args.warmup_epochs), 0)
    total_epochs = int(args.epochs)

    if warmup_epochs > 0 and total_epochs > warmup_epochs:
        scheduler_warmup = LinearLR(
            optimizer,
            start_factor=args.warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        scheduler_cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(total_epochs - warmup_epochs, 1),
            eta_min=args.eta_min
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler_warmup, scheduler_cosine],
            milestones=[warmup_epochs]
        )
    elif warmup_epochs > 0:
        scheduler = LinearLR(
            optimizer,
            start_factor=args.warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
    else:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(args.Tmax, 1),
            eta_min=args.eta_min
        )

    return optimizer, scheduler


def get_current_lrs(optimizer):
    return [pg["lr"] for pg in optimizer.param_groups]


def safe_auc(labels, scores) -> float:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    auc_val, _, _, _, _, _, _, _, _ = get_cm(labels, scores)
    return float(auc_val)


def build_class_weight(train_df: pd.DataFrame, label_col: str, device: str):
    neg = int((train_df[label_col] == 0).sum())
    pos = int((train_df[label_col] == 1).sum())
    if neg == 0 or pos == 0:
        return None
    weights = torch.tensor([1.0, neg / max(pos, 1)], dtype=torch.float32, device=device)
    return weights


def build_criterion(train_df: pd.DataFrame, label_col: str, device: str, use_class_weight: bool):
    if not use_class_weight:
        return nn.CrossEntropyLoss()
    class_weight = build_class_weight(train_df, label_col, device)
    if class_weight is None:
        return nn.CrossEntropyLoss()
    return nn.CrossEntropyLoss(weight=class_weight)


# =============================================================================
# Load pretrained WSI branch
# =============================================================================
def load_pretrained_wsi_branch(model, ckpt_path: str, device: str):
    if ckpt_path is None or ckpt_path.strip() == "":
        print("[Info] no pretrained WSI ckpt provided.")
        return

    print(f"[Info] loading pretrained WSI ckpt from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("dimreduction."):
            new_sd[k] = v
        elif k.startswith("attention."):
            new_sd[k] = v
        elif k.startswith("classifier."):
            new_sd[k] = v
        elif k.startswith("Slide_classifier."):
            new_key = "wsi_head." + k[len("Slide_classifier."):]
            new_sd[new_key] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"[Info] pretrained load done.")
    print(f"[Info] missing keys: {len(missing)}")
    print(f"[Info] unexpected keys: {len(unexpected)}")


def set_wsi_backbone_trainable(model, trainable: bool):
    modules = [
        model.dimreduction,
        model.attention,
        model.classifier,
        model.wsi_head,
    ]

    for module in modules:
        for p in module.parameters():
            p.requires_grad = trainable


# =============================================================================
# Loss
# =============================================================================
def compute_sub_loss(sub_preds, y, criterion, device, n_token):
    if n_token <= 1:
        return torch.zeros((), device=device)

    # 期望 [K, 2]
    if sub_preds.dim() == 3 and sub_preds.size(1) == 1:
        sub_preds = sub_preds.squeeze(1)

    if sub_preds.dim() != 2:
        raise ValueError(f"sub_preds should be [K,2], got {tuple(sub_preds.shape)}")

    y_rep = y.repeat_interleave(sub_preds.size(0))
    return criterion(sub_preds, y_rep)


def compute_diff_loss(attn_logits, device, n_token):
    """
    attn_logits: [1, K, N]
    """
    diff_loss = torch.zeros((), device=device, dtype=torch.float32)

    if n_token <= 1:
        return diff_loss

    attn = torch.softmax(attn_logits, dim=-1)

    denom = n_token * (n_token - 1) / 2
    for i in range(n_token):
        for j in range(i + 1, n_token):
            diff_loss = diff_loss + torch.cosine_similarity(
                attn[:, i], attn[:, j], dim=-1
            ).mean() / denom

    return diff_loss


def compute_teacher_loss(out, y, criterion, args, device):
    loss_fuse = criterion(out["slide_logits"], y)
    loss_wsi = criterion(out["wsi_logits"], y)
    loss_pair = criterion(out["pair_logits"], y)

    loss_sub = compute_sub_loss(
        sub_preds=out["sub_preds"],
        y=y,
        criterion=criterion,
        device=device,
        n_token=args.n_token
    )

    diff_loss = compute_diff_loss(
        attn_logits=out["attn_fused_logits"],
        device=device,
        n_token=args.n_token
    )

    total_loss = (
        loss_fuse
        + args.lambda_wsi * loss_wsi
        + args.lambda_pair * loss_pair
        + args.lambda_sub * loss_sub
        + args.lambda_diff * diff_loss
    )

    return {
        "loss": total_loss,
        "loss_fuse": loss_fuse,
        "loss_wsi": loss_wsi,
        "loss_pair": loss_pair,
        "loss_sub": loss_sub,
        "loss_diff": diff_loss,
    }


def get_cm(AllLabels, AllValues, threshold_train=None):
    Auc = 0
    m = t = 0
    presnet_t = 0
    Pos_num = sum(AllLabels)
    Neg_num = len(AllLabels) - Pos_num

    if len(AllValues) > 10 and Pos_num > 0 and Neg_num > 0:
        fpr, tpr, threshold = roc_curve(AllLabels, AllValues, pos_label=1)
        Auc = auc(fpr, tpr)

        for i in range(len(threshold)):
            if tpr[i] - fpr[i] > m:
                m = abs(-fpr[i] + tpr[i])
                t = threshold[i]
                presnet_t = threshold[i]

    if threshold_train is not None:
        t = threshold_train
    AllPred = [int(i >= t) for i in AllValues]
    Acc = sum([AllLabels[i] == AllPred[i]
               for i in range(len(AllPred))]) / len(AllPred)

    Pos_num = sum(AllLabels)
    Neg_num = len(AllLabels) - Pos_num
    Pos_rate = 0
    Neg_rate = 0
    return Auc, Acc, threshold_train, Neg_rate, Pos_rate, len(AllLabels), Pos_num, Neg_num, presnet_t


@torch.no_grad()
def validate(dataloader, model, criterion, device, epoch, args, mode="val", key=None, threshold=None):
    model.eval()

    loss_list = []
    loss_fuse_list = []
    loss_wsi_list = []
    loss_pair_list = []
    loss_sub_list = []
    loss_diff_list = []

    preds_fuse = []
    preds_wsi = []
    preds_pair = []
    trues = []

    presnet_t = threshold if threshold is not None else None
    fuse_auc = 0.0
    fuse_acc = 0.0

    dataset = dataloader.dataset.feature_path_pd.copy()
    dataset['pid'] = dataset['pid'].astype(str)

    bar = tqdm(dataloader, file=sys.stdout)

    for step, batch in enumerate(bar):
        pid = batch['pid']
        wsi = batch["wsi_feat"].to(device).float()
        ct = batch["pair_feat"].to(device).float()
        y = batch["label"].to(device).long().view(-1)

        out = model(wsi, ct)
        loss_dict = compute_teacher_loss(out, y, criterion, args, device)

        fuse_score = torch.softmax(out["slide_logits"], dim=1)[:, 1]
        wsi_score = torch.softmax(out["wsi_logits"], dim=1)[:, 1]
        pair_score = torch.softmax(out["pair_logits"], dim=1)[:, 1]

        preds_fuse.extend(fuse_score.detach().cpu().numpy().tolist())
        preds_wsi.extend(wsi_score.detach().cpu().numpy().tolist())
        preds_pair.extend(pair_score.detach().cpu().numpy().tolist())
        trues.extend(y.detach().cpu().numpy().tolist())

        loss_list.append(float(loss_dict["loss"].detach().cpu()))
        loss_fuse_list.append(float(loss_dict["loss_fuse"].detach().cpu()))
        loss_wsi_list.append(float(loss_dict["loss_wsi"].detach().cpu()))
        loss_pair_list.append(float(loss_dict["loss_pair"].detach().cpu()))
        loss_sub_list.append(float(loss_dict["loss_sub"].detach().cpu()))
        loss_diff_list.append(float(loss_dict["loss_diff"].detach().cpu()))

        avg_loss = float(np.mean(loss_list))
        avg_loss_fuse = float(np.mean(loss_fuse_list))
        avg_loss_wsi = float(np.mean(loss_wsi_list))
        avg_loss_pair = float(np.mean(loss_pair_list))

        fuse_auc, fuse_acc, _, _, _, _, _, _, presnet_t = get_cm(
            np.array(trues),
            np.array(preds_fuse),
            threshold
        )

        wsi_auc = safe_auc(trues, preds_wsi)
        pair_auc = safe_auc(trues, preds_pair)

        dataset.loc[
            dataset['pid'].isin(pid), 'pred_score'
        ] = fuse_score.detach().cpu().numpy().squeeze()

        show_thr = 0 if threshold is None else threshold
        bar.desc = (
            f"[{key}] Ep:{epoch+1} "
            f"AUCf:{fuse_auc:.4f} "
            f"ACCf:{fuse_acc:.4f} "
            f"AUCw:{wsi_auc:.4f} "
            f"AUCp:{pair_auc:.4f} "
            f"loss:{avg_loss:.4f} "
            f"fuseL:{avg_loss_fuse:.4f} "
            f"wsiL:{avg_loss_wsi:.4f} "
            f"pairL:{avg_loss_pair:.4f} "
            f"thr_val:{show_thr:.4f}"
            f"thr_cur:{presnet_t:.4f}"
        )

    return {
        "fuse_auc": float(fuse_auc),
        "fuse_acc": float(fuse_acc),
        "wsi_auc": float(safe_auc(trues, preds_wsi)),
        "pair_auc": float(safe_auc(trues, preds_pair)),
        "loss": float(np.mean(loss_list)) if len(loss_list) > 0 else 0.0,
        "loss_fuse": float(np.mean(loss_fuse_list)) if len(loss_fuse_list) > 0 else 0.0,
        "loss_wsi": float(np.mean(loss_wsi_list)) if len(loss_wsi_list) > 0 else 0.0,
        "loss_pair": float(np.mean(loss_pair_list)) if len(loss_pair_list) > 0 else 0.0,
        "loss_sub": float(np.mean(loss_sub_list)) if len(loss_sub_list) > 0 else 0.0,
        "loss_diff": float(np.mean(loss_diff_list)) if len(loss_diff_list) > 0 else 0.0,
        "threshold": float(presnet_t if presnet_t is not None else 0.5),
        "dataset": dataset
    }


if __name__ == '__main__':
    args = get_parser()
    device = rf"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    seed = 42
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    datasets_val = ['', '']
    pid_col_name = 'pid'
    label_col_name = 'label'
    img_path_col = 'img_path'

    args.n_token = 5
    args.mini_batch = 1

    feature_ct_dir = 'feature_256'
    args.ct_input_shape = 256

    test_dataloader = {}
    top_10_results_wsi_val = []
    top_10_results_wsi_qilu = []
    top_10_results_wsi_beida = []

    train_data_info_wsi = pd.read_csv(r'')
    train_data_info = pd.read_csv(r'')
    train_val_pair_wsi_info_train, train_val_pair_wsi_info_val = train_test_split(
        train_data_info,
        test_size=0.2,
        stratify=train_data_info[label_col_name],
        random_state=seed
    )

    train_val_wsi_feature_root = r'r'
    train_val_pair_feature_root = r''

    train_dataset = RCCDatasetWSITumor(args=args, pid_col_name=pid_col_name,
                                      pid=train_val_pair_wsi_info_train[
                                          pid_col_name].values,
                                      feature_path_pd=train_val_pair_wsi_info_train,
                                      wsi_feature_root=train_val_wsi_feature_root,
                                      pair_feature_root=train_val_pair_feature_root)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   drop_last=False)

    val_dataset = RCCDatasetWSITumor(args=args, pid_col_name=pid_col_name,
                                    pid=train_val_pair_wsi_info_val[
                                        pid_col_name].values,
                                    feature_path_pd=train_val_pair_wsi_info_val,
                                    wsi_feature_root=train_val_wsi_feature_root,
                                    pair_feature_root=train_val_pair_feature_root)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 drop_last=False)


    # Creating model_construct
    model = Teacher(
        conf=args,
        D=args.attn_hidden_dim,
        droprate=args.drop_rate,
        pair_dim=args.pair_dim,
        n_token=args.n_token,
        n_masked_patch=args.n_masked_patch,
        mask_drop=args.mask_drop,
    ).to(device)
    pretrained_dict = torch.load(rf'', map_location='cuda:' + str(args.device))['model']
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in pretrained_dict.items() if
                  k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    # -------------------------------------------------------------------------
    # criterion / optimizer / scheduler
    # -------------------------------------------------------------------------
    criterion = build_criterion(
        train_df=train_val_pair_wsi_info_train,
        label_col=label_col_name,
        device=device,
        use_class_weight=args.use_class_weight
    )

    early_stop = 0
    max_auc = 0
    max_auc_wsi_qilu = 0
    max_auc_wsi_val = 0
    max_auc_wsi_beida = 0
    all_epochs_auc_records = []
    all_epochs_auc_records_mix = []
    epoch = 0
    for epoch in range(1):

        set_wsi_backbone_trainable(model, True)
        stage_msg = "freeze_wsi_backbone=False"
        AUC_list = []
        epoch_record = {'Epoch': epoch}

        if epoch % 1 == 0:
            train_val_metrics = validate(
                train_dataloader, model, criterion, device, epoch, args, 'val', '')
            AUC_list.append(train_val_metrics['fuse_auc'])
            epoch_record['train_val_AUC'] = train_val_metrics['fuse_auc']
            train_val_metrics["dataset"].to_csv(rf'', index=False, encoding='utf-8-sig')

            val_metrics = validate(val_dataloader, model, criterion,
                                          device, epoch, args, 'val', '', threshold=train_val_metrics['threshold'])
            AUC_list.append(val_metrics['fuse_auc'])
            epoch_record['val_AUC'] = val_metrics['fuse_auc']
            epoch_record['val_AUCw'] = val_metrics['wsi_auc']
            epoch_record['val_AUCp'] = val_metrics['pair_auc']
            val_metrics["dataset"].to_csv(rf'', index=False, encoding='utf-8-sig')