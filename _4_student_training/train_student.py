import sys
import math
import os
import copy
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import roc_curve, auc
from model_student.distill_student import Student
from model_teacher.wsi_pair.acmil_wsi_pair import Teacher
from datasets.dataset_student import RCCDatasetWSIDistillTumor
from _4_student_training.config.student_config import get_parser

from sklearn.model_selection import train_test_split
torch.backends.cuda.matmul.allow_tf32 = True


class WarmupStableDecayScheduler(LambdaLR):
    def __init__(
            self,
            optimizer,
            warmup_epochs: int,
            stable_epochs: int,
            decay_epochs: int,
            eta_min_ratio: float = 0.01,
            last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.stable_epochs = stable_epochs
        self.decay_epochs = decay_epochs
        self.eta_min_ratio = eta_min_ratio

        # 计算各阶段边界
        self.warmup_end = warmup_epochs
        self.stable_end = warmup_epochs + stable_epochs
        self.total_epochs = warmup_epochs + stable_epochs + decay_epochs

        def lr_lambda(epoch):
            if epoch < self.warmup_end:
                # Phase 1: 线性 warmup, 从 eta_min_ratio 到 1.0
                return self.eta_min_ratio + (1.0 - self.eta_min_ratio) * (epoch / max(self.warmup_end, 1))
            elif epoch < self.stable_end:
                # Phase 2: 稳定平台
                return 1.0
            else:
                # Phase 3: 余弦衰减, 从 1.0 到 eta_min_ratio
                progress = (epoch - self.stable_end) / max(self.decay_epochs, 1)
                progress = min(progress, 1.0)
                return self.eta_min_ratio + 0.5 * (1.0 - self.eta_min_ratio) * (1.0 + math.cos(math.pi * progress))

        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)


def kd_ramp_weight(epoch, kd_start_epoch, kd_ramp_epochs, target):
    if epoch < kd_start_epoch:
        return 0.0
    elapsed = epoch - kd_start_epoch
    if kd_ramp_epochs <= 0:
        return target
    alpha = min(1.0, elapsed / kd_ramp_epochs)
    return target * alpha


def build_master_table(
    df_wsi,
    df_ct=None,
    df_mri=None,
    pid_col="pid",
    label_col="label",
    wsi_img_col="img_name",
    ct_feat_path_col=None,
    mri_feat_path_col=None,
):
    df_wsi = df_wsi.copy()
    df_wsi = df_wsi[df_wsi[label_col].isin([0, 1])].copy()
    df_wsi[pid_col] = df_wsi[pid_col].astype(str)
    df_wsi.drop_duplicates(subset=[pid_col], keep="first", inplace=True)

    master = df_wsi.rename(
        columns={wsi_img_col: "wsi_img_name"}
    ).reset_index(drop=True)

    master["has_ct"] = 0
    master["has_mri"] = 0
    master["ct_feat_path"] = ""
    master["mri_feat_path"] = ""

    if df_ct is not None:
        ct = df_ct.copy()
        ct = ct[ct[label_col].isin([0, 1])].copy()
        ct[pid_col] = ct[pid_col].astype(str)
        ct.drop_duplicates(subset=[pid_col], keep="first", inplace=True)

        keep_cols = [pid_col]
        if ct_feat_path_col is not None and ct_feat_path_col in ct.columns:
            keep_cols.append(ct_feat_path_col)

        ct = ct[keep_cols].copy()
        ct["has_ct"] = 1
        if ct_feat_path_col is not None and ct_feat_path_col in ct.columns:
            ct = ct.rename(columns={ct_feat_path_col: "ct_feat_path"})
        else:
            ct["ct_feat_path"] = ""

        master = master.merge(
            ct[[pid_col, "has_ct", "ct_feat_path"]],
            on=pid_col,
            how="left",
            suffixes=("", "_ct"),
        )
        master["has_ct"] = master["has_ct_ct"].fillna(master["has_ct"]).fillna(0).astype(int)
        master["ct_feat_path"] = master["ct_feat_path_ct"].fillna(master["ct_feat_path"]).fillna("")
        master.drop(columns=["has_ct_ct", "ct_feat_path_ct"], inplace=True)

    if df_mri is not None:
        mri = df_mri.copy()
        mri = mri[mri[label_col].isin([0, 1])].copy()
        mri[pid_col] = mri[pid_col].astype(str)
        mri.drop_duplicates(subset=[pid_col], keep="first", inplace=True)

        keep_cols = [pid_col]
        if mri_feat_path_col is not None and mri_feat_path_col in mri.columns:
            keep_cols.append(mri_feat_path_col)

        mri = mri[keep_cols].copy()
        mri["has_mri"] = 1
        if mri_feat_path_col is not None and mri_feat_path_col in mri.columns:
            mri = mri.rename(columns={mri_feat_path_col: "mri_feat_path"})
        else:
            mri["mri_feat_path"] = ""

        master = master.merge(
            mri[[pid_col, "has_mri", "mri_feat_path"]],
            on=pid_col,
            how="left",
            suffixes=("", "_mri"),
        )
        master["has_mri"] = master["has_mri_mri"].fillna(master["has_mri"]).fillna(0).astype(int)
        master["mri_feat_path"] = master["mri_feat_path_mri"].fillna(master["mri_feat_path"]).fillna("")
        master.drop(columns=["has_mri_mri", "mri_feat_path_mri"], inplace=True)

    master["source_type"] = 0
    master.loc[(master["has_ct"] == 1) & (master["has_mri"] == 0), "source_type"] = 1
    master.loc[(master["has_ct"] == 0) & (master["has_mri"] == 1), "source_type"] = 2
    master.loc[(master["has_ct"] == 1) & (master["has_mri"] == 1), "source_type"] = 3

    master.dropna(subset=["wsi_img_name"], inplace=True)
    master = master.reset_index(drop=True)
    return master


def build_wsi_only_table(df_wsi, pid_col="pid", label_col="label", wsi_img_col="img_name"):
    df = df_wsi.copy()
    df = df[df[label_col].isin([0, 1])].copy()
    df[pid_col] = df[pid_col].astype(str)
    df.drop_duplicates(subset=[pid_col], keep="first", inplace=True)

    out = df.rename(columns={wsi_img_col: "wsi_img_name"}).reset_index(drop=True)
    out["has_ct"] = 0
    out["has_mri"] = 0
    out["ct_feat_path"] = ""
    out["mri_feat_path"] = ""
    out["source_type"] = 0
    return out


def safe_auc(labels, scores) -> float:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    auc_val, _, _, _, _, _, _, _, _ = get_cm(labels, scores)
    return float(auc_val)


def build_class_weight(train_df, label_col, device, max_pos_weight=4.0):
    neg = int((train_df[label_col] == 0).sum())
    pos = int((train_df[label_col] == 1).sum())
    if neg == 0 or pos == 0:
        return None

    pos_weight = (neg / max(pos, 1)) ** 0.5
    pos_weight = min(pos_weight, max_pos_weight)

    return torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device)



def build_criterion(train_df: pd.DataFrame, label_col: str, device: str, use_class_weight: bool):
    if not use_class_weight:
        return nn.CrossEntropyLoss()
    class_weight = build_class_weight(train_df, label_col, device)
    if class_weight is None:
        return nn.CrossEntropyLoss()
    return nn.CrossEntropyLoss(weight=class_weight)


def build_optimizer(model, lr, weight_decay, accum_steps=1):
    effective_lr = lr * math.sqrt(accum_steps)

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay,    "lr": effective_lr, "weight_decay": weight_decay},
            {"params": no_decay, "lr": effective_lr, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.98),
        eps=1e-8,
    )
    return optimizer


def ramp_weight(epoch, start_epoch, ramp_epochs, target):
    if epoch < start_epoch:
        return 0.0
    if ramp_epochs <= 0:
        return target
    alpha = min(1.0, (epoch - start_epoch + 1) / ramp_epochs)
    return target * alpha


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

    if sub_preds.dim() == 3 and sub_preds.size(1) == 1:
        sub_preds = sub_preds.squeeze(1)

    if sub_preds.dim() != 2:
        raise ValueError(f"sub_preds should be [K,2], got {tuple(sub_preds.shape)}")

    y_rep = y.repeat_interleave(sub_preds.size(0))
    return criterion(sub_preds, y_rep)


def compute_diff_loss(attn_logits, device, n_token):
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


def _get_branch_feat(t_out, feat_source):
    if feat_source not in t_out:
        raise KeyError(f"{feat_source} not found in teacher output keys: {list(t_out.keys())}")
    feat = t_out[feat_source]
    if feat.dim() == 1:
        feat = feat.unsqueeze(0)
    return F.normalize(feat.detach(), dim=-1)


def select_proto_teacher_feat(teacher_out, feat_source="bag_feat_fused"):
    if teacher_out is None:
        return None

    out_ct = teacher_out.get("ct", None)
    out_mri = teacher_out.get("mri", None)

    z_ct = _get_branch_feat(out_ct, feat_source) if out_ct is not None else None
    z_mri = _get_branch_feat(out_mri, feat_source) if out_mri is not None else None

    if z_ct is not None and z_mri is not None:
        z_both = F.normalize((z_ct + z_mri) / 2.0, dim=-1)
        return z_both

    if z_ct is not None:
        return z_ct

    if z_mri is not None:
        return z_mri

    return None




def kd_logits(student_logits, teacher_logits, T=4.0):
    s_log_prob = F.log_softmax(student_logits / T, dim=-1)
    t_prob = F.softmax(teacher_logits / T, dim=-1)
    return F.kl_div(s_log_prob, t_prob, reduction="batchmean") * (T ** 2)


def kd_attention(student_attn_logits, teacher_attn_logits):
    if student_attn_logits.shape != teacher_attn_logits.shape:
        raise ValueError(
            f"Attention shape mismatch: student {tuple(student_attn_logits.shape)} "
            f"vs teacher {tuple(teacher_attn_logits.shape)}"
        )

    s_log_prob = F.log_softmax(student_attn_logits, dim=-1)
    t_prob = F.softmax(teacher_attn_logits, dim=-1)
    return F.kl_div(s_log_prob, t_prob, reduction="batchmean")


def kd_logits_from_prob(student_logits, teacher_prob, T):
    s_log_prob = F.log_softmax(student_logits / T, dim=-1)
    return F.kl_div(s_log_prob, teacher_prob, reduction="batchmean") * (T * T)


def kd_attention_from_prob(student_attn_logits, teacher_attn_prob):
    s_log_prob = F.log_softmax(student_attn_logits, dim=-1)
    return F.kl_div(s_log_prob, teacher_attn_prob, reduction="batchmean")


def teacher_confidence_weight(teacher_logits):
    prob = F.softmax(teacher_logits, dim=-1)
    entropy = -(prob * torch.log(prob.clamp_min(1e-8))).sum(dim=-1)
    max_entropy = math.log(prob.size(-1))
    conf = 1.0 - entropy / max_entropy
    return conf.mean().detach()


def load_ckpt_state_dict(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    return ckpt


def load_student_init(student, ckpt_path, device):
    if ckpt_path is None or ckpt_path.strip() == "":
        print("[Info] no student init ckpt.")
        return

    print(f"[Info] loading student init from: {ckpt_path}")
    state_dict = load_ckpt_state_dict(ckpt_path, device)

    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("dimreduction."):
            new_sd[k] = v
        elif k.startswith("attention."):
            new_sd[k] = v
        elif k.startswith("classifier."):
            new_sd[k] = v
        elif k.startswith("Slide_classifier."):
            new_key = "slide_head." + k[len("Slide_classifier."):]
            new_sd[new_key] = v
        elif k.startswith("wsi_head."):
            new_key = "slide_head." + k[len("wsi_head."):]
            new_sd[new_key] = v

    missing, unexpected = student.load_state_dict(new_sd, strict=False)
    print(f"[Info] student init done. missing={len(missing)}, unexpected={len(unexpected)}")


def build_teacher_model(args, ckpt_path, pair_dim, device):
    if ckpt_path is None or ckpt_path.strip() == "":
        return None

    teacher_args = copy.deepcopy(args)
    teacher_args.pair_dim = pair_dim

    model = Teacher(
        conf=teacher_args,
        D=teacher_args.attn_hidden_dim,
        droprate=teacher_args.drop_rate,
        pair_dim=pair_dim,
        n_token=teacher_args.n_token,
        n_masked_patch=teacher_args.n_masked_patch,
        mask_drop=teacher_args.mask_drop,
    ).to(device)

    state_dict = load_ckpt_state_dict(ckpt_path, device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        f"[Info] teacher loaded: {ckpt_path}\n"
        f"       missing={len(missing)}, unexpected={len(unexpected)}"
    )

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model


def get_teacher_out(batch, teacher_ct, teacher_mri, device):
    source_type = int(batch["source_type"].item())
    out = {}

    with torch.no_grad():
        if source_type in [1, 3] and teacher_ct is not None:
            out["ct"] = teacher_ct(
                batch["wsi_feat"].to(device).float(),
                batch["ct_feat"].to(device).float()
            )

        if source_type in [2, 3] and teacher_mri is not None:
            out["mri"] = teacher_mri(
                batch["wsi_feat"].to(device).float(),
                batch["mri_feat"].to(device).float()
            )

    return out if len(out) > 0 else None



def compute_student_loss(
    student_out,
    teacher_out,
    y,
    criterion,
    args,
    device,
    lambda_kd=None,
    lambda_feat=None,
    lambda_anchor=None,
):

    _lkd  = lambda_kd     if lambda_kd     is not None else args.lambda_kd
    _lfeat = lambda_feat   if lambda_feat   is not None else args.lambda_feat
    _lanc  = lambda_anchor if lambda_anchor is not None else args.lambda_anchor

    # ---------- CE loss ----------
    loss_ce = criterion(student_out["slide_logits"], y)

    # ---------- sub / diff ----------
    loss_sub = compute_sub_loss(
        sub_preds=student_out["sub_preds"],
        y=y, criterion=criterion, device=device, n_token=args.n_token,
    )
    loss_diff = compute_diff_loss(
        attn_logits=student_out["attn_logits"],
        device=device, n_token=args.n_token,
    )

    # ---------- KD / feat / anchor ----------
    loss_kd     = torch.zeros((), device=device)
    loss_feat   = torch.zeros((), device=device)
    loss_anchor = torch.zeros((), device=device)

    student_feat_norm = F.normalize(student_out["bag_feat"], dim=-1)

    if teacher_out is not None:
        has_ct  = "ct"  in teacher_out
        has_mri = "mri" in teacher_out

        if has_ct and not has_mri:
            t = teacher_out["ct"]
            t_feat = _get_branch_feat(t, args.proto_feat_source)
            loss_kd = kd_logits(
                student_logits=student_out["slide_logits"],
                teacher_logits=t["slide_logits"].detach(),
                T=args.distill_temperature,
            )
            loss_feat = F.mse_loss(student_feat_norm, t_feat)

        elif has_mri and not has_ct:
            t = teacher_out["mri"]
            t_feat = _get_branch_feat(t, args.proto_feat_source)
            loss_kd = kd_logits(
                student_logits=student_out["slide_logits"],
                teacher_logits=t["slide_logits"].detach(),
                T=args.distill_temperature,
            )
            loss_feat = F.mse_loss(student_feat_norm, t_feat)

        elif has_ct and has_mri:
            t_ct  = teacher_out["ct"]
            t_mri = teacher_out["mri"]
            conf_ct  = teacher_confidence_weight(t_ct["slide_logits"].detach())
            conf_mri = teacher_confidence_weight(t_mri["slide_logits"].detach())
            w_ct  = conf_ct  / (conf_ct + conf_mri + 1e-8)
            w_mri = conf_mri / (conf_ct + conf_mri + 1e-8)

            prob_target = (
                w_ct * torch.softmax(
                    t_ct["slide_logits"].detach() / args.distill_temperature, dim=-1
                )
                + w_mri * torch.softmax(
                    t_mri["slide_logits"].detach() / args.distill_temperature, dim=-1
                )
            )
            feat_ct  = _get_branch_feat(t_ct,  args.proto_feat_source)
            feat_mri = _get_branch_feat(t_mri, args.proto_feat_source)
            feat_target = F.normalize(w_ct * feat_ct + w_mri * feat_mri, dim=-1)

            loss_kd = kd_logits_from_prob(
                student_logits=student_out["slide_logits"],
                teacher_prob=prob_target,
                T=args.distill_temperature,
            )
            loss_feat = F.mse_loss(student_feat_norm, feat_target)

            z_both = F.normalize((feat_ct + feat_mri) / 2.0, dim=-1)
            loss_anchor = F.mse_loss(student_feat_norm, z_both)


    total_loss = (
        args.lambda_ce   * loss_ce
        + args.lambda_sub  * loss_sub
        + args.lambda_diff * loss_diff
        + _lkd             * loss_kd
        + _lfeat           * loss_feat
        + _lanc            * loss_anchor
    )

    return {
        "loss":        total_loss,
        "loss_ce":     loss_ce,
        "loss_sub":    loss_sub,
        "loss_diff":   loss_diff,
        "loss_kd":     loss_kd,
        "loss_feat":   loss_feat,
        "loss_anchor": loss_anchor
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

    TP = sum([AllLabels[i] == 1 and AllPred[i] == 1 for i in range(len(AllPred))])
    FP = sum([AllLabels[i] == 0 and AllPred[i] == 1 for i in range(len(AllPred))])
    FN = sum([AllLabels[i] == 1 and AllPred[i] == 0 for i in range(len(AllPred))])

    Sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
    Precision = TP / (TP + FP) if (TP + FP) != 0 else 0

    return Auc, Acc, threshold_train, Neg_rate, Pos_rate, len(AllLabels), Pos_num, Neg_num, presnet_t, Sensitivity, Precision



def train(dataloader, model, teacher_ct, teacher_mri, criterion, optimizer, device, epoch, args):
    model.train()

    kd_start_epoch   = getattr(args, 'kd_start_epoch', 10)
    kd_ramp_ep  = getattr(args, 'kd_ramp_epochs', 5)

    cur_lkd  = kd_ramp_weight(epoch, kd_start_epoch, kd_ramp_ep, args.lambda_kd)
    cur_lfeat = kd_ramp_weight(epoch, kd_start_epoch, kd_ramp_ep, args.lambda_feat)
    cur_lanc  = kd_ramp_weight(epoch, kd_start_epoch, kd_ramp_ep, args.lambda_anchor)

    loss_list, ce_list, sub_list, diff_list = [], [], [], []
    kd_list, feat_list = [], []
    preds, trues = [], []

    threshold = 0.5
    auc_now = acc_now = sensitivity = precision = 0.0

    bar = tqdm(dataloader)
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(bar):
        wsi = batch["wsi_feat"].to(device).float()
        y   = batch["label"].to(device).long().view(-1)

        student_out = model(wsi_feat=wsi, use_attention_mask=(args.n_masked_patch > 0))
        teacher_out = get_teacher_out(batch, teacher_ct, teacher_mri, device)




        loss_dict = compute_student_loss(
            student_out=student_out,
            teacher_out=teacher_out,
            y=y, criterion=criterion, args=args, device=device,
            lambda_kd=cur_lkd,
            lambda_feat=cur_lfeat,
            lambda_anchor=cur_lanc,
        )

        loss = loss_dict["loss"]
        (loss / args.accum_steps).backward()

        should_step = ((step + 1) % args.accum_steps == 0) or ((step + 1) == len(dataloader))
        if should_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # ★ 更保守
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            pos_score = torch.softmax(student_out["slide_logits"], dim=1)[:, 1]

        preds.extend(pos_score.detach().cpu().numpy().tolist())
        trues.extend(y.detach().cpu().numpy().tolist())

        loss_list.append(float(loss.detach().cpu()))
        ce_list.append(float(loss_dict["loss_ce"].detach().cpu()))
        sub_list.append(float(loss_dict["loss_sub"].detach().cpu()))
        diff_list.append(float(loss_dict["loss_diff"].detach().cpu()))
        kd_list.append(float(loss_dict["loss_kd"].detach().cpu()))
        feat_list.append(float(loss_dict["loss_feat"].detach().cpu()))

        auc_now, acc_now, _, _, _, _, _, _, presnet_t, sensitivity, precision = get_cm(
            np.array(trues), np.array(preds)
        )

        bar.desc = (
            f"[student train] Ep:{epoch+1} "
            f"AUC:{auc_now:.4f} ACC:{acc_now:.4f} "
            f"Sen:{sensitivity:.4f} Pre:{precision:.4f} "
            f"loss:{np.mean(loss_list):.4f} "
            f"ce:{np.mean(ce_list):.4f} "
            f"kd:{np.mean(kd_list):.4f}(w={cur_lkd:.3f}) "
            f"feat:{np.mean(feat_list):.4f} "
            f"thr:{presnet_t:.4f}"
        )

    return {
        "auc": float(auc_now),
        "acc": float(acc_now),
        "sen": float(sensitivity),
        "pre": float(precision),
        "loss": float(np.mean(loss_list)) if loss_list else 0.0,
        "loss_ce": float(np.mean(ce_list)) if ce_list else 0.0,
        "loss_sub": float(np.mean(sub_list)) if sub_list else 0.0,
        "loss_diff": float(np.mean(diff_list)) if diff_list else 0.0,
        "loss_kd": float(np.mean(kd_list)) if kd_list else 0.0,
        "loss_feat": float(np.mean(feat_list)) if feat_list else 0.0,
        "threshold": float(threshold),
    }

@torch.no_grad()
def validate(dataloader, model, criterion, device, epoch, args, mode="val", key=None, threshold=None):
    model.eval()

    loss_list = []
    preds = []
    trues = []

    presnet_t = threshold if threshold is not None else None
    auc_now = acc_now = sensitivity = precision = 0.0

    bar = tqdm(dataloader, file=sys.stdout)

    for step, batch in enumerate(bar):
        wsi = batch["wsi_feat"].to(device).float()
        y = batch["label"].to(device).long().view(-1)

        out = model(wsi_feat=wsi, use_attention_mask=False)

        loss = criterion(out["slide_logits"], y)
        loss_list.append(float(loss.detach().cpu()))

        pos_score = torch.softmax(out["slide_logits"], dim=1)[:, 1]
        preds.extend(pos_score.detach().cpu().numpy().tolist())
        trues.extend(y.detach().cpu().numpy().tolist())

        auc_now, acc_now, _, _, _, _, _, _, presnet_t, sensitivity, precision = get_cm(
            np.array(trues),
            np.array(preds),
            threshold
        )

        show_thr = 0 if threshold is None else threshold

        bar.desc = (
            f"[{key}] Ep:{epoch + 1} "
            f"AUC:{auc_now:.4f} "
            f"ACC:{acc_now:.4f} "
            f"Sen:{sensitivity:.4f} Pre:{precision:.4f} "
            f"loss:{np.mean(loss_list):.4f} "
            f"thr_val:{show_thr:.4f}"
            f"thr_cur:{presnet_t:.4f}"
        )

    return {
        "auc": float(auc_now),
        "acc": float(acc_now),
        "sen": float(sensitivity),
        "pre": float(precision),
        "loss": float(np.mean(loss_list)) if len(loss_list) > 0 else 0.0,
        "threshold": float(presnet_t if presnet_t is not None else 0.5),
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

    pid_col_name = 'pid'
    label_col_name = 'label'
    img_path_col = 'img_path'
    ck_file = r'student_both'

    args.mask_drop = 0.3
    args.n_masked_patch = 10
    args.lr= 2e-4
    args.n_token = 5
    args.weight_decay = 1e-4
    args.epochs = 10

    accum_stepss = [1, 4, 8]

    minimum_tumor_patch = 5

    top_10_results_wsi_val = []
    top_10_results_wsi_qilu = []
    top_10_results_wsi_beida = []

    for accum_steps in accum_stepss:
        args.accum_steps = accum_steps

        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True

        # ---------------------------------------------------------------------------------------
        train_data_info_wsi = pd.read_csv(r'')
        train_data_info_ct = pd.read_csv(r'')
        train_data_info_mri = pd.read_csv(r'')
        train_data_info = build_master_table(df_wsi=train_data_info_wsi, df_mri=train_data_info_mri, df_ct=train_data_info_ct)
        # ---------------------------------------------------------------------------------------
        train_data_info_multi  = train_data_info[train_data_info['source_type'].isin([1, 2, 3])]
        train_data_info_wsi_only = train_data_info[train_data_info['source_type'] == 0]
        train_data_info_wsi_only_train, train_data_info_wsi_only_val = train_test_split(
            train_data_info_wsi_only,
            test_size=0.36339,
            stratify=train_data_info_wsi_only[label_col_name],  # 按目标列分层
            random_state=seed
        )

        train_data_info_train = pd.concat([train_data_info_multi, train_data_info_wsi_only_train])

        train_data_info_train_val = train_data_info_train.copy()
        train_data_info_train_val.loc[:, ["has_ct", "has_mri", "source_type"]] = 0
        train_data_info_train_val.loc[:, ["ct_feat_path", "mri_feat_path"]] = ""

        train_val_wsi_feature_root = r''
        train_val_ct_feature_root = r''
        train_val_mr_feature_root = r''

        train_dataset = RCCDatasetWSIDistillTumor(args=args, pid_col_name=pid_col_name,
                                             feature_path_pd=train_data_info_train,
                                             wsi_feature_root=train_val_wsi_feature_root,
                                             ct_feature_root=train_val_ct_feature_root,
                                             mri_feature_root=train_val_mr_feature_root)
        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=args.batch_size,
                                                       shuffle=True,
                                                       drop_last=False)

        train_val_dataset = RCCDatasetWSIDistillTumor(args=args, pid_col_name=pid_col_name,
                                             feature_path_pd=train_data_info_train_val,
                                             wsi_feature_root=train_val_wsi_feature_root)
        train_val_dataloader = torch.utils.data.DataLoader(train_val_dataset,
                                                       batch_size=args.batch_size,
                                                       shuffle=True,
                                                       drop_last=False)

        val_dataset = RCCDatasetWSIDistillTumor(args=args, pid_col_name=pid_col_name,
                                           feature_path_pd=train_data_info_wsi_only_val,
                                           wsi_feature_root=train_val_wsi_feature_root)
        val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=1,
                                                     shuffle=False,
                                                     drop_last=False)

        # Creating model_construct
        # -------------------------
        # student
        # -------------------------
        model = Student(
            conf=args,
            D=args.attn_hidden_dim,
            droprate=args.drop_rate,
            n_token=args.n_token,
            n_masked_patch=args.n_masked_patch,
            mask_drop=args.mask_drop,
        ).to(device)

        # load_student_init(model, args.student_init_ckpt, device)

        # -------------------------
        # teachers
        # -------------------------
        teacher_ct = build_teacher_model(
            args=args,
            ckpt_path=args.teacher_ct_ckpt,
            pair_dim=args.ct_pair_dim,
            device=device
        )

        teacher_mri = build_teacher_model(
            args=args,
            ckpt_path=args.teacher_mri_ckpt,
            pair_dim=args.mri_pair_dim,
            device=device
        )

        # -------------------------------------------------------------------------
        # criterion / optimizer / scheduler
        # -------------------------------------------------------------------------
        criterion = build_criterion(
            train_df=train_data_info_train,
            label_col=args.label_col,
            device=device,
            use_class_weight=args.use_class_weight
        )

        optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)

        decay_epochs = max(args.epochs - args.warmup_epochs - args.stable_epochs, 1)
        scheduler = WarmupStableDecayScheduler(
         optimizer,
         warmup_epochs = args.warmup_epochs,
         stable_epochs = args.stable_epochs,
         decay_epochs  = decay_epochs,
         eta_min_ratio = 0.01,
        )

        early_stop = 0
        max_auc = 0
        max_auc_wsi_qilu = 0
        max_auc_wsi_val = 0
        max_auc_wsi_beida = 0
        all_epochs_auc_records = []
        all_epochs_auc_records_mix = []
        epoch = 0
        for epoch in range(args.epochs):
            early_stop = early_stop + 1
            AUC_list = []

            if early_stop > 4:
                print(
                    'Early stop!')
                break

            epoch_record = {'Epoch': epoch}
            train_metrics = train(train_dataloader, model, teacher_ct, teacher_mri, criterion,
                                             optimizer, device, epoch, args)
            AUC_list.append(train_metrics['auc'])
            epoch_record['train_AUC'] = train_metrics['auc']
            epoch_record['train_ACC'] = train_metrics['acc']

            scheduler.step()
            print(f'Epoch {epoch+1} lr: {scheduler.get_last_lr()[0]:.2e}')

            if epoch % 1 == 0:
                train_val_metrics = validate(
                    train_val_dataloader, model, criterion, device, epoch, args, 'val','')
                AUC_list.append(train_val_metrics['auc'])
                epoch_record['train_val_AUC'] = train_val_metrics['auc']
                epoch_record['train_val_ACC'] = train_val_metrics['acc']
                epoch_record['train_val_SEN'] = train_val_metrics['sen']
                epoch_record['train_val_PRE'] = train_val_metrics['pre']

                val_metrics = validate(val_dataloader, model, criterion,
                                              device, epoch, args, 'val', '', threshold=train_val_metrics['threshold'])
                AUC_list.append(val_metrics['auc'])
                epoch_record['val_AUC'] = val_metrics['auc']
                epoch_record['val_ACC'] = val_metrics['acc']
                epoch_record['val_SEN'] = val_metrics['sen']
                epoch_record['val_PRE'] = val_metrics['pre']



                filename = (
                    f"../checkpoints/ck_student/ac{args.accum_steps}_ep{epoch}_wp{args.warmup_epochs}_sp{args.stable_epochs}_kp{args.kd_ramp_epochs}_nmp{args.n_masked_patch}_nt{args.n_token}_md{args.mask_drop}"
                    f"_wd{args.weight_decay}_lr{args.lr}_mb{args.accum_steps}"
                    f"_f{epoch_record['val_dif_AUC']:.2f}_v{epoch_record['val_AUC']:.2f}.pth")
                state = dict(epoch=epoch + 1, model=model.state_dict(),
                             optimizer=optimizer.state_dict())
                torch.save(state, filename)
                print('。。。')
                avg_auc_wsi_val = epoch_record['val_AUC']

                if avg_auc_wsi_val > max_auc_wsi_val:
                    max_auc_wsi_val = avg_auc_wsi_val
                    print(max_auc_wsi_val)
                    epoch_test = epoch
                    early_stop = 0



