import argparse

def get_parser():
    parser = argparse.ArgumentParser("WSI+MRI Teacher Training")

    # # -------------------------
    # # data path
    # # -------------------------
    # parser.add_argument("--wsi_csv", type=str, required=True,
    #                     help="WSI csv, 至少包含 pid / label / img_name")
    # parser.add_argument("--mri_csv", type=str, required=True,
    #                     help="MRI csv, 至少包含 pid / label")
    # parser.add_argument("--wsi_feature_root", type=str, required=True,
    #                     help="WSI h5 feature 根目录")
    # parser.add_argument("--mr_feature_root", type=str, required=True,
    #                     help="MRI pkl feature 根目录")
    # parser.add_argument("--save_dir", type=str, default="./teacher_wsi_mri_runs")

    # -------------------------
    # columns
    # -------------------------
    parser.add_argument("--pid_col", type=str, default="pid")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--wsi_img_col", type=str, default="img_name")
    parser.add_argument("--model_name", type=str, default="uni",
                        help="WSI feature 文件命名中的模型名，如 xxx__uni_tiles.h5ad")

    # -------------------------
    # training basic
    # -------------------------
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="当前脚本按 batch_size=1 设计，以兼容 variable-length WSI bag")
    parser.add_argument("--accum_steps", type=int, default=8,
                        help="梯度累积步数，batch_size=1 时可模拟更大 batch")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Windows + h5py 推荐先用 0")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20262026)
    parser.add_argument("--early_stop_patience", type=int, default=8)

    # -------------------------
    # optimization
    # -------------------------
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--Tmax", type=int, default=20)
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--use_class_weight", action="store_true",
                        help="是否给 CE 加类别权重")
    parser.add_argument("--drop_rate", type=float, default=0.2)

    # -------------------------
    # WSI / MRI dims
    # -------------------------
    parser.add_argument("--D_feat", type=int, default=1024,
                        help="WSI patch feature dim")
    parser.add_argument("--D_inner", type=int, default=512,
                        help="ACMIL 内部维度")
    parser.add_argument("--pair_dim", type=int, default=1024,
                        help="MRI embedding dim")
    parser.add_argument("--n_class", type=int, default=2)

    # -------------------------
    # ACMIL settings
    # -------------------------
    parser.add_argument("--n_token", type=int, default=5)
    parser.add_argument("--n_masked_patch", type=int, default=50)
    parser.add_argument("--mask_drop", type=float, default=0.5)
    parser.add_argument("--attn_hidden_dim", type=int, default=128,
                        help="Attention_Gated 中间维度 D")

    # -------------------------
    # teacher loss weights
    # -------------------------
    parser.add_argument("--lambda_wsi", type=float, default=0.5)
    parser.add_argument("--lambda_pair", type=float, default=1)
    parser.add_argument("--lambda_sub", type=float, default=0.5)
    parser.add_argument("--lambda_diff", type=float, default=0.1)

    # -------------------------
    # staged training
    # -------------------------
    parser.add_argument("--pretrained_wsi_ckpt", type=str, default="",
                        help="单模态 WSI 最优 ckpt，可选，建议提供")
    parser.add_argument("--freeze_wsi_epochs", type=int, default=3,
                        help="前几个 epoch 冻结原 WSI 主干，仅训 MRI/fusion 新模块")
    parser.add_argument("--selection_wsi_weight", type=float, default=0.3,
                        help="best teacher 选模分数 = val_fuse_auc + w * val_wsi_auc")

    # -------------------------
    # optional export teacher targets
    # -------------------------
    parser.add_argument("--export_teacher_targets", action="store_true")
    parser.add_argument("--teacher_targets_name", type=str, default="teacher_targets_full.pt")

    parser.add_argument("--lr_wsi_backbone", type=float, default=2e-5)
    parser.add_argument("--lr_wsi_head", type=float, default=5e-5)
    parser.add_argument("--lr_new", type=float, default=1e-4)

    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_start_factor", type=float, default=0.1)

    parser.add_argument("--grad_clip", type=float, default=1.0)

    # parser.add_argument("--pretrained_wsi_ckpt", type=str, default="")
    # parser.add_argument("--`accum_steps`", type=int, default=1)

    args = parser.parse_args()

    if args.batch_size != 1:
        print("[Warning] 当前版本按 variable-length WSI bag 设计，建议 batch_size=1。已自动改为 1。")
        args.batch_size = 1

    if args.accum_steps < 1:
        args.accum_steps = 1

    if args.pretrained_wsi_ckpt.strip() == "" and args.freeze_wsi_epochs > 0:
        print("[Warning] 未提供 pretrained WSI ckpt，freeze_wsi_epochs 自动改为 0。")
        args.freeze_wsi_epochs = 0

    return args