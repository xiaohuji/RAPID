import argparse

def get_parser():
    parser = argparse.ArgumentParser("WSI-only student distillation")

    # basic
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20262026)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=1)      # 建议固定 1
    parser.add_argument("--accum_steps", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)

    # optimization
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--stable_epochs', type=int, default=5)
    parser.add_argument('--kd_ramp_epochs', type=int, default=20)
    parser.add_argument('--kd_start_epoch', type=int, default=3)
    parser.add_argument("--proto_start_epoch", type=int, default=5)


    # model
    parser.add_argument("--D_feat", type=int, default=1024)
    parser.add_argument("--D_inner", type=int, default=512)
    parser.add_argument("--attn_hidden_dim", type=int, default=128)
    parser.add_argument("--n_class", type=int, default=2)
    parser.add_argument("--n_token", type=int, default=5)
    parser.add_argument("--drop_rate", type=float, default=0.2)
    parser.add_argument("--n_masked_patch", type=int, default=0)
    parser.add_argument("--mask_drop", type=float, default=0.0)

    # distill
    parser.add_argument("--distill_temperature", type=float, default=3.0)
    parser.add_argument("--lambda_ce", type=float, default=1.0)
    parser.add_argument("--lambda_sub", type=float, default=1.0)
    parser.add_argument("--lambda_diff", type=float, default=1.0)
    parser.add_argument("--lambda_kd", type=float, default=0.5)
    parser.add_argument("--lambda_attn", type=float, default=0.5)
    parser.add_argument("--lambda_feat", type=float, default=0.5)
    parser.add_argument("--lambda_anchor", type=float, default=0.0)
    parser.add_argument("--use_teacher_conf", action="store_true")
    parser.add_argument("--use_class_weight", action="store_true")

    # data columns
    parser.add_argument("--pid_col", type=str, default="pid")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--wsi_name_col", type=str, default="wsi_img_name")
    parser.add_argument("--model_name", type=str, default="uni")

    # paths
    # parser.add_argument("--train_wsi_csv", type=str, required=True)
    # parser.add_argument("--train_ct_pair_csv", type=str, default="")
    # parser.add_argument("--train_mri_pair_csv", type=str, default="")
    # parser.add_argument("--val_wsi_csv", type=str, default="")

    # parser.add_argument("--wsi_feature_root", type=str, required=True)
    # parser.add_argument("--ct_feature_root", type=str, default="")
    # parser.add_argument("--mri_feature_root", type=str, default="")

    parser.add_argument("--student_init_ckpt", type=str, default="")
    parser.add_argument("--teacher_ct_ckpt", type=str, default=r"D:\pyfile\RCC_Classify\rcc_code_BOTH\train_teacher\ck_acmil_wsi_ct_teacher_merlin_MJ_Vol160_lr_control_tumor_nd\sd10703181_ep9_Tm20_nmp50_nt5_md0.7_wd1e-05_lr0.0001_mb1_t1.00_v0.97_vp0.87.pth")
    parser.add_argument("--teacher_mri_ckpt", type=str, default=r"D:\pyfile\RCC_Classify\rcc_code_BOTH\train_teacher\ck_acmil_wsi_mri_teacher_merlin_DINO64_lr_control_tumor_nd\sd8316259_ep11_nmp200_nt5_md0.7_wd0.0001_mb8_ln0.0001_lw1e-05_lwb1e-05_t1.00_v0.96_vw0.96_vp0.84.pth")


    parser.add_argument("--ct_pair_dim", type=int, default=2048)
    parser.add_argument("--mri_pair_dim", type=int, default=1024)

    parser.add_argument("--save_dir", type=str, default="./ck_student_distill")

    parser.add_argument("--lambda_proto", type=float, default=0.0)
    parser.add_argument("--proto_momentum", type=float, default=0.9)

    parser.add_argument(
        "--proto_feat_source",
        type=str,
        default="bag_feat_fused",
        choices=["bag_feat_fused", "pair_proj_feat"]
    )

    args = parser.parse_args()

    if args.batch_size != 1:
        print("[Warning] 当前版本按 variable-length WSI bag 设计，建议 batch_size=1。已自动改为 1。")
        args.batch_size = 1

    if args.accum_steps < 1:
        args.accum_steps = 1

    # if args.pretrained_wsi_ckpt.strip() == "" and args.freeze_wsi_epochs > 0:
    #     print("[Warning] 未提供 pretrained WSI ckpt，freeze_wsi_epochs 自动改为 0。")
    #     args.freeze_wsi_epochs = 0

    return args