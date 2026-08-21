import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.architecture.transformer import DimReduction, Attention_Gated, Classifier_1fc

class MLPHead(nn.Module):
    def __init__(self, in_dim, out_dim=2, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class Teacher(nn.Module):
    def __init__(self, conf, D=128, droprate=0.2,
                 pair_dim=1024, n_token=1, n_masked_patch=0, mask_drop=0):
        super().__init__()

        self.d_inner = conf.D_inner
        self.n_token = n_token
        self.n_masked_patch = n_masked_patch
        self.mask_drop = mask_drop

        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.attention = Attention_Gated(conf.D_inner, D, n_token)

        self.classifier = nn.ModuleList()
        for _ in range(n_token):
            self.classifier.append(
                Classifier_1fc(conf.D_inner, conf.n_class, droprate)
            )

        self.wsi_head = Classifier_1fc(conf.D_inner, conf.n_class, droprate)

        # ---- pair branch ----
        self.pair_proj = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, conf.D_inner),
            nn.GELU(),
            nn.Dropout(droprate),
        )

        self.pair_head = MLPHead(
            in_dim=conf.D_inner,
            out_dim=conf.n_class,
            hidden_dim=256,
            dropout=droprate
        )

        # ---- pair-conditioned attention ----
        self.patch_key = nn.Linear(conf.D_inner, conf.D_inner, bias=False)
        self.pair_to_token = nn.Linear(conf.D_inner, conf.D_inner * n_token, bias=False)

        self.cross_scale = nn.Parameter(torch.tensor(1.0))

        # ---- final fusion head ----
        self.fusion_head = MLPHead(
            in_dim=conf.D_inner * 3,
            out_dim=conf.n_class,
            hidden_dim=conf.D_inner,
            dropout=droprate
        )

    def _apply_attention_mask(self, A):
        # A: [K, N]
        if self.n_masked_patch > 0 and self.training:
            k, n = A.shape
            n_masked_patch = min(self.n_masked_patch, n)
            if n_masked_patch > 0:
                _, indices = torch.topk(A, n_masked_patch, dim=-1)

                n_drop = int(n_masked_patch * self.mask_drop)
                if n_drop > 0:
                    rand_selected = torch.argsort(
                        torch.rand(*indices.shape, device=A.device), dim=-1
                    )[:, :n_drop]

                    masked_indices = indices[
                        torch.arange(indices.shape[0], device=A.device).unsqueeze(-1),
                        rand_selected
                    ]

                    random_mask = torch.ones(k, n, device=A.device)
                    random_mask.scatter_(-1, masked_indices, 0)
                    A = A.masked_fill(random_mask == 0, -1e9)
        return A

    def _aggregate(self, A_logits, x):
        # A_logits: [K, N], x: [N, D]
        A_prob = F.softmax(A_logits, dim=1)       # [K, N]
        token_feat = torch.mm(A_prob, x)          # [K, D]
        bag_A = A_prob.mean(0, keepdim=True)      # [1, N]
        bag_feat = torch.mm(bag_A, x)             # [1, D]
        return A_prob, token_feat, bag_feat

    def forward(self, wsi_feat, pair_feat):
        """
        wsi_feat: [1, N, 1024] or [N, 1024]
        pair_feat:  [1, 1024] or [1024]
        """
        # ---- shape normalize ----
        if wsi_feat.dim() == 3:
            x = wsi_feat[0]   # [N, 1024]
        else:
            x = wsi_feat

        if pair_feat.dim() == 1:
            m = pair_feat.unsqueeze(0)   # [1, 1024]
        else:
            m = pair_feat

        # ---- WSI encoder ----
        x = self.dimreduction(x)   # [N, D]
        A_base = self.attention(x) # [K, N]

        # ---- pair encoder ----
        z_m = self.pair_proj(m)      # [1, D]

        # ---- pair-conditioned attention ----
        patch_keys = self.patch_key(x)  # [N, D]
        pair_tokens = self.pair_to_token(z_m).view(self.n_token, self.d_inner)  # [K, D]

        A_cross = torch.matmul(pair_tokens, patch_keys.t()) / math.sqrt(self.d_inner)  # [K, N]
        A_fused = A_base + self.cross_scale * A_cross
        A_fused = self._apply_attention_mask(A_fused)

        # ---- teacher branch: pair-guided WSI aggregation ----
        A_fused_prob, token_feat_fused, bag_feat_fused = self._aggregate(A_fused, x)

        # token-level preds（沿用你原本 ACMIL 逻辑）
        sub_preds = []
        for i, head in enumerate(self.classifier):
            sub_preds.append(head(token_feat_fused[i]))
        sub_preds = torch.stack(sub_preds, dim=0)   # [K, 2]

        # ---- pure WSI auxiliary branch ----
        _, _, bag_feat_wsi = self._aggregate(A_base, x)
        wsi_logits = self.wsi_head(bag_feat_wsi)   # [1, 2]

        # ---- pure pair auxiliary branch ----
        pair_logits = self.pair_head(z_m)              # [1, 2]

        # ---- final fusion ----
        fusion_feat = torch.cat(
            [bag_feat_fused, z_m, bag_feat_fused * z_m], dim=-1
        )  # [1, 3D]

        slide_logits = self.fusion_head(fusion_feat)   # [1, 2]

        return {
            "sub_preds": sub_preds,                  # [K, 2]
            "slide_logits": slide_logits,            # [1, 2]
            "wsi_logits": wsi_logits,                # [1, 2]
            "pair_logits": pair_logits,                  # [1, 2]
            "attn_fused_logits": A_fused.unsqueeze(0),  # [1, K, N]
            "attn_base_logits": A_base.unsqueeze(0),    # [1, K, N]
            "bag_feat_fused": bag_feat_fused,        # [1, D]
            "bag_feat_wsi": bag_feat_wsi,            # [1, D]
            "pair_proj_feat": z_m,                     # [1, D]
            "fusion_feat": fusion_feat,              # [1, 3D]
        }

    def forward_wsi_only(self, wsi_feat, use_attention_mask=False):
        # 可选：后面拿来混入 WSI-only 样本做 regularization
        if wsi_feat.dim() == 3:
            x = wsi_feat[0]
        else:
            x = wsi_feat

        x = self.dimreduction(x)
        A_base = self.attention(x)

        if use_attention_mask:
            A_base = self._apply_attention_mask(A_base)

        A_prob, token_feat, bag_feat = self._aggregate(A_base, x)

        sub_preds = []
        for i, head in enumerate(self.classifier):
            sub_preds.append(head(token_feat[i]))
        sub_preds = torch.stack(sub_preds, dim=0)

        slide_logits = self.wsi_head(bag_feat)

        return {
            "sub_preds": sub_preds,
            "slide_logits": slide_logits,
            "attn_logits": A_base.unsqueeze(0),
            "bag_feat": bag_feat,
        }
