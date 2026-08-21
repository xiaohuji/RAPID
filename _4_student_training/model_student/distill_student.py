import torch
import torch.nn as nn
import torch.nn.functional as F
from model.architecture.transformer import DimReduction, Attention_Gated, Classifier_1fc

class Student(nn.Module):

    def __init__(
        self,
        conf,
        D=128,
        droprate=0.2,
        n_token=1,
        n_masked_patch=0,
        mask_drop=0.0,
    ):
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

        self.slide_head = Classifier_1fc(conf.D_inner, conf.n_class, droprate)

    def _apply_attention_mask(self, A):
        """
        A: [K, N]
        """
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
        """
        A_logits: [K, N]
        x: [N, D]
        """
        A_prob = F.softmax(A_logits, dim=1)     # [K, N]
        token_feat = torch.mm(A_prob, x)        # [K, D]
        bag_A = A_prob.mean(0, keepdim=True)    # [1, N]
        bag_feat = torch.mm(bag_A, x)           # [1, D]
        return A_prob, token_feat, bag_feat

    def forward(self, wsi_feat, use_attention_mask=False):
        """
        wsi_feat: [1, N, 1024] or [N, 1024]
        """
        if wsi_feat.dim() == 3:
            x = wsi_feat[0]   # [N, 1024]
        else:
            x = wsi_feat

        x = self.dimreduction(x)      # [N, D]
        A_logits = self.attention(x)  # [K, N]

        if use_attention_mask:
            A_logits = self._apply_attention_mask(A_logits)

        _, token_feat, bag_feat = self._aggregate(A_logits, x)

        sub_preds = []
        for i, head in enumerate(self.classifier):
            sub_preds.append(head(token_feat[i]))
        sub_preds = torch.stack(sub_preds, dim=0)   # [K, 2]

        slide_logits = self.slide_head(bag_feat)    # [1, 2]

        return {
            "sub_preds": sub_preds,                 # [K, 2]
            "slide_logits": slide_logits,           # [1, 2]
            "attn_logits": A_logits.unsqueeze(0),   # [1, K, N]
            "bag_feat": bag_feat,                   # [1, D]
        }
