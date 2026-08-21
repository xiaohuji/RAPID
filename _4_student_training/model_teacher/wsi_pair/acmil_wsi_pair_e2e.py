import torch
import torch.nn as nn

class End2EndTeacher(nn.Module):
    def __init__(self, backbone, teacher_model, pair_adapter=None):
        super().__init__()
        self.backbone = backbone          # 3DINO / ViT
        self.teacher_model = teacher_model      # ACMIL_Teacher
        self.pair_adapter = pair_adapter        # 如果维度不一致，可加 Linear/MLP；一致就 None

    def _extract_pair_feat(self, mr_img):
        """
        mr_img: [B, 1, 112, 112, 64]
        return: [B, C]
        """
        feat = self.backbone(mr_img)

        # 兼容不同 backbone 输出形式
        if isinstance(feat, dict):
            if "x_norm_clstoken" in feat:
                feat = feat["x_norm_clstoken"]
            elif "cls_token" in feat:
                feat = feat["cls_token"]
            elif "feat" in feat:
                feat = feat["feat"]
            else:
                raise ValueError(f"Unknown dict keys from mr_backbone: {feat.keys()}")

        if isinstance(feat, (list, tuple)):
            feat = feat[0]

        # 若输出不是 [B, C]，做展平 / 池化
        if feat.ndim > 2:
            feat = feat.flatten(1)

        if self.pair_adapter is not None:
            feat = self.pair_adapter(feat)

        return feat

    def forward(self, wsi_feat, mr_img, return_pair_feat=False):
        pair_feat = self._extract_pair_feat(mr_img)
        out = self.teacher_model(wsi_feat, pair_feat)

        if return_pair_feat:
            out["pair_feat_online"] = pair_feat
        return out



class End2EndTeacherCT(nn.Module):
    """
    将 Merlin backbone 和 ACMIL_Teacher 串联，支持端到端前向和梯度回传。

    forward(wsi_feat, ct_img):
        1) ct_embed = backbone(ct_img)   # Merlin → embedding
        2) 如果 pair_adapter 不为 None，做维度映射
        3) out = teacher_model(wsi_feat, ct_embed)
        4) return out  (包含 slide_logits, wsi_logits, pair_logits 等)
    """

    def __init__(self, backbone, teacher_model, pair_adapter=None):
        super().__init__()
        self.backbone = backbone
        self.teacher_model = teacher_model
        self.pair_adapter = pair_adapter

    def _extract_pair_feat(self, ct_img):
        """
        从 Merlin 提取 CT 嵌入。

        Merlin(ImageEmbedding=True) 的输出形式取决于具体版本:
          - 若返回 tensor: 直接使用
          - 若返回 dict:   取 embed / image_embedding / cls_token 等

        ★ 请根据你的 Merlin 版本调整此处逻辑 ★
        """
        raw = self.backbone(ct_img)

        # —— 情况 1: 直接返回 tensor ——
        if isinstance(raw, torch.Tensor):
            embed = raw
        # —— 情况 2: 返回 dict ——
        elif isinstance(raw, dict):
            for key in ['image_embedding', 'embed', 'cls_token', 'x']:
                if key in raw:
                    embed = raw[key]
                    break
            else:
                raise KeyError(
                    f"Merlin 返回 dict 但找不到嵌入 key，"
                    f"可用 keys: {list(raw.keys())}"
                )
        # —— 情况 3: 返回 tuple ——
        elif isinstance(raw, (tuple, list)):
            embed = raw[0]
        else:
            raise TypeError(f"Merlin 返回了未知类型: {type(raw)}")

        # 确保形状为 (B, D) 或 (B, 1, D)
        if embed.dim() == 3 and embed.shape[1] == 1:
            embed = embed.squeeze(1)

        if self.pair_adapter is not None:
            embed = self.pair_adapter(embed)

        return embed

    def forward(self, wsi_feat, ct_img):
        ct_embed = self._extract_pair_feat(ct_img)
        out = self.teacher_model(wsi_feat, ct_embed)
        return out
