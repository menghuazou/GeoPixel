import torch
from torch import nn


class ChangeDetectionModule(nn.Module):
    def __init__(self, visual_dim):
        super().__init__()
        # 时序注意力模块
        self.temporal_attn = nn.MultiheadAttention(visual_dim, 8)

        # 变化特征提取
        self.change_extractor = nn.Sequential(
            nn.Conv2d(visual_dim * 2, visual_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(visual_dim, visual_dim, 3, padding=1)
        )

    def forward(self, feat_a, feat_b):
        # 重塑特征 (B, C, H, W) -> (B, H*W, C)
        b, c, h, w = feat_a.shape
        feat_a_flat = feat_a.view(b, c, -1).permute(0, 2, 1)
        feat_b_flat = feat_b.view(b, c, -1).permute(0, 2, 1)

        # 时序注意力
        attn_output, _ = self.temporal_attn(
            feat_b_flat, feat_a_flat, feat_a_flat
        )

        # 恢复空间结构
        attn_feat = attn_output.permute(0, 2, 1).view(b, c, h, w)

        # 差异特征提取
        diff_feature = self.change_extractor(
            torch.cat([feat_b, attn_feat], dim=1)
        )

        return diff_feature


# 交叉注意力融合
class CrossModalFusion(nn.Module):
    def __init__(self, visual_dim, text_dim):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, text_dim)
        self.text_proj = nn.Linear(text_dim, text_dim)
        self.attn = nn.MultiheadAttention(text_dim, 8)

    def forward(self, visual_feat, text_feat):
        # 投影到相同维度
        visual_proj = self.visual_proj(visual_feat)

        # 注意力融合
        fused_feat, _ = self.attn(
            self.text_proj(text_feat),
            visual_proj,
            visual_proj
        )
        return fused_feat


# # 在变化检测中使用
# text_feat = torch.cat([change_features, seg_features], dim=1)
# fused_prompt = self.cross_modal_fusion(diff_feature, text_feat)