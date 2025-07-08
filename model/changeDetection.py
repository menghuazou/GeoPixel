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
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        
        # 视觉特征投影到文本维度
        self.visual_proj = nn.Linear(visual_dim, text_dim)
        self.text_proj = nn.Linear(text_dim, text_dim)
        
        # 注意力融合
        self.attn = nn.MultiheadAttention(text_dim, 8)
        
        # 输出投影
        self.output_proj = nn.Linear(text_dim, text_dim)

    def forward(self, visual_feat, text_feat):
        """
        Args:
            visual_feat: [B, C, H, W] 或 [B, H*W, C] 视觉特征
            text_feat: [N, seq_len, D] 文本特征
        Returns:
            fused_feat: [N, seq_len, D] 融合后的特征
        """
        # 处理视觉特征维度
        if visual_feat.dim() == 4:
            B, C, H, W = visual_feat.shape
            visual_feat = visual_feat.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        
        # 投影到相同维度
        visual_proj = self.visual_proj(visual_feat)  # [B, H*W, text_dim]
        text_proj = self.text_proj(text_feat)        # [N, seq_len, text_dim]
        
        # 注意力融合：文本特征作为Query，视觉特征作为Key和Value
        fused_feat, _ = self.attn(
            query=text_proj,      # [N, seq_len, text_dim]
            key=visual_proj,      # [B, H*W, text_dim] 
            value=visual_proj     # [B, H*W, text_dim]
        )
        
        # 输出投影
        fused_feat = self.output_proj(fused_feat)
        
        return fused_feat


# # 在变化检测中使用
# text_feat = torch.cat([change_features, seg_features], dim=1)
# fused_prompt = self.cross_modal_fusion(diff_feature, text_feat)