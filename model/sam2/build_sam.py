# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

import os
# 注意os.environ得在import huggingface库相关语句之前执行。
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def build_sam2(
    config_file, # 'sam2_hiera_l.yaml'
    ckpt_path=None, # '/root/autodl-tmp/models/sam2/sam2_hiera_large.pt'
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra) # 这里就指向了yaml文件：model/sam2_configs/sam2_hiera_l.yaml
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    #model = model.to(device)
    if mode == "eval":
        model.eval()
    return model  #SAM2Base( (image_encoder): ImageEncoder( (trunk): Hiera( (patch_embed): PatchEmbed( (proj): Conv2d(3, 144, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3)) ) (blocks): ModuleList( (0-1): 2 x MultiScaleBlock( (norm1): LayerNorm((144,), eps=1e-06, elementwise_affine=True) (attn): MultiScaleAttention( (qkv): Linear(in_features=144, out_features=432, bias=True) (proj): Linear(in_features=144, out_features=144, bias=True) ) (drop_path): Identity() (norm2): LayerNorm((144,), eps=1e-06, elementwise_affine=True) (mlp): MLP( (layers): ModuleList( (0): Linear(in_features=144, out_features=576, bias=True) (1): Linear(in_features=576, out_features=144, bias=True) ) (act): GELU(approximate='none') ) ) (2): MultiScaleBlock( (norm1): LayerNorm((144,), eps=1e-06, elementwise_affine=True) (pool): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False) (attn): MultiScaleAttention( (q_pool): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False) (qkv): Linear(in_features=144, out_features=864, bias=True) (proj): Linear(in_features=288, out_features=288, bias=True) ) (drop_path): Identity() (norm2): LayerNorm((288,), eps=1e-06, elementwise_affine=True) (mlp): MLP( (layers): ModuleList( (0): Linear(in_features=288, out_features=1152, bias=True) (1): Linear(in_features=1152, out_features=288, bias=True) ) (act): GELU(approximate='none') ) (proj): Linear(in_features=144, out_features=288, bias=True) ) (3-7): 5 x MultiScaleBlock( (norm1): LayerNorm((288,), eps=1e-06, elementwise_affine=True) (attn): MultiScaleAttention( (qkv): Linear(in_features=288, out_features=864, bias=True) (proj): Linear(in_features=288, out_features=288, bias=True) ) (drop_path): Identity() (norm2): LayerNorm((288,), eps=1e-06, elementwise_affine=True) (mlp): MLP( (layers): ModuleList( (0): Linear(in_features=288, out_features=1152, bias=True) (1): Linear(in_features=1152, out_features=288, bias=True) ) (act): GELU(approximate='none') ) ) (8): MultiScaleBlock( (norm1): LayerNorm((288,), eps=1e-06, elementwise_affine=True) (pool): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False) (attn): MultiScaleAttention( (q_pool): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False) (qkv): Linear(in_features=288, out_features=1728, bias=True) (proj): Linear(in_features=576, out_features=576, bias=True) ) (drop_path): Identity() (norm2): LayerNorm((576,), eps=1e-06, elementwise_affine=True) (mlp): MLP( (layers): ModuleList( (0): Linear(in_features=576, out_features=2304, bias=True) (1): Linear(in_features=2304, out_features=576, bias=True) ) (act): GELU(approximate='none') ) (proj): Linear(in_features=288, out_features=576, bias=True) ) (9-43): 35 x MultiScaleBlock( (norm1): LayerNorm((576,), eps=1e-06, elementwise_affine=True) (attn): MultiScaleAttention( (qkv): Linear(in_features=576, out_features=1728, bias=True) (proj): Linear(in_features=576, out_features=576, bias=True) ) (drop_path): Identity() (norm2): LayerNorm((576,), eps=1e-06, elementwise_affine=True) (mlp): MLP( (layers): ModuleList( (0): Linear(in_features=576, out_features=2304, bias=True) (1): Linear(in_features=2304, out_features=576, bias=True) ) (act): GELU(approximate='none') ) ) (44): MultiScaleBlock( (norm1): LayerNorm((576,), eps=1e-06, elementwise_affine=True) (pool): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False) (attn): MultiScaleAttention( (q_pool): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False) (qkv): Linear(in_features=576, out_features=3456, bias=True) (proj): Linear(in_features=1152, out_features=1152, bias=True) ) (drop_path): Identity() (norm2): LayerNorm((1152,), eps=1e-06, elementwise_affine=True) (mlp): MLP( (layers): ModuleList( (0): Linear(in_features=1152, out_features=4608, bias=True) (1): Linear(in_features=4608, out_features=1152, bias=True) ) (act): GELU(approximate='none') ) (proj): Linear(in_features=576, out_features=1152, bias=True) ) (45-47): 3 x MultiScaleBlock( (norm1): LayerNorm((1152,), eps=1e-06, elementwise_affine=True) (attn): MultiScaleAttention( (qkv): Linear(in_features=1152, out_features=3456, bias=True) (proj): Linear(in_features=1152, out_features=1152, bias=True) ) (drop_path): Identity() (norm2): LayerNorm((1152,), eps=1e-06, elementwise_affine=True) (mlp): MLP( (layers): ModuleList( (0): Linear(in_features=1152, out_features=4608, bias=True) (1): Linear(in_features=4608, out_features=1152, bias=True) ) (act): GELU(approximate='none') ) ) ) ) (neck): FpnNeck( (position_encoding): PositionEmbeddingSine() (convs): ModuleList( (0): Sequential( (conv): Conv2d(1152, 256, kernel_size=(1, 1), stride=(1, 1)) ) (1): Sequential( (conv): Conv2d(576, 256, kernel_size=(1, 1), stride=(1, 1)) ) (2): Sequential( (conv): Conv2d(288, 256, kernel_size=(1, 1), stride=(1, 1)) ) (3): Sequential( (conv): Conv2d(144, 256, kernel_size=(1, 1), stride=(1, 1)) ) ) ) ) (mask_downsample): Conv2d(1, 1, kernel_size=(4, 4), stride=(4, 4)) (memory_attention): MemoryAttention( (layers): ModuleList( (0-3): 4 x MemoryAttentionLayer( (self_attn): RoPEAttention( (q_proj): Linear(in_features=256, out_features=256, bias=True) (k_proj): Linear(in_features=256, out_features=256, bias=True) (v_proj): Linear(in_features=256, out_features=256, bias=True) (out_proj): Linear(in_features=256, out_features=256, bias=True) ) (cross_attn_image): RoPEAttention( (q_proj): Linear(in_features=256, out_features=256, bias=True) (k_proj): Linear(in_features=64, out_features=256, bias=True) (v_proj): Linear(in_features=64, out_features=256, bias=True) (out_proj): Linear(in_features=256, out_features=256, bias=True) ) (linear1): Linear(in_features=256, out_features=2048, bias=True) (dropout): Dropout(p=0.1, inplace=False) (linear2): Linear(in_features=2048, out_features=256, bias=True) (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True) (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True) (norm3): LayerNorm((256,), eps=1e-05, elementwise_affine=True) (dropout1): Dropout(p=0.1, inplace=False) (dropout2): Dropout(p=0.1, inplace=False) (dropout3): Dropout(p=0.1, inplace=False) ) ) (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True) ) (memory_encoder): MemoryEncoder( (mask_downsampler): MaskDownSampler( (encoder): Sequential( (0): Conv2d(1, 4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) (1): LayerNorm2d() (2): GELU(approximate='none') (3): Conv2d(4, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) (4): LayerNorm2d() (5): GELU(approximate='none') (6): Conv2d(16, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) (7): LayerNorm2d() (8): GELU(approximate='none') (9): Conv2d(64, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) (10): LayerNorm2d() (11): GELU(approximate='none') (12): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1)) ) ) (pix_feat_proj): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1)) (fuser): Fuser( (proj): Identity() (layers): ModuleList( (0-1): 2 x CXBlock( (dwconv): Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=256) (norm): LayerNorm2d() (pwconv1): Linear(in_features=256, out_features=1024, bias=True) (act): GELU(approximate='none') (pwconv2): Linear(in_features=1024, out_features=256, bias=True) (drop_path): Identity() ) ) ) (position_encoding): PositionEmbeddingSine() (out_proj): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1)) ) (sam_prompt_encoder): PromptEncoder( (pe_layer): PositionEmbeddingRandom() (point_embeddings): ModuleList( (0-3): 4 x Embedding(1, 256) ) (not_a_point_embed): Embedding(1, 256) (mask_downscaling): Sequential( (0): Conv2d(1, 4, kernel_size=(2, 2), stride=(2, 2)) (1): LayerNorm2d() (2): GELU(approximate='none') (3): Conv2d(4, 16, kernel_size=(2, 2), stride=(2, 2)) (4): LayerNorm2d() (5): GELU(approximate='none') (6): Conv2d(16, 256, kernel_size=(1, 1), stride=(1, 1)) ) (no_mask_embed): Embedding(1, 256) ) (sam_mask_decoder): MaskDecoder( (transformer): TwoWayTransformer( (layers): ModuleList( (0-1): 2 x TwoWayAttentionBlock( (self_attn): Attention( (q_proj): Linear(in_features=256, out_features=256, bias=True) (k_proj): Linear(in_features=256, out_features=256, bias=True) (v_proj): Linear(in_features=256, out_features=256, bias=True) (out_proj): Linear(in_features=256, out_features=256, bias=True) ) (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True) (cross_attn_token_to_image): Attention( (q_proj): Linear(in_features=256, out_features=128, bias=True) (k_proj): Linear(in_features=256, out_features=128, bias=True) (v_proj): Linear(in_features=256, out_features=128, bias=True) (out_proj): Linear(in_features=128, out_features=256, bias=True) ) (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True) (mlp): MLP( (layers): ModuleList( (0): Linear(in_features=256, out_features=2048, bias=True) (1): Linear(in_features=2048, out_features=256, bias=True) ) (act): ReLU() ) (norm3): LayerNorm((256,), eps=1e-05, elementwise_affine=True) (norm4): LayerNorm((256,), eps=1e-05, elementwise_affine=True) (cross_attn_image_to_token): Attention( (q_proj): Linear(in_features=256, out_features=128, bias=True) (k_proj): Linear(in_features=256, out_features=128, bias=True) (v_proj): Linear(in_features=256, out_features=128, bias=True) (out_proj): Linear(in_features=128, out_features=256, bias=True) ) ) ) (final_attn_token_to_image): Attention( (q_proj): Linear(in_features=256, out_features=128, bias=True) (k_proj): Linear(in_features=256, out_features=128, bias=True) (v_proj): Linear(in_features=256, out_features=128, bias=True) (out_proj): Linear(in_features=128, out_features=256, bias=True) ) (norm_final_attn): LayerNorm((256,), eps=1e-05, elementwise_affine=True) ) (iou_token): Embedding(1, 256) (mask_tokens): Embedding(4, 256) (obj_score_token): Embedding(1, 256) (output_upscaling): Sequential( (0): ConvTranspose2d(256, 64, kernel_size=(2, 2), stride=(2, 2)) (1): LayerNorm2d() (2): GELU(approximate='none') (3): ConvTranspose2d(64, 32, kernel_size=(2, 2), stride=(2, 2)) (4): GELU(approximate='none') ) (conv_s0): Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1)) (conv_s1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1)) (output_hypernetworks_mlps): ModuleList( (0-3): 4 x MLP( (layers): ModuleList( (0-1): 2 x Linear(in_features=256, out_features=256, bias=True) (2): Linear(in_features=256, out_features=32, bias=True) ) (act): ReLU() ) ) (iou_prediction_head): MLP( (layers): ModuleList( (0-1): 2 x Linear(in_features=256, out_features=256, bias=True) (2): Linear(in_features=256, out_features=4, bias=True) ) (act): ReLU() ) (pred_obj_score_head): MLP( (layers): ModuleList( (0-1): 2 x Linear(in_features=256, out_features=256, bias=True) (2): Linear(in_features=256, out_features=1, bias=True) ) (act): ReLU() ) ) (obj_ptr_proj): MLP( (layers): ModuleList( (0-2): 3 x Linear(in_features=256, out_features=256, bias=True) ) (act): ReLU() ) (obj_ptr_tpos_proj): Identity() )


def build_sam2_video_predictor(
    config_file,
    ckpt_path=None,
    device=None,
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):
    hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
    ]
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_hf(model_id, **kwargs):

    from huggingface_hub import hf_hub_download

    model_id_to_filenames = {
        "facebook/sam2-hiera-tiny": ("sam2_hiera_t.yaml", "sam2_hiera_tiny.pt"),
        "facebook/sam2-hiera-small": ("sam2_hiera_s.yaml", "sam2_hiera_small.pt"),
        "facebook/sam2-hiera-base-plus": (
            "sam2_hiera_b+.yaml",
            "sam2_hiera_base_plus.pt",
        ),
        "facebook/sam2-hiera-large": ("sam2_hiera_l.yaml", "sam2_hiera_large.pt"),
    }
    config_name, checkpoint_name = model_id_to_filenames[model_id]
    ckpt_path = hf_hub_download(
        repo_id=model_id,
        filename=checkpoint_name,
        resume_download=True,
        local_dir="/root/autodl-tmp/models/sam2",
    )
    return build_sam2(config_file=config_name, ckpt_path=ckpt_path, **kwargs)


def build_sam2_video_predictor_hf(model_id, **kwargs):

    from huggingface_hub import hf_hub_download

    model_id_to_filenames = {
        "facebook/sam2-hiera-tiny": ("sam2_hiera_t.yaml", "sam2_hiera_tiny.pt"),
        "facebook/sam2-hiera-small": ("sam2_hiera_s.yaml", "sam2_hiera_small.pt"),
        "facebook/sam2-hiera-base-plus": (
            "sam2_hiera_b+.yaml",
            "sam2_hiera_base_plus.pt",
        ),
        "facebook/sam2-hiera-large": ("sam2_hiera_l.yaml", "sam2_hiera_large.pt"),
    }
    config_name, checkpoint_name = model_id_to_filenames[model_id]
    ckpt_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)
    return build_sam2_video_predictor(
        config_file=config_name, ckpt_path=ckpt_path, **kwargs
    )


def _load_checkpoint(model, ckpt_path):
    # model.load_state_dict() 方法实现权重加载
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["model"]
        renamed_sd = {k.replace("gamma", "weight") if "gamma" in k else k: v for k, v in sd.items()}
        missing_keys, unexpected_keys = model.load_state_dict(renamed_sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")