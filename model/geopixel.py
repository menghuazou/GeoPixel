from typing import List, Optional, Tuple, Union

import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from model.IXC.modeling_internlm_xcomposer2 import InternLMXComposer2ForCausalLM
from model.IXC.modeling_internlm2 import InternLM2Model
from model.changeDetection import ChangeDetectionModule, CrossModalFusion
from model.sam2.build_sam import build_sam2_hf
from model.sam2.utils.transforms import SAM2Transforms
from transformers import TextStreamer
try:
    from transformers.generation.streamers import BaseStreamer
except:  # noqa # pylint: disable=bare-except
    BaseStreamer = None


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


class GeoPixelMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(GeoPixelMetaModel, self).__init__(config)
        self.config = config
        self.config.train_mask_decoder = getattr(self.config, "train_mask_decoder", kwargs.get("train_mask_decoder", False)) # 是否训练掩码解码器
        self.config.out_dim = getattr(self.config, "out_dim", kwargs.get("out_dim", 256)) # 输出维度256
        self.vision_pretrained = kwargs.get("vision_pretrained", None) # 视觉预训练模型的路径或标识
        self.initialize_geopixel_modules(self.config)

    def initialize_geopixel_modules(self, config):
        # 初始化与地理像素处理相关的视觉模型、图像变换、特征图维度设置以及文本投影层; 基于sam2
        # grounding vision model
        self.visual_model = build_sam2_hf(self.vision_pretrained)

        self._transform = SAM2Transforms(
                    resolution=self.visual_model.image_size,
                    mask_threshold=0.0,
                    max_hole_area=0.0,
                    max_sprinkle_area=0.0,
                )
        # Spatial dim for backbone feature maps
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ] # 骨干特征图的空间维度

        # 预训练视觉知识冻结，在反向传播时不进行梯度更新
        for param in self.visual_model.parameters():
            param.requires_grad = False

        if config.train_mask_decoder:
            self.visual_model.sam_mask_decoder.train()
            # 条件性地训练掩码解码器
            for param in self.visual_model.sam_mask_decoder.parameters():
                param.requires_grad = True

        # text projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_projection_layers = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_projection_layers)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class GeoPixelModel(GeoPixelMetaModel, InternLM2Model):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(GeoPixelModel, self).__init__(config, **kwargs)
        self.config.use_cache = False


class GeoPixelForCausalLM(InternLMXComposer2ForCausalLM):
    def __init__(self,config,**kwargs,):
        
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        self.seg_token_idx = kwargs.pop("seg_token_idx")
        self.change_token_idx = kwargs.pop("change_token_idx", 250001)  # 默认值

        self.change_detector = ChangeDetectionModule(visual_dim=256)
        self.cross_modal_fusion = CrossModalFusion(visual_dim=256, text_dim=256)

        super().__init__(config)
        self.model = GeoPixelModel(config, **kwargs)
        self.vocab_size = config.vocab_size
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def encode_g_img(self, image):
        """
        用于处理图像输入（无论是文件路径还是Tensor），通过预处理和模型计算，得到图像的嵌入表示，并返回这些嵌入以及图像的原始尺寸信息。
        Calculates the image embeddings for the provided image
        Arguments:
          image (np.ndarray or str)
        """
        if image is None:
            return None
        if isinstance(image, str):
            _, ext = os.path.splitext(image)
            if ext.lower() in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp','.tif'}:
                image = Image.open(image)
                w, h = image.size
                _orig_hw = [(h, w)] 
            else:
                print ('Unknow input format', image)
                return None
        else:
            assert isinstance(image, torch.Tensor)
            _orig_hw = [image.shape[:2]]
        image = self.model._transform(image)
        image = image[None, ...].to(self.device)
        assert ( len(image.shape) == 4 and image.shape[1] == 3), f"image must be of size 1x3xHxW, got {image.shape}"
        features = self.get_visual_embs(image)   
        return features,_orig_hw

    def get_visual_embs(self, img_batch: torch.FloatTensor):
        # 从输入图像中提取核心特征
        with torch.no_grad():
            torch.cuda.empty_cache() # 清空缓存
            img_batch = img_batch.to(self.device)
            batch_size = img_batch.shape[0]
            assert (
                len(img_batch.shape) == 4 and img_batch.shape[1] == 3
            ), f"grounding_img_batch must be of size Bx3xHxW, got {img_batch.shape}"
            backbone_out = self.model.visual_model.forward_image(img_batch) # 通过sam获取图像特征
            _, vision_feats, _, _ = self.model.visual_model._prepare_backbone_features(backbone_out) # 多尺度特征
            if self.model.visual_model.directly_add_no_mem_embed:
                vision_feats[-1] = vision_feats[-1] + self.model.visual_model.no_mem_embed
            feats = [
                feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
                for feat, feat_size in zip(vision_feats[::-1], self.model._bb_feat_sizes[::-1])
            ][::-1]
            features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        return features
    
    def forward(self, **kwargs):
        return super().forward(**kwargs) if "past_key_values" in kwargs else self.model_forward(**kwargs)
    
    def model_forward(
            self,
            inference: bool = False, #推理模式
            **kwargs,
    ):
        samples = kwargs.get('samples', None)
        if samples and samples['data_type'][0] == 'grounding': 
            kwargs['output_hidden_states'] = True
            kwargs['use_cache'] = False

            torch.cuda.empty_cache()
            outputs = super().forward(**kwargs)

            if inference:
                assert len(samples['text_input']) == 1 and len(samples['image'][0]) == 1 #single image and single query
                output_hidden_states = [outputs.hidden_states]
                outputs = None
            else:
                output_hidden_states = outputs.hidden_states

            hidden_states = []
            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))
            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)

            seg_token_mask = outputs.seg_token_mask
            pred_embeddings = [states[masks] for states, masks in zip(last_hidden_state, seg_token_mask)]
            
           
            change_token_mask = getattr(outputs, 'change_token_mask', None)
            change_embeddings = []
            if change_token_mask is not None:
                change_embeddings = [states[masks] for states, masks in zip(last_hidden_state, change_token_mask)]
            
            image_g_batch = torch.cat(samples['image_g'][0],dim = 0)

            image_g_features = self.get_visual_embs(image_g_batch)
            ori_hw = samples['ori_hw'][0]
            all_pred_masks = []
            all_change_masks = []  
            
            for i in range(len(pred_embeddings)): #(bs,)
                if (pred_embeddings[i].numel()== 0):
                    pred_masks.append([])
                    continue
                (sparse_embeddings, dense_embeddings,) = self.model.visual_model.sam_prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )
                batch_mode = (pred_embeddings[i].shape[0]>1)
                high_res_features = [
                    feat_level[i].unsqueeze(0)
                    for feat_level in image_g_features["high_res_feats"]
                ]
                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                image_g_embeds = image_g_features['image_embed'][i].unsqueeze(0).to(torch.bfloat16)
                low_res_masks, _, _ , _ = self.model.visual_model.sam_mask_decoder(
                    image_embeddings=image_g_embeds,
                    image_pe=self.model.visual_model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    repeat_image=batch_mode,
                    multimask_output=False,
                    high_res_features=high_res_features,
                )
                pred_masks = self.model._transform.postprocess_masks(
                    low_res_masks,
                    ori_hw[i],
                )
                all_pred_masks.append(pred_masks[:, 0])

                if change_embeddings and change_embeddings[i].numel() > 0:
                    all_change_masks.append([])
                else:
                    all_change_masks.append([])
                

            model_output = outputs
            gt_masks =  samples['masks'][0]
            pred_masks = all_pred_masks 

            if inference:
                return {
                    "pred_masks": pred_masks,
                    "gt_masks": gt_masks,
                    "change_masks": all_change_masks,  # 添加变化掩码输出
                }

            ce_loss = model_output.loss
            ce_loss = ce_loss * self.ce_loss_weight
            mask_bce_loss = 0
            mask_dice_loss = 0
            num_masks = 0

            for batch_idx in range(len(pred_masks)): # for every image
                cur_gt_masks = torch.stack(
                    [
                        torch.from_numpy(gt_mask).to(dtype=pred_masks[batch_idx].dtype, device=pred_masks[batch_idx].device)
                        for gt_mask in gt_masks[batch_idx]
                    ],
                    dim=0
                ) # expected (bs,H,W)
                cur_pred_masks = pred_masks[batch_idx]
                assert (
                    cur_gt_masks.shape[0] == cur_pred_masks.shape[0]
                ), "gt_masks.shape: {}, pred_masks.shape: {}".format(
                    cur_gt_masks.shape, cur_pred_masks.shape
                )
                mask_bce_loss += (
                    sigmoid_ce_loss(cur_pred_masks, cur_gt_masks, num_masks=cur_gt_masks.shape[0])
                    * cur_gt_masks.shape[0]
                )
                mask_dice_loss += (
                    dice_loss(cur_pred_masks, cur_gt_masks, num_masks=cur_gt_masks.shape[0])
                    * cur_gt_masks.shape[0]
                )
                num_masks += cur_gt_masks.shape[0] 

            mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
            mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
            mask_loss = mask_bce_loss + mask_dice_loss

            loss = ce_loss + mask_loss
            outputs = CausalLMOutputWithPast(
                loss=loss,
                logits=model_output.logits,
                past_key_values=model_output.past_key_values,
                hidden_states=output_hidden_states,
                attentions=model_output.attentions,
            )
            outputs.ce_loss = ce_loss
            outputs.mask_bce_loss = mask_bce_loss
            outputs.mask_dice_loss = mask_dice_loss
            outputs.mask_loss = mask_loss
        else: 
            outputs =  super().forward(**kwargs)
        return outputs

    def evaluate(
            self,
            tokenizer,
            query: str,
            images: Union[
                List[Union[str, Tuple[str, str]]],
                Tuple[Union[str, Tuple[str, str]], Union[str, Tuple[str, str]]]
            ] = [],
            change_detection: bool = False,
            hd_num: int = 9,
            history: List[Tuple[str, str]] = [],
            max_new_tokens: int = 1024,
            stream: bool = False,
            **kwargs,
    ):
        """
        增强的评估函数，支持单图像分割和双图像变化检测
        """
        # 添加输入验证
        if change_detection:
            if len(images) != 2:
                raise ValueError(f"变化检测需要2个图像输入，当前提供{len(images)}个")
            if not isinstance(images, (list, tuple)):
                raise ValueError("变化检测的图像输入必须是列表或元组格式")
        
        with torch.no_grad():
            # 1. 输入预处理
            inputs, im_mask, _ = self.interleav_wrap_chat(
                query, images, history=history, hd_num=hd_num, change_detection=change_detection
            )
            inputs = {
                k: v.to(self.device)
                for k, v in inputs.items() if torch.is_tensor(v)
            }
            eos_token_id = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids(['[UNUSED_TOKEN_145]'])[0]
            ]
            all_pred_masks = []  # 存储分割掩码
            all_change_masks = []  # 存储变化掩码

            # 流式输出设置
            if stream:
                streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            else:
                streamer = None

            # 2. 模型生成文本响应; model.generate()
            outputs = self.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                im_mask=im_mask,
                input_ids=None,
                streamer=streamer,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                eos_token_id=eos_token_id,
                repetition_penalty=1.0,
                infer_mode='base',
                output_hidden_states=True,
                return_dict_in_generate=True,
                **kwargs,
            )

            # 3. 解码文本响应
            output_ids = outputs['sequences']
            response = tokenizer.decode(output_ids[0].cpu().tolist(), skip_special_tokens=True)
            response = response.replace("[UNUSED_TOKEN_145]", "")
            history = history + [(query, response)]

            # 4. 确定目标图像和变化检测状态
            imageA, imageB, image_to_ground = None, None, None
            if not change_detection and len(images) == 1:
                # 单图像分割模式
                img = images[0]
                image_to_ground = img if isinstance(img, str) else img[0]
            elif change_detection and len(images) == 2:
                # 变化检测模式
                imageA = images[0] if isinstance(images[0], str) else images[0][0]
                imageB = images[1] if isinstance(images[1], str) else images[1][0]
                image_to_ground = imageB  # 在当前图像上解码分割和变化

            # 5. 掩码解码（分割和/或变化检测）
            if image_to_ground:
                # 提取隐藏状态
                output_hidden_states = outputs.hidden_states[-1]

                # 创建分割标记掩码
                seg_token_mask = output_ids[:, 1:-1] == self.seg_token_idx
                inputs_embeds_len = inputs['inputs_embeds'].size(1)
                seg_token_mask = torch.cat(
                    [
                        torch.zeros((seg_token_mask.shape[0], inputs_embeds_len)).bool().cuda(),
                        seg_token_mask,
                    ],
                    dim=1,
                )

                # 通过文本投影层
                hidden_states = self.model.text_hidden_fcs[0](output_hidden_states)

                # 提取分割嵌入特征
                pred_embeddings = [states[masks] for states, masks in zip(hidden_states, seg_token_mask)]

                # 变化检测特定处理
                if change_detection:
                    # 创建变化检测标记掩码
                    change_token_mask = output_ids[:, 1:-1] == self.change_token_idx
                    change_token_mask = torch.cat([
                        torch.zeros((change_token_mask.shape[0], inputs_embeds_len)).bool().cuda(),
                        change_token_mask,
                    ], dim=1)

                    # 提取变化检测特征
                    change_features = [states[masks] for states, masks in zip(hidden_states, change_token_mask)]

                # 编码目标图像
                if change_detection:
                    # 变化检测需要双图像编码
                    imageA_features, ori_hw_A = self.encode_g_img(imageA)
                    imageB_features, ori_hw_B = self.encode_g_img(imageB)
                else:
                    # 单图像分割只需当前图像
                    imageB_features, ori_hw_B = self.encode_g_img(image_to_ground)

                # 处理每个样本的掩码
                for i in range(len(pred_embeddings)):
                    # 5.1 分割掩码生成
                    seg_masks = []
                    if pred_embeddings[i].numel() > 0:
                        # 使用SAM解码器生成分割掩码
                        (sparse_embeddings, dense_embeddings) = self.model.visual_model.sam_prompt_encoder(
                            points=None,
                            boxes=None,
                            masks=None,
                            text_embeds=pred_embeddings[i].unsqueeze(1),
                        ) #sam 提示编码

                        batch_mode = (pred_embeddings[i].shape[0] > 1)
                        high_res_features = [
                            feat_level[i].unsqueeze(0)
                            for feat_level in imageB_features["high_res_feats"]
                        ]
                        sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                        image_g_embeds = imageB_features['image_embed'][i].unsqueeze(0).to(torch.bfloat16)

                        low_res_masks, _, _, _ = self.model.visual_model.sam_mask_decoder(
                            image_embeddings=image_g_embeds,
                            image_pe=self.model.visual_model.sam_prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                            repeat_image=batch_mode,
                            multimask_output=False,
                            high_res_features=high_res_features,
                        ) #掩码解码器

                        # 后处理分割掩码
                        seg_masks = self.model._transform.postprocess_masks(
                            low_res_masks,
                            ori_hw_B[i],
                        )
                        seg_masks = seg_masks[:, 0]  # 取第一个掩码

                    # 5.2 变化检测掩码生成
                    change_mask = None
                    if change_detection and change_features[i].numel() > 0:
                        # # 计算双图像特征差异
                        # diff_feature = torch.abs(
                        #     imageB_features['image_embed'][i] -
                        #     imageA_features['image_embed'][i]
                        # ).unsqueeze(0)
                        #
                        # # 融合分割和变化特征
                        # fused_prompt = torch.cat([
                        #     change_features[i].unsqueeze(1),
                        #     pred_embeddings[i].unsqueeze(1) if pred_embeddings[i].numel() > 0 else torch.zeros_like(
                        #         change_features[i].unsqueeze(1))
                        # ], dim=1) #将 change_features（变化相关的文本特征）和 pred_embeddings（分割相关的文本特征）拼接

                        # 1. 增强特征差异计算
                        diff_feature = self.change_detector(
                            imageA_features['image_embed'][i],
                            imageB_features['image_embed'][i]
                        )

                        # 2. 增强特征融合
                        if pred_embeddings[i].numel() > 0:
                            text_feat = torch.cat([
                                change_features[i].unsqueeze(1),
                                pred_embeddings[i].unsqueeze(1)
                            ], dim=1)
                        else:
                            text_feat = change_features[i].unsqueeze(1)

                        fused_prompt = self.cross_modal_fusion(diff_feature, text_feat)

                        # 通过提示编码器
                        (sparse_change, dense_change) = self.model.visual_model.sam_prompt_encoder(
                            points=None,
                            boxes=None,
                            masks=None,
                            text_embeds=fused_prompt,
                        )

                        # 变化检测解码
                        change_low_res, _, _, _ = self.model.visual_model.sam_mask_decoder(
                            image_embeddings=diff_feature,
                            image_pe=self.model.visual_model.sam_prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_change,
                            dense_prompt_embeddings=dense_change,
                            multimask_output=False,
                        )

                        # 后处理变化掩码
                        change_mask = self.model._transform.postprocess_masks(
                            change_low_res,
                            ori_hw_B[i],
                        )
                        change_mask = change_mask[:, 0]  # 取第一个掩码

                    # 保存结果
                    all_pred_masks.append(seg_masks)
                    if change_detection:
                        all_change_masks.append(change_mask)

        # 6. 返回结果
        if change_detection:
            return response, all_pred_masks, all_change_masks
        else:
            return response, all_pred_masks

