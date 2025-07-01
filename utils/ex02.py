import torch
from PIL import Image
from transformers import TextStreamer
from typing import List, Tuple, Union


class MockModel:
    """模拟模型的关键组件"""

    def __init__(self):
        self.device = "cpu"
        self.seg_token_idx = 250000  # 模拟分割标记ID
        self.change_token_idx = 250001  # 模拟变化检测标记ID
        self._transform = MockTransform()

        # 模拟视觉模型组件
        self.visual_model = MockVisualModel()
        self.text_hidden_fcs = [torch.nn.Linear(768, 256)]

    def generate(self, **kwargs):
        """模拟生成过程"""
        print("生成文本响应...")
        return {
            'sequences': torch.tensor([[1, 2, self.seg_token_idx, self.change_token_idx, 3]]),
            'hidden_states': (None, torch.randn(1, 5, 768))
        }

    def interleav_wrap_chat(self, *args, **kwargs):
        """模拟输入预处理"""
        print("预处理输入...")
        return {
            'inputs_embeds': torch.randn(1, 10, 768)
        }, torch.zeros(1, 10).bool(), 10

    def encode_g_img(self, image):
        """模拟图像编码"""
        print(f"编码图像: {image}")
        return {
            "image_embed": torch.randn(1, 256, 64, 64),
            "high_res_feats": [torch.randn(1, 512, 128, 128)]
        }, [(512, 512)]  # 原始尺寸

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
        评估函数，支持单图像分割和双图像变化检测
        """
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

            # 2. 模型生成文本响应
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
                output_hidden_states = outputs['hidden_states'][-1]

                # 创建分割标记掩码
                seg_token_mask = output_ids[:, 1:-1] == self.seg_token_idx
                inputs_embeds_len = inputs['inputs_embeds'].size(1)
                seg_token_mask = torch.cat(
                    [
                        torch.zeros((seg_token_mask.shape[0], inputs_embeds_len)).bool().to(self.device),
                        seg_token_mask,
                    ],
                    dim=1,
                )

                # 通过文本投影层
                hidden_states = self.text_hidden_fcs[0](output_hidden_states)

                # 提取分割嵌入特征
                pred_embeddings = [states[masks] for states, masks in zip(hidden_states, seg_token_mask)]

                # 变化检测特定处理
                if change_detection:
                    # 创建变化检测标记掩码
                    change_token_mask = output_ids[:, 1:-1] == self.change_token_idx
                    change_token_mask = torch.cat([
                        torch.zeros((change_token_mask.shape[0], inputs_embeds_len)).bool().to(self.device),
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
                        (sparse_embeddings, dense_embeddings) = self.visual_model.sam_prompt_encoder(
                            points=None,
                            boxes=None,
                            masks=None,
                            text_embeds=pred_embeddings[i].unsqueeze(1),
                        )

                        batch_mode = (pred_embeddings[i].shape[0] > 1)
                        high_res_features = [
                            feat_level[i].unsqueeze(0)
                            for feat_level in imageB_features["high_res_feats"]
                        ]
                        sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                        image_g_embeds = imageB_features['image_embed'][i].unsqueeze(0).to(torch.bfloat16)

                        low_res_masks, _, _, _ = self.visual_model.sam_mask_decoder(
                            image_embeddings=image_g_embeds,
                            image_pe=self.visual_model.sam_prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                            repeat_image=batch_mode,
                            multimask_output=False,
                            high_res_features=high_res_features,
                        )

                        # 后处理分割掩码
                        seg_masks = self._transform.postprocess_masks(
                            low_res_masks,
                            ori_hw_B[i],
                        )
                        seg_masks = seg_masks[:, 0]  # 取第一个掩码

                    # 5.2 变化检测掩码生成
                    change_mask = None
                    if change_detection and change_features[i].numel() > 0:
                        # 计算双图像特征差异
                        diff_feature = torch.abs(
                            imageB_features['image_embed'][i] -
                            imageA_features['image_embed'][i]
                        ).unsqueeze(0)

                        # 融合分割和变化特征
                        fused_prompt = torch.cat([
                            change_features[i].unsqueeze(1),
                            pred_embeddings[i].unsqueeze(1) if pred_embeddings[i].numel() > 0 else torch.zeros_like(
                                change_features[i].unsqueeze(1))
                        ], dim=1)

                        # 通过提示编码器
                        (sparse_change, dense_change) = self.visual_model.sam_prompt_encoder(
                            points=None,
                            boxes=None,
                            masks=None,
                            text_embeds=fused_prompt,
                        )

                        # 变化检测解码
                        change_low_res, _, _, _ = self.model.visual_model.sam_mask_decoder(
                            image_embeddings=diff_feature,
                            image_pe=self.visual_model.sam_prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_change,
                            dense_prompt_embeddings=dense_change,
                            multimask_output=False,
                        )

                        # 后处理变化掩码
                        change_mask = self._transform.postprocess_masks(
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


class MockVisualModel:
    """模拟视觉模型组件"""

    def sam_prompt_encoder(self, *args, **kwargs):
        print("调用提示编码器...")
        return torch.randn(1, 1, 256), torch.randn(1, 1, 256)  # sparse, dense

    def sam_mask_decoder(self, *args, **kwargs):
        print("调用掩码解码器...")
        return torch.randn(1, 1, 256, 256), None, None, None  # 低分辨率掩码

    def get_dense_pe(self):
        return torch.randn(1, 256, 64, 64)


class MockTransform:
    """模拟图像变换"""

    def postprocess_masks(self, masks, original_size):
        print(f"后处理掩码，原始尺寸: {original_size}")
        return masks  # 直接返回输入


class MockTokenizer:
    """模拟分词器"""

    def __init__(self):
        self.eos_token_id = 2

    def convert_tokens_to_ids(self, tokens):
        return [3]  # 模拟特殊标记ID

    def decode(self, token_ids, **kwargs):
        return "模拟的文本响应"

    def __call__(self, *args, **kwargs):
        return type('', (), {'input_ids': torch.tensor([[1, 2, 3]])})


def test_evaluate():
    """测试evaluate函数"""
    # 创建模拟对象
    model = MockModel()
    tokenizer = MockTokenizer()

    # 使用MockModel自带的evaluate方法

    # 测试场景1: 单图像分割
    print("\n=== 测试1: 单图像分割 ===")
    response, seg_masks = model.evaluate(
        tokenizer,
        query="分割图中的建筑物",
        images=[("test_image.jpg", "测试图像")],
        change_detection=False
    )
    print(f"响应: {response}")
    print(f"分割掩码数量: {len(seg_masks)}")

    # 测试场景2: 变化检测
    print("\n=== 测试2: 变化检测 ===")
    response, seg_masks, change_masks = model.evaluate(
        tokenizer,
        query="分析城市变化",
        images=[
            ("historical_image.tif", "历史图像"),
            ("current_image.tif", "当前图像")
        ],
        change_detection=True
    )
    print(f"响应: {response}")
    print(f"分割掩码数量: {len(seg_masks)}")
    print(f"变化掩码数量: {len(change_masks)}")

    # 测试场景3: 无图像输入
    print("\n=== 测试3: 无图像输入 ===")
    response, _ = model.evaluate(
        tokenizer,
        query="什么是城市扩张？",
        images=[]
    )
    print(f"纯文本响应: {response}")


if __name__ == "__main__":
    test_evaluate()