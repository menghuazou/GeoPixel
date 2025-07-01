# 测试准备 - 模拟最小环境
import torch


class MockModel:
    def __init__(self):
        self.tok_embeddings = lambda x: torch.randn(1, len(x[0]), 4096)  # 随机嵌入
        self.device = "cpu"


def mock_encode_img(img_path, hd_num):
    """模拟图像编码器"""
    print(f"编码图像: {img_path} (特征数: {hd_num})")
    return torch.randn(1, hd_num, 4096)  # 随机图像特征


# 创建测试对象
class Tester:
    def __init__(self):
        self.model = MockModel()
        self.device = "cpu"
        self.max_length = 512
        self.tokenizer = type('', (), {'__call__': lambda self, text, **kw: type('', (), {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        })})()
        self.encode_img = mock_encode_img  # 使用模拟图像编码器

    # 直接复制要测试的函数
    interleav_wrap_chat = lambda self, **kwargs: self.__class__.interleav_wrap_chat(self, **kwargs)

    # 粘贴要测试的函数代码 (原封不动)
    def interleav_wrap_chat(self, query, images, history=[], meta_instruction='',
                            max_length=16384, hd_num=24, change_detection=False):
        self.max_length = max_length
        prompt = ''

        # 添加系统指令
        if meta_instruction:
            prompt += f"""[UNUSED_TOKEN_146]system\n{meta_instruction}[UNUSED_TOKEN_145]\n"""

        # 添加历史对话
        for record in history:
            prompt += f"""[UNUSED_TOKEN_146]user\n{record[0]}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n{record[1]}[UNUSED_TOKEN_145]\n"""

        # 添加当前查询
        prompt += f"""[UNUSED_TOKEN_146]user\n{query}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n"""

        # ===== 核心修改：支持双图像变化检测 =====
        if change_detection:
            # 确保有两个图像输入
            if len(images) != 2:
                raise ValueError("变化检测需要两个图像输入: [历史图像, 当前图像]")

            # 自动添加双图像占位符
            if prompt.find('<ImageAHere>') == -1 and prompt.find('<ImageBHere>') == -1:
                prompt = '<ImageAHere>历史图像<ImageBHere>当前图像\n' + prompt # todo 变化检测的该如何设计？
        else:
            # 单图像自动添加占位符
            image_nums = len(images)
            if image_nums == 1 and prompt.find('<ImageHere>') == -1:
                prompt = '<ImageHere>' + prompt

        # ===== 多图像占位符处理 =====
        if change_detection:
            # 双图像特殊占位符处理
            parts = []
            temp_parts = prompt.split('<ImageAHere>')
            for part in temp_parts:
                sub_parts = part.split('<ImageBHere>')
                parts.extend(sub_parts)
                # 在A和B之间添加分隔标记
                if len(sub_parts) > 1:
                    parts.insert(-1, "[CHANGE_DETECT]")
        else:
            # 原始单图像处理
            parts = prompt.split('<ImageHere>')

        wrap_embeds, wrap_im_mask = [], []
        temp_len = 0
        need_bos = True

        # 调整图像数量检测
        image_nums = 2 if change_detection else len(images)

        if image_nums > 1:
            hd_num = max(6, hd_num)  # 确保足够的特征数量

        # ===== 嵌入处理循环 =====
        image_index = 0
        for idx, part in enumerate(parts):
            if part == "[CHANGE_DETECT]":
                # 添加变化检测特殊标记
                change_token = self.tokenizer(
                    "[CHANGE]",
                    return_tensors='pt',
                    add_special_tokens=False
                )
                change_embed = self.model.tok_embeddings(change_token.input_ids)
                wrap_embeds.append(change_embed)
                wrap_im_mask.append(torch.zeros(change_embed.shape[:2]))
                temp_len += change_embed.shape[1]
                continue

            if need_bos or len(part) > 0:
                part_tokens = self.tokenizer(
                    part,
                    return_tensors='pt',
                    padding='longest',
                    add_special_tokens=need_bos
                )

                if need_bos:
                    need_bos = False

                part_embeds = self.model.tok_embeddings(part_tokens.input_ids)
                wrap_embeds.append(part_embeds)
                wrap_im_mask.append(torch.zeros(part_embeds.shape[:2]))
                temp_len += part_embeds.shape[1]

            # 图像嵌入处理
            if image_index < image_nums:
                # 处理元组输入 (路径, ID)
                img_source = images[image_index]
                img_path = img_source[0] if isinstance(img_source, tuple) else img_source  # 处理元组的情况

                # 变化检测图像特殊处理
                if change_detection:
                    # 历史图像使用低分辨率特征
                    if image_index == 0:
                        img = self.encode_img(img_path, hd_num=6)  # todo: encode_img
                    # 当前图像使用高分辨率特征
                    else:
                        img = self.encode_img(img_path, hd_num=12)
                else:
                    img = self.encode_img(img_path, hd_num)

                wrap_embeds.append(img)
                wrap_im_mask.append(torch.ones(img.shape[:2]))
                temp_len += img.shape[1]
                image_index += 1

            if temp_len > self.max_length:
                break

        # ===== 特征拼接 =====
        wrap_embeds = torch.cat(wrap_embeds, dim=1)
        wrap_im_mask = torch.cat(wrap_im_mask, dim=1)

        # 裁剪到最大长度
        wrap_embeds = wrap_embeds[:, :self.max_length].to(self.device)
        wrap_im_mask = wrap_im_mask[:, :self.max_length].to(self.device).bool()

        inputs = {
            'inputs_embeds': wrap_embeds
        }

        # 变化检测添加额外标记
        if change_detection:
            inputs['change_detection'] = True

        return inputs, wrap_im_mask, temp_len

# ===== 测试用例 =====
if __name__ == "__main__":
    tester = Tester()

    # 测试用例1: 单图像分析
    print("\n=== 测试1: 单图像分析 ===")
    inputs, mask, length = tester.interleav_wrap_chat(
        query="描述这张卫星图像",
        images=[r"D:\AAALook This\Work\luxitech\GeoPixel\data\GeoPixelD\train\P0224_0_800_0_800.png"]
    )
    print(f"输入形状: {inputs['inputs_embeds'].shape}")
    print(f"掩码形状: {mask.shape}")

    # 测试用例2: 变化检测
    print("\n=== 测试2: 变化检测 ===")
    inputs, mask, length = tester.interleav_wrap_chat(
        query="分析城市变化",
        images=[
            (r"D:\AAALook This\Work\luxitech\Dataset\unet_dafei\dataset_voc\JPEGImages\P0224_0_800_600_1400.jpg", "历史图像"),
            ("/data/current.png", "当前图像")
        ],
        change_detection=True
    )
    print(f"包含变化检测标记: {'change_detection' in inputs}")

    # 测试用例3: 带历史对话
    print("\n=== 测试3: 带历史对话 ===")
    inputs, mask, length = tester.interleav_wrap_chat(
        query="洪水影响范围有多大?",
        images=["/data/flood.jpg"],
        history=[
            ("河南的洪水情况如何?", "河南遭遇特大暴雨"),
            ("上传最新的卫星图像", "已收到图像数据")
        ],
        meta_instruction="你是一个灾害评估专家"
    )
    print(f"最终序列长度: {length}")