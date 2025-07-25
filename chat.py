import os
import re
import sys
import cv2
import torch
import random
import argparse
import numpy as np
import transformers 
from model.geopixel import GeoPixelForCausalLM

def rgb_color_text(text, r, g, b):
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"

def parse_args(args):
    parser = argparse.ArgumentParser(description="Chat with GeoPixel")
    # parser.add_argument("--version", default="MBZUAI/GeoPixel-7B")
    parser.add_argument("--version", default="/root/autodl-tmp/models/geopixel-7b")
    # parser.add_argument("--version", default=r"D:\LLM\models\geopixel-7b")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    return parser.parse_args(args)

def main(args): # Namespace(version='/root/autodl-tmp/models/geopixel-7b', vis_save_path='./vis_output')
    args = parse_args(args)

    os.makedirs(args.vis_save_path, exist_ok=True)

    print(f'initialing tokenizer from: {args.version}')
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        padding_side='right',
        use_fast=False,
        trust_remote_code=True,
    ) # InternLM2Tokenizer(name_or_path='/root/autodl-tmp/models/geopixel-7b', vocab_size=92544, model_max_length=1000000000000000019884624838656, is_fast=False, padding_side='right', truncation_side='right', special_tokens={'bos_token': AddedToken("<s>", rstrip=False, lstrip=False, single_word=False, normalized=False), 'eos_token': AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=False), 'unk_token': AddedToken("<unk>", rstrip=False, lstrip=False, single_word=False, normalized=False), 'pad_token': '<unk>', 'additional_special_tokens': ['<|im_start|>', '<|im_end|>', '<|action_start|>', '<|action_end|>', '<|interpreter|>', '<|plugin|>']}, clean_up_tokenization_spaces=False)
    tokenizer.pad_token = tokenizer.unk_token # "<unk>"
    seg_token_idx, bop_token_idx, eop_token_idx = [
        tokenizer(token, add_special_tokens=False).input_ids[0] for token in ['[SEG]','<p>', '</p>']
    ] # 获取特殊标记在分词器中的数字ID：92552, 92551, 92550

    kwargs = {"torch_dtype": torch.bfloat16}
    geo_model_args = {
        "vision_pretrained": 'facebook/sam2-hiera-large',
        # "vision_pretrained": '/root/autodl-tmp/models/sam2-hiera-large',
        # "vision_pretrained": r"D:\LLM\models\sam2-hiera-large",
        "seg_token_idx" : seg_token_idx, # segmentation token index
        "bop_token_idx" : bop_token_idx, # begining of phrase token index
        "eop_token_idx" : eop_token_idx  # end of phrase token index
    }

    # Load model
    print(f'Load model from: {args.version}')
    model = GeoPixelForCausalLM.from_pretrained(
        args.version, # '/root/autodl-tmp/models/geopixel-7b'
        low_cpu_mem_usage=True,
        device_map='auto',
        # torch_dtype=torch.bfloat16,  # 直接加载为bfloat16
        **kwargs, # {'torch_dtype': torch.bfloat16}
        **geo_model_args # {'bop_token_idx': 92551, 'eop_token_idx': 92552, 'seg_token_idx': 92550, 'vision_pretrained': 'facebook/sam2-hiera-large'}
    )

    model.config.eos_token_id = tokenizer.eos_token_id # 2 todo: 这些和前面的token_idx区别？
    model.config.bos_token_id = tokenizer.bos_token_id # 1
    model.config.pad_token_id = tokenizer.pad_token_id # 0
    model.tokenizer = tokenizer

    # model = model.bfloat16().cuda().eval()
    model = model.bfloat16().eval()

    while True:

        # query = input("Please input your query: ")
        query = "Can you provide a thorough description of this image? Please output with interleaved segmentation masks for the corresponding phrases."
        #query = f"Can you provide a thorough description of the this image? Please output with interleaved segmentation masks for the corresponding phrases."
        # image_path = input("Please input the image path: ")
        image_path = "images/example1.png"
        if not os.path.exists(image_path):
            print("File not found in {}".format(image_path))
            continue

        image = [image_path]

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            response, pred_masks = model.evaluate(tokenizer, query, images = image, max_new_tokens = 300)

        if pred_masks and '[SEG]' in response:
            pred_masks = pred_masks[0]
            pred_masks = pred_masks.detach().cpu().numpy()
            pred_masks = pred_masks > 0
            image_np = cv2.imread(image_path)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

            save_img = image_np.copy()
            pattern = r'<p>(.*?)</p>\s*\[SEG\]'
            matched_text = re.findall(pattern, response)
            phrases = [text.strip() for text in matched_text]

            for i in range(pred_masks.shape[0]):
                mask = pred_masks[i]

                color = [random.randint(0, 255) for _ in range(3)]
                if matched_text:
                    phrases[i] = rgb_color_text(phrases[i], color[0], color[1], color[2])
                mask_rgb = np.stack([mask, mask, mask], axis=-1) 
                color_mask = np.array(color, dtype=np.uint8) * mask_rgb

                save_img = np.where(mask_rgb, 
                        (save_img * 0.5 + color_mask * 0.5).astype(np.uint8), 
                        save_img)
            if matched_text:    # [' swimming pool ', ' tennis court ', ' numerous small vehicles ']
                split_desc = response.split('[SEG]')
                cleaned_segments = [re.sub(r'<p>(.*?)</p>', '', part).strip() for part in split_desc]
                reconstructed_desc = ""
                for i, part in enumerate(cleaned_segments):
                    reconstructed_desc += part + ' '
                    if i < len(phrases):
                        reconstructed_desc += phrases[i] + ' '    
                print(reconstructed_desc)
            else:
                print(response.replace("\n", "").replace("  ", " "))
            save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
            save_path = "{}/{}_masked.jpg".format(
                args.vis_save_path, image_path.split("/")[-1].split(".")[0]
                )
            cv2.imwrite(save_path, save_img)
            print("{} has been saved.".format(save_path))
        else:
            print(response.replace("\n", "").replace("  ", " "))

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
