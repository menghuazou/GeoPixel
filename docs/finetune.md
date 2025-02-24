# GeoPixel Finetuning

You can easily finetune the pretrained [GeoPixel](https://huggingface.co/MBZUAI/GeoPixel-7B) model on downstream tasks using our finetuning scripts. For setup guidance, check out the [installation instructions](../docs/install.md).

> The data format of GeoPixel follows the [InternLM-XComposer2.5](https://github.com/InternLM/InternLM-XComposer/tree/main) with additional structure for segmentation masks.

### Data preparation

Your fine-tuning data should follow the following format:

1. Saved as a list in json format, each conversation corresponds to an element of the list
2. The image-text-mask conversation contains four keys: `id`, `conversation`, `image` and `polygon/segmentation`.
3. `image` is the file path of the image. `polygon` is path to json file with segmentation masks as polygons. `segmentation` is  list of COCO style RLE masks.
4. conversation is in list format

```
# An example of conversations including polygon 
temp = {
 'id': 0,
 'conversations': [
     {'from': 'human',   'value': 'Q'},
     {'from': 'bot',   'value': 'A'}，
  ],
 'image': 'path_to_image'
 'polygon': 'path_to_json_with_polygons'
}

```
```
# An example of conversations including rle_masks
temp = {
 'id': 0,
 'conversations': [
     {'from': 'human',   'value': 'Q'},
     {'from': 'bot',   'value': 'A'}，
  ],
 'image': 'path_to_image'
 'segmentation': 'list_of_COCO_rle_masks'
}

```

5. for image **no placeholder is required**


After pre-pareing the JSON files, you are required to define all the JSON file paths in a text file (e.g., `data.txt`) using the format:

```
<json path> <sample number (k)>
```

For example:

```
data/GeoPixel.json 0.02
```

This means the model will sample 20 samples from `data/GeoPixel.json`

### Training / Finetuning
After data preparation, you can use the provided bash scripts (`train.sh`) to train (from base models weights) or (`finetune.sh`) to finetune the pre-trained GeoPixel model. Remember to specify the pre-train model path ($MODEL) and the txt data file path ($DATA) in the bash script.

- For Remote Sensing Grounded Conversation Generation (RS-GCG) task : set $MODEL to "MBZUAI/GeoPixel-7B"
- For Referring Remote Sensing Image Segmentation (RRSIS) task : set $MODEL to "MBZUAI/GeoPixel-7B-RES"

When training is finished, to get the full model weight:

```
cd ./output/checkpoint-last && python zero_to_fp32.py . ../pytorch_model.bin
```

### Merge LoRA Weight
Merge the LoRA weights of `pytorch_model.bin`, save the resulting model into your desired path in the Hugging Face format:
```
CUDA_VISIBLE_DEVICES="" python merge_lora_weights_and_save_hf_model.py \
  --version="PATH_TO_BASE_MODEL" \
  --weight="PATH_TO_pytorch_model.bin" \
  --save_path="PATH_TO_SAVE_MODEL_IN_HF_FORMAT"
```
