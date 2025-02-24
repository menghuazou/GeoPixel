# GeoPixel Finetuning

We provide the official scripts for easy finetuning of the pretrained [GeoPixel](https://huggingface.co/MBZUAI/GeoPixel-7B) model on downstream tasks. Please refer to the [installation instructions](../docs/install.md) for installation details.

> The data format of GeoPixel follows the [InternLM-XComposer2.5](https://github.com/InternLM/InternLM-XComposer/tree/main) with additional structure for segmentation masks.

### Data preparation

Two different formats of fine-tuning data are explained below:

- `data/polygons_example.json`: polygons
- `data/rle_masks_example.json`：RLE masks

Your fine-tuning data should follow the following format:

1. Saved as a list in json format, each conversation corresponds to an element of the list
2. The image-text-mask conversation contains four keys: `id`, `conversation`, `image` and `polygon/mask`.
3. `image` is the file path of the image. `polygon` is path to json file with segmentation masks as polygons. `mask` is 
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

5. for image **no placeholder required**


After pre-pareing the JSON files, you are required to define all the JSON file paths in a text file (e.g., `data.txt`) using the format:

```
<json path> <sample number (k)>
```

For example:

```
data/GeoPixel.json 0.02
```

This means the model will sample 20 samples from `data/GeoPixel.json`

After data preparation, you can use the provided bash scripts (`finetune.sh`) to finetune the model. Remember to specify the pre-train model path ($MODEL) and the txt data file path ($DATA) in the bash script.
