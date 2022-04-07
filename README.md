# Noisy Boundaries: Lemon or Lemonade for semi-supervised instance segmentation?

This is the mmdetection implementation of our CVPR 2022 paper. [ArXiv](https://arxiv.org/abs/2203.13427).


# Installation

This code is based on mmdetection v2.18.
Please install the code according to the [mmdetection step](https://github.com/open-mmlab/mmdetection/blob/v2.18.0/docs/get_started.md) first.

### data preparation

```bash
noisyboundaries
├──data
|  ├──cityscapes
|  |  ├──annotations
|  |  |  ├──instancesonly_filtered_gtFine_train.json
|  |  |  ├──instancesonly_filtered_gtFine_val.json
|  |  ├──leftImg8bit
|  |  |  ├──train
|  |  |  ├──val
|  ├──coco
|  |  ├──annotations
|  |  |  ├──instances_train2017.json
|  |  |  ├──instances_val2017.json
|  |  ├──images
|  |  |  ├──train2017
|  |  |  ├──val2017
```

# Running scripts

## cityscapes
We take the experiment with the 20% labeled images for example.

make the label file first:
```bash
mkdir labels
python scripts/cityscapes/prepare_cityscape_data.py --percent 20 --seed 1
```

Then, to train the supervised model, run:
```bash
bash tools/dist_train.sh configs/noisyboundaries/cityscapes/mask_rcnn_r50_fpn_1x_cityscapes_sup.py 8
```
With the supervised model, generating pseudo labels for semi-supervised learning:
```bash
bash scripts/cityscapes/extract_pl.sh 8 labels/rcity.pkl labels/cityscapes_1@20_pl.json 
```
Then, perform semi-supervised learning:
```bash
bash tools/dist_train.sh configs/noisyboundaries/cityscapes/mask_rcnn_r50_fpn_1x_cityscapes_pl.py 8
```

