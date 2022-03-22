
GPUS=$1
RESULTNAME=$2
ANNFILE=$3

#bash tools/dist_test.sh configs/noisyboundaries/coco/mask_rcnn_r50_fpn_1x_coco_generatepl.py  work_dirs/mask_rcnn_r50_fpn_1x_coco_sup/epoch_12.pth $GPUS --out $RESULTNAME

python scripts/coco/pkl2json.py $RESULTNAME

python scripts/coco/filter_pl.py $RESULTNAME

python scripts/coco/form_ann.py $RESULTNAME $ANNFILE
