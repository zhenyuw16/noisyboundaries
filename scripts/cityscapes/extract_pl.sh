
GPUS=$1
RESULTNAME=$2
ANNFILE=$3

bash tools/dist_test.sh configs/noisyboundaries/cityscapes/mask_rcnn_r50_fpn_1x_cityscapes_generatepl.py work_dirs/mask_rcnn_r50_fpn_1x_cityscapes_sup/epoch_8.pth $GPUS --out $RESULTNAME

python scripts/cityscapes/pkl2json.py $RESULTNAME

python scripts/cityscapes/filter_pl.py $RESULTNAME

python scripts/cityscapes/form_ann.py $RESULTNAME $ANNFILE
