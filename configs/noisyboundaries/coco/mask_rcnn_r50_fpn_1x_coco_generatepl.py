_base_ = [
    '../../_base_/models/mask_rcnn_r50_fpn.py',
    '../../_base_/datasets/coco_instance_generatepl.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]
model=dict(test_cfg=dict(rcnn=dict(mask_thr_binary=0.44)) )
