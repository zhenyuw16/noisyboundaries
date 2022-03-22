_base_ = [
    '../../_base_/models/mask_rcnn_r50_fpn.py',
    '../../_base_/datasets/coco_instance_pl.py',
    #'../../_base_/schedules/schedule_1x.py', 
    '../../_base_/default_runtime.py'
]

model = dict(
    roi_head=dict(
        type='StandardRoIHeadNTM',
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        mask_head=dict(
            type='FCNMaskHeadNTM',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=80,
            num_fcs=2,
            downsample_factor=2,
            fc_out_channels=1024,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, use_bpm=True, loss_weight=1.))
))

# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    by_epoch=False,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[60000, 80000])
runner = dict(type='IterBasedRunner', max_iters=90000)
checkpoint_config = dict(by_epoch=False, interval=7500)
evaluation = dict(metric=['bbox', 'segm'], interval=7500)

