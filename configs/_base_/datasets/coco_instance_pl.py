dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

albu_train_transforms = [
    dict(type='OneOf',
        transforms = [
            dict(type='GaussianBlur', sigma_limit=[0,3], p=1.0),
            dict(type='Blur', blur_limit=[2,7], p=1.0),
            dict(type='Sharpen', alpha=(0.,1.), lightness=(0.75,1.5), p=1.0),
            dict(type='GaussNoise', var_limit=(0.0, 0.05*255), p=1.0),
            dict(type='InvertImg', p=0.05),
            dict(type='MultiplicativeNoise', multiplier=(0.5, 1.5)),
            dict(type='RandomBrightnessContrast', brightness_limit=[0.1, 0.3], contrast_limit=[0.1, 0.3], p=1.0),
            ], p=7/8
        )
    ]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=[(1333, 400), (1333, 1200)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Albu', transforms=albu_train_transforms, 
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes',
            'gt_masks': 'masks',
            },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='CutOut', n_holes=(1,5), cutout_shape=[(0,0), (33,20), (66,40), (132,80), (198,120), (264,160)] ),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train = dict(
        type='ConcatDataset',
        datasets=[
           dict(type='RepeatDataset',
                times=19,
                dataset = dict(type=dataset_type,
                    ann_file=data_root + 'annotations/instances_train2017.1@10.json',
                    img_prefix=data_root + 'images/train2017/',
                    pipeline=train_pipeline)),
           dict(type=dataset_type,
                ann_file='labels/coco_1@10_pl.json',
                img_prefix=data_root + 'images/train2017/',
                pipeline=train_pipeline),
        ]),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
