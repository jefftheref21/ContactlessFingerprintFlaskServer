_base_ = [
    './mmdetection/configs/_base_/models/faster_rcnn_r50_fpn.py',
    './mmdetection/configs/_base_/datasets/coco_detection.py',
    './mmdetection/configs/_base_/schedules/schedule_2x.py', 
    './mmdetection/configs/_base_/default_runtime.py'
]

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1,)),
    
    # data_preprocessor=dict(
    #     type='DetDataPreprocessor',
    #     mean=[123.675, 116.28, 103.53],
    #     std=[58.395, 57.12, 57.375],
    #     bgr_to_rgb=True,
    #     pad_mask=True,
    #     pad_size_divisor=32)
    
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='CocoDataset',
        data_root='',
        ann_file='./coco_fingerprint_TRAIN.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='',
        ann_file='./coco_fingerprint_TEST.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline
    )
)
test_dataloader = val_dataloader  # Same as val_dataloader

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=12,
    val_interval=2
)

val_cfg = dict(type='ValLoop')
# test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.02,
        momentum=0.9,
        weight_decay=0.0001),
    clip_grad=None
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500
    ),
    dict(
        type='MultiStepLR',
        by_epoch=True,
        begin=0,
        end=12,
        milestones=[8, 11],
        gamma=0.1
    )
]

# Other configurations (e.g., logging, checkpoint settings) should be adjusted as needed based on the migration guidelines and your project requirements.
