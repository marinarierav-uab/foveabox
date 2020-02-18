# model settings
model = dict(
    type='FOVEA',
    pretrained='torchvision://resnet101',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3), # C2, C3, C4, C5
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        num_outs=5,
        add_extra_convs=True),
    bbox_head=dict(
        type='FoveaHead',
        num_classes=81,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        base_edge_list=[16, 32, 64, 128, 256],
        scale_ranges=((1, 64), (32, 128), (64, 256), (128, 512), (256, 2048)),
        sigma=0.4,
        with_deform=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=1.50,
            alpha=0.4,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)
    ))
# training and testing settings
train_cfg = dict()
test_cfg = dict(
    nms_pre=1000,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=100)
# dataset settings
dataset_type = 'Polyp'
data_root = 'data/HDClassif/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=8,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/HDClassif.json',
        img_prefix=data_root + 'images/',
        #img_scale=[(384, 288), (384*0.9, 288*0.9), (384*0.8, 288*0.8), (384*0.7, 288*0.7)],  # escalado de imagen --> adds grey padding pocho, need to fix
        #img_scale=[(384*0.9, 288*0.9),(384*1.1, 288*1.1)],  # escalado de imagen --> adds grey padding pocho, need to fix
        img_scale=[(384, 288)],
        multiscale_mode='range',
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=False,
        with_label=True,
        # following are not yet implemented:
        hsv_h=0,  # image HSV-Hue augmentation (fraction)
        hsv_s=0,  # image HSV-Saturation augmentation (fraction)
        hsv_v=0,  # image HSV-Value augmentation (fraction)
        degrees=0,  # image rotation (+/- deg)
        translate=0,  # image translation (+/- fraction)
        scale=0,  # image scale (+/- gain)
        shear=0  # image shear (+/- deg)
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + '../CVC-VideoClinicDBtrain_valid/annotations/renamed-valid.json',
        img_prefix=data_root + '../CVC-VideoClinicDBtrain_valid/images/train/',
        img_scale=(384, 288),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + '../CVC-VideoClinicDBtrain_valid/annotations/test.json',
        img_prefix=data_root + '../CVC-VideoClinicDBtrain_valid/images/test/',
        img_scale=(384, 288),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=False,
        test_mode=True))
# optimizer
#optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
#optimizer = dict(type='SGD', lr=0.0075, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[16, 22])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 20
device_ids = range(4)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/exp29'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]
"""
workflow (list[tuple]): A list of (phase, epochs) to specify the
running order and epochs. E.g, [('train', 2), ('val', 1)] means
running 2 epochs for training and 1 epoch for validation,
iteratively.
"""