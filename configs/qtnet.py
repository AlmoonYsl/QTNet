point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
voxel_size = [0.075, 0.075, 0.2]
out_size_factor = 8
num_frames = 4
dataset_type = 'NuScenesDatasetMemory'
data_root = 'data/nuscenes/'
root_dir = data_root + 'memorybank/'
features_root_train = root_dir + 'transfusionL/queries/train/'
features_root_val = root_dir + 'transfusionL/queries/val/'
features_root_test = root_dir + 'transfusionL/queries/test/'
pred_result_root_train = root_dir + 'transfusionL/detections/train/'
pred_result_root_val = root_dir + 'transfusionL/detections/val/'
pred_result_root_test = root_dir + 'transfusionL/detections/test/'
file_client_args = dict(backend='disk')
input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)
train_pipeline = [
    dict(
        type='LoadMultiFeaturesFromFile',
        num_frames=num_frames,
        file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['queries', 'pred_results', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadMultiFeaturesFromFile',
        num_frames=num_frames,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['queries', 'pred_results'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        num_frames=num_frames,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_train.pkl',
        load_interval=1,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        features_root=features_root_train,
        pred_result_root=pred_result_root_train,
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        num_frames=num_frames,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        load_interval=1,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        features_root=features_root_val,
        pred_result_root=pred_result_root_val,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        num_frames=num_frames,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        load_interval=1,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        features_root=features_root_val,
        pred_result_root=pred_result_root_val,
        box_type_3d='LiDAR'))
model = dict(
    type='QTNetDetector',
    pts_bbox_head=dict(
        type='QTNetHead',
        num_frames=num_frames,
        extension=True,
        pred_weight=0.5,
        hidden_channel=128,
        num_classes=len(class_names),
        num_heatmap_convs=2,
        common_heads=dict(center=(2, 3), height=(1, 3), dim=(3, 3), rot=(2, 3), vel=(2, 3)),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=point_cloud_range[:2],
            voxel_size=voxel_size[:2],
            out_size_factor=out_size_factor,
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=0.0,
            code_size=10,
        ),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2, alpha=0.25, reduction='mean', loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        loss_iou=dict(type='SmoothL1Loss', reduction='mean'),
    ),
    train_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            assigner=dict(
                type='HungarianAssigner3D',
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(type='FocalLossCost', gamma=2, alpha=0.25, weight=0.15),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                iou_cost=dict(type='IoU3DCost', weight=0.25)
            ),
            pos_weight=-1,
            gaussian_overlap=0.1,
            min_radius=2,
            grid_size=[1440, 1440, 40],  # [x_len, y_len, 1]
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            code_weights=[0.2, 0.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            point_cloud_range=point_cloud_range)),
    test_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            grid_size=[1440, 1440, 40],
            out_size_factor=out_size_factor,
            pc_range=point_cloud_range[0:2],
            voxel_size=voxel_size[:2],
            nms_type=None,
        )))
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.01)  # for 8gpu * 2sample_per_gpu
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 0.0001),
    cyclic_times=1,
    step_ratio_up=0.4)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.4)
total_epochs = 10
evaluation = dict(interval=10)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 8)
