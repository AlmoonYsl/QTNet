import copy
from functools import reduce

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer, kaiming_init
from mmcv.runner import force_fp32
from mmdet.core import build_bbox_coder, multi_apply, build_assigner, PseudoSampler
from torch import nn

from mmdet3d.models.builder import HEADS, build_loss


class PositionEmbeddingLearned(nn.Module):
    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


class MTMDecoder(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.linear_attn_v = nn.Linear(d_model, d_model)
        self.linear_attn_out = nn.Linear(d_model, d_model)

        def _get_activation_fn(activation):
            """Return an activation function given a string"""
            if activation == "relu":
                return F.relu
            if activation == "gelu":
                return F.gelu
            if activation == "glu":
                return F.glu
            raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

        self.activation = _get_activation_fn(activation)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, query, key, attn_map):
        # (B, C, L)
        value = self.linear_attn_v(key.permute(0, 2, 1))
        pre_feature = torch.bmm(attn_map, value)
        pre_feature = self.linear_attn_out(pre_feature)

        # (B, L, C)
        query = query.permute(0, 2, 1)
        query = query + self.dropout1(pre_feature)
        query = self.norm1(query)

        query2 = self.linear2(self.dropout2(self.activation(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm2(query)

        query = query.permute(0, 2, 1)
        return query


class FFN(nn.Module):
    def __init__(self,
                 in_channels,
                 heads,
                 head_conv=64,
                 final_kernel=1,
                 init_bias=-2.19,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 bias='auto',
                 **kwargs):
        super(FFN, self).__init__()

        self.heads = heads
        self.init_bias = init_bias
        for head in self.heads:
            classes, num_conv = self.heads[head]

            conv_layers = []
            c_in = in_channels
            for i in range(num_conv - 1):
                conv_layers.append(
                    ConvModule(
                        c_in,
                        head_conv,
                        kernel_size=final_kernel,
                        stride=1,
                        padding=final_kernel // 2,
                        bias=bias,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg))
                c_in = head_conv

            conv_layers.append(
                build_conv_layer(
                    conv_cfg,
                    head_conv,
                    classes,
                    kernel_size=final_kernel,
                    stride=1,
                    padding=final_kernel // 2,
                    bias=True))
            conv_layers = nn.Sequential(*conv_layers)

            self.__setattr__(head, conv_layers)

    def init_weights(self):
        """Initialize weights."""
        for head in self.heads:
            if head == 'heatmap':
                self.__getattr__(head)[-1].bias.data.fill_(self.init_bias)
            else:
                for m in self.__getattr__(head).modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_init(m)

    def forward(self, x):
        """Forward function for SepHead.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].

        Returns:
            dict[str: torch.Tensor]: contains the following keys:

                -reg （torch.Tensor): 2D regression value with the \
                    shape of [B, 2, H, W].
                -height (torch.Tensor): Height value with the \
                    shape of [B, 1, H, W].
                -dim (torch.Tensor): Size value with the shape \
                    of [B, 3, H, W].
                -rot (torch.Tensor): Rotation value with the \
                    shape of [B, 1, H, W].
                -vel (torch.Tensor): Velocity value with the \
                    shape of [B, 2, H, W].
                -heatmap (torch.Tensor): Heatmap with the shape of \
                    [B, N, H, W].
        """
        ret_dict = dict()
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict


@HEADS.register_module()
class QTNetHead(nn.Module):
    def __init__(self,
                 num_frames=2,
                 extension=True,
                 pred_weight=0.5,
                 det_weight=0.5,
                 num_classes=4,
                 hidden_channel=128,
                 ffn_channel=256,
                 dropout=0.1,
                 activation='relu',
                 # config for FFN
                 common_heads=dict(),
                 num_heatmap_convs=2,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 bias='auto',
                 # loss
                 loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_bbox=dict(type='L1Loss', reduction='mean'),
                 loss_iou=None,
                 max_diff=[4, 4, 5, 5.5, 3, 0.2, 13, 3, 1, 0.2],
                 # others
                 train_cfg=None,
                 test_cfg=None,
                 bbox_coder=None):
        super(QTNetHead, self).__init__()
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.num_proposals = None
        self.extension = extension
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.iou_enable = loss_iou is not None

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        if self.iou_enable:
            self.loss_iou = build_loss(loss_iou)

        self.bbox_coder = build_bbox_coder(bbox_coder)

        self.mtm_decoder_cls = MTMDecoder(hidden_channel, ffn_channel, dropout, activation)
        self.mtm_decoder_reg = MTMDecoder(hidden_channel, ffn_channel, dropout, activation)

        heads = copy.deepcopy(common_heads)
        if self.iou_enable:
            heads.update(dict(iou=(1, 3)))
        self.prediction_heads_reg = FFN(hidden_channel, heads, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias)
        heads = dict(heatmap=(self.num_classes, num_heatmap_convs))
        self.prediction_heads_cls = FFN(hidden_channel, heads, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias)

        self.init_weights()
        if self.train_cfg is not None:
            self.bbox_sampler = PseudoSampler()
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)

        self.max_diff = np.asarray(max_diff, dtype=np.float32)
        self.pred_weight = pred_weight
        self.det_weight = det_weight

    def init_weights(self):
        for m in self.mtm_decoder_cls.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        for m in self.mtm_decoder_reg.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = 0.1

    @staticmethod
    def inverse_sigmoid(x, eps=1e-5):
        x = x.clamp(min=0, max=1)
        x1 = x.clamp(min=eps)
        x2 = (1 - x).clamp(min=eps)
        return torch.log(x1 / x2)

    def bev2lidar(self, center):
        voxel_size = self.test_cfg['voxel_size']
        pc_range = self.test_cfg['pc_range']
        out_size_factor = self.test_cfg['out_size_factor']
        lidar_coord = center.clone()
        lidar_coord[..., 0] = lidar_coord[..., 0] * voxel_size[0] * out_size_factor + pc_range[0]
        lidar_coord[..., 1] = lidar_coord[..., 1] * voxel_size[1] * out_size_factor + pc_range[1]
        return lidar_coord

    def lidar2bev(self, center, replace=False):
        voxel_size = self.test_cfg['voxel_size']
        pc_range = self.test_cfg['pc_range']
        out_size_factor = self.test_cfg['out_size_factor']
        if replace:
            lidar_coord = center
        else:
            lidar_coord = center.clone()
        lidar_coord[..., 0] = (lidar_coord[..., 0] - pc_range[0]) / voxel_size[0] / out_size_factor
        lidar_coord[..., 1] = (lidar_coord[..., 1] - pc_range[1]) / voxel_size[1] / out_size_factor
        return lidar_coord

    def lidar2bev_boxes(self, boxes):
        voxel_size = self.test_cfg['voxel_size']
        pc_range = self.test_cfg['pc_range']
        out_size_factor = self.test_cfg['out_size_factor']
        targets = torch.zeros([boxes.shape[0], boxes.shape[1], boxes.shape[2] + 1]).to(boxes.device)
        targets[..., 0] = (boxes[..., 0] - pc_range[0]) / (out_size_factor * voxel_size[0])
        targets[..., 1] = (boxes[..., 1] - pc_range[1]) / (out_size_factor * voxel_size[1])
        targets[..., 3] = boxes[..., 3].log()
        targets[..., 4] = boxes[..., 4].log()
        targets[..., 5] = boxes[..., 5].log()
        targets[..., 2] = boxes[..., 2] + boxes[..., 5] * 0.5  # bottom center to gravity center
        targets[..., 6] = torch.sin(boxes[..., 6])
        targets[..., 7] = torch.cos(boxes[..., 6])
        targets[..., 8] = boxes[..., 7]
        targets[..., 9] = boxes[..., 8]
        return targets

    def _align_center(self, center, vel, lidar2ego, ego2global, cur_lidar2ego, cur_ego2global):
        B, L, _ = center.size()

        lidar_coord = center.clone()
        lidar_coord[..., 0] = lidar_coord[..., 0] + vel[..., 0] * 0.5
        lidar_coord[..., 1] = lidar_coord[..., 1] + vel[..., 1] * 0.5

        # (B, L, 4) (x, y, z, 1)
        expand_lidar_coord = torch.cat([lidar_coord, center.new_ones((B, L, 2))], dim=-1)
        # TODO BUG: torch.inverse may return NaN.
        #  Use np.linalg.inv to solve the bug.
        lidar2ego = lidar2ego.cpu().numpy()
        ego2global = ego2global.cpu().numpy()
        lidar2ego_inverse = np.linalg.inv(lidar2ego)
        ego2global_inverse = np.linalg.inv(ego2global)
        lidar2ego_inverse = torch.from_numpy(lidar2ego_inverse).to(center.device)
        ego2global_inverse = torch.from_numpy(ego2global_inverse).to(center.device)
        tm = reduce(torch.bmm, [lidar2ego_inverse, ego2global_inverse, cur_ego2global, cur_lidar2ego])
        aligned_lidar_coord = tm.bmm(expand_lidar_coord.permute(0, 2, 1))[:, 0:2].permute(0, 2, 1)
        # (B, H*W, 2)
        aligned_lidar_coord = aligned_lidar_coord.clone()
        return aligned_lidar_coord

    @torch.no_grad()
    def align_center(self, center, vel, index, target_index, img_metas):
        batch_size = len(img_metas)

        cur_lidar2ego_list = list()
        cur_ego2global_list = list()
        prev_lidar2ego_list = list()
        prev_ego2global_list = list()

        for batch in range(batch_size):
            if target_index == 0:
                cur_poses = img_metas[batch]['poses']
            else:
                cur_poses = img_metas[batch]['prev_img_metas'][target_index - 1]['poses']
            cur_lidar2ego_single = torch.from_numpy(cur_poses['lidar2ego']).to(center.device)
            cur_ego2global_single = torch.from_numpy(cur_poses['ego2global']).to(center.device)
            cur_lidar2ego_list.append(cur_lidar2ego_single)
            cur_ego2global_list.append(cur_ego2global_single)

            prev_poses = img_metas[batch]['prev_img_metas'][index - 1]['poses']
            prev_lidar2ego_single = torch.from_numpy(prev_poses['lidar2ego']).to(center.device)
            prev_ego2global_single = torch.from_numpy(prev_poses['ego2global']).to(center.device)
            prev_lidar2ego_list.append(prev_lidar2ego_single)
            prev_ego2global_list.append(prev_ego2global_single)

        cur_lidar2ego = torch.stack(cur_lidar2ego_list, dim=0)
        cur_ego2global = torch.stack(cur_ego2global_list, dim=0)
        prev_lidar2ego = torch.stack(prev_lidar2ego_list, dim=0)
        prev_ego2global = torch.stack(prev_ego2global_list, dim=0)

        aligned_center = self._align_center(center, vel, cur_lidar2ego, cur_ego2global, prev_lidar2ego, prev_ego2global)
        return aligned_center

    def _align_boxes(self, boxes, lidar2ego, ego2global, cur_lidar2ego, cur_ego2global, motion_update=True,
                     forward=True):
        B, L, _ = boxes.size()
        boxes = boxes.clone()
        center = boxes[..., 0:3]
        rot = boxes[..., 6:7]
        vel = boxes[..., 7:9]
        extra = boxes[..., 9:]

        if motion_update:
            center[..., 0] = center[..., 0] + vel[..., 0] * 0.5 * 1 if forward else -1
            center[..., 1] = center[..., 1] + vel[..., 1] * 0.5 * 1 if forward else -1

        # (B, L, 4) (x, y, z, 1)
        expand_lidar_coord = torch.cat([center, center.new_ones((B, L, 1))], dim=-1)
        expand_lidar_vel_coord = torch.cat([vel, vel.new_ones((B, L, 1))], dim=-1)

        rot = rot + torch.atan2(cur_lidar2ego[..., 1, 0], cur_lidar2ego[..., 0, 0]).unsqueeze(-1).unsqueeze(-1)
        rot = rot + torch.atan2(cur_ego2global[..., 1, 0], cur_ego2global[..., 0, 0]).unsqueeze(-1).unsqueeze(-1)
        rot = rot - torch.atan2(ego2global[..., 1, 0], ego2global[..., 0, 0]).unsqueeze(-1).unsqueeze(-1)
        rot = rot - torch.atan2(lidar2ego[..., 1, 0], lidar2ego[..., 0, 0]).unsqueeze(-1).unsqueeze(-1)

        # TODO BUG: torch.inverse may return NaN.
        #  Use np.linalg.inv to solve the bug.
        lidar2ego = lidar2ego.cpu().numpy()
        ego2global = ego2global.cpu().numpy()
        lidar2ego_inverse = np.linalg.inv(lidar2ego)
        ego2global_inverse = np.linalg.inv(ego2global)
        lidar2ego_inverse = torch.from_numpy(lidar2ego_inverse).to(center.device)
        ego2global_inverse = torch.from_numpy(ego2global_inverse).to(center.device)
        tm = reduce(torch.bmm, [lidar2ego_inverse, ego2global_inverse, cur_ego2global, cur_lidar2ego])

        aligned_lidar_coord = tm.bmm(expand_lidar_coord.permute(0, 2, 1))[:, 0:3].permute(0, 2, 1)

        tm = reduce(torch.bmm,
                    [lidar2ego_inverse[:, 0:3, 0:3], ego2global_inverse[:, 0:3, 0:3], cur_ego2global[:, 0:3, 0:3],
                     cur_lidar2ego[:, 0:3, 0:3]])
        aligned_lidar_vel_coord = tm.bmm(expand_lidar_vel_coord.permute(0, 2, 1))[:, 0:2].permute(0, 2, 1)

        aligned_boxes = torch.cat([aligned_lidar_coord, boxes[..., 3:6], rot, aligned_lidar_vel_coord, extra], dim=-1)
        return aligned_boxes, tm

    @torch.no_grad()
    def align_boxes(self, boxes, index, target_index, img_metas, motion_update=True, forward=True):
        batch_size = len(img_metas[0])

        cur_lidar2ego_list = list()
        cur_ego2global_list = list()
        prev_lidar2ego_list = list()
        prev_ego2global_list = list()

        for batch in range(batch_size):
            cur_poses = img_metas[target_index][batch]['poses']
            cur_lidar2ego_single = torch.from_numpy(cur_poses['lidar2ego']).to(boxes.device)
            cur_ego2global_single = torch.from_numpy(cur_poses['ego2global']).to(boxes.device)
            cur_lidar2ego_list.append(cur_lidar2ego_single)
            cur_ego2global_list.append(cur_ego2global_single)

            prev_poses = img_metas[index][batch]['poses']
            prev_lidar2ego_single = torch.from_numpy(prev_poses['lidar2ego']).to(boxes.device)
            prev_ego2global_single = torch.from_numpy(prev_poses['ego2global']).to(boxes.device)
            prev_lidar2ego_list.append(prev_lidar2ego_single)
            prev_ego2global_list.append(prev_ego2global_single)

        cur_lidar2ego = torch.stack(cur_lidar2ego_list, dim=0)
        cur_ego2global = torch.stack(cur_ego2global_list, dim=0)
        prev_lidar2ego = torch.stack(prev_lidar2ego_list, dim=0)
        prev_ego2global = torch.stack(prev_ego2global_list, dim=0)

        aligned_boxes, tm = self._align_boxes(boxes, cur_lidar2ego, cur_ego2global, prev_lidar2ego, prev_ego2global,
                                          motion_update, forward)
        return aligned_boxes, tm

    def forward_ffn(self, tokens_cls, tokens_reg, cur_boxes):
        res_layer = self.prediction_heads_cls(tokens_cls)
        res_layer.update(self.prediction_heads_reg(tokens_reg))

        cur_boxes = self.lidar2bev_boxes(cur_boxes)
        cur_center = cur_boxes[..., 0:2]
        cur_height = cur_boxes[..., 2:3]
        cur_dim = cur_boxes[..., 3:6]
        cur_rot = cur_boxes[..., 6:8]
        cur_vel = cur_boxes[..., 8:10]

        res_layer['center'] = res_layer['center'] + cur_center.permute(0, 2, 1)
        res_layer['height'] = res_layer['height'] + cur_height.permute(0, 2, 1)
        res_layer['dim'] = res_layer['dim'] + cur_dim.permute(0, 2, 1)
        res_layer['rot'] = res_layer['rot'] + cur_rot.permute(0, 2, 1)
        res_layer['vel'] = res_layer['vel'] + cur_vel.permute(0, 2, 1)

        return res_layer

    @torch.no_grad()
    def mtm_attn(self, prev_center, prev_label, cur_center, cur_label):
        batch_size = prev_center.shape[0]
        # (N, M), detections: N, tracks: M
        N = cur_center.shape[1]
        M = prev_center.shape[1]
        dist = (
            ((prev_center.reshape(batch_size, 1, -1, 2) - cur_center.reshape(batch_size, -1, 1, 2)) ** 2).sum(axis=3))
        dist = torch.sqrt(dist)
        max_diff = torch.from_numpy(self.max_diff).to(prev_label.device)
        max_diff = max_diff[cur_label]
        mask = ((dist > max_diff.reshape(batch_size, N, 1)) + (
                cur_label.reshape(batch_size, N, 1) != prev_label.reshape(batch_size, 1, M))) > 0
        dist = dist + mask * 1e8

        mtm_map = (-1 * dist).softmax(dim=-1)

        return mtm_map

    def forward_temporal_fusion(self, queries, pred_results, img_metas):
        device = queries[0].device
        fused_queries_list_cls = list()
        fused_queries_list_reg = list()
        fused_boxes_list = list()
        fused_labels_list = list()
        temporal_queries_list_cls = list()
        temporal_queries_list_reg = list()

        temporal_queries_list_cls.append(queries[-1])
        temporal_queries_list_reg.append(queries[-1])
        for i in range(self.num_frames - 1):
            prev_index = self.num_frames - i - 1
            cur_index = prev_index - 1
            cur_queries = queries[cur_index].to(device)
            cur_boxes = pred_results['boxes'][cur_index].to(device)
            cur_labels = pred_results['labels'][cur_index].long().to(device)
            if len(fused_queries_list_cls) > 0:
                prev_queries_cls = fused_queries_list_cls[-1]
                prev_queries_reg = fused_queries_list_reg[-1]
                prev_boxes = fused_boxes_list[-1]
                prev_labels = fused_labels_list[-1]
            else:
                prev_queries_cls = prev_queries_reg = queries[prev_index].to(device)
                prev_boxes = pred_results['boxes'][prev_index].to(device)
                prev_labels = pred_results['labels'][prev_index].long().to(device)

            aligned_boxes, tm = self.align_boxes(prev_boxes, prev_index, cur_index, img_metas, motion_update=True)
            aligned_prev_center = aligned_boxes[..., 0:2]
            cur_center = cur_boxes[..., 0:2]

            attn_map = self.mtm_attn(aligned_prev_center, prev_labels, cur_center, cur_labels)
            temporal_queries_cls = self.mtm_decoder_cls(cur_queries, prev_queries_cls, attn_map)
            temporal_queries_reg = self.mtm_decoder_reg(cur_queries, prev_queries_reg, attn_map)
            temporal_queries_list_cls.append(temporal_queries_cls)
            temporal_queries_list_reg.append(temporal_queries_reg)

            if cur_index != 0 and self.extension:
                prev_mask = torch.max(attn_map, dim=1).values
                prev_mask = torch.topk(-1 * prev_mask, k=100, dim=-1).indices
                batch_index = \
                    torch.arange(prev_queries_cls.shape[0], dtype=torch.long).unsqueeze(-1).repeat(1, prev_mask.shape[-1])
                prev_mask_queries_cls = prev_queries_cls.permute(0, 2, 1)[batch_index, prev_mask].permute(0, 2, 1)
                prev_mask_queries_reg = prev_queries_reg.permute(0, 2, 1)[batch_index, prev_mask].permute(0, 2, 1)
                prev_mask_boxes = aligned_boxes[batch_index, prev_mask]
                prev_mask_labels = prev_labels[batch_index, prev_mask]
                fused_queries_cls = torch.cat([temporal_queries_cls, prev_mask_queries_cls], dim=-1)
                fused_queries_reg = torch.cat([temporal_queries_reg, prev_mask_queries_reg], dim=-1)
                fused_boxes = torch.cat([cur_boxes, prev_mask_boxes], dim=1)
                fused_labels = torch.cat([cur_labels, prev_mask_labels], dim=1)
            else:
                fused_queries_cls = temporal_queries_cls
                fused_queries_reg = temporal_queries_reg
                fused_boxes = cur_boxes
                fused_labels = cur_labels
            fused_queries_list_cls.append(fused_queries_cls)
            fused_queries_list_reg.append(fused_queries_reg)
            fused_boxes_list.append(fused_boxes)
            fused_labels_list.append(fused_labels)
        return temporal_queries_list_cls, temporal_queries_list_reg

    def forward_single(self, queries, img_metas, pred_results):
        assert self.num_frames == len(queries)
        self.num_proposals = queries[0].shape[-1]
        device = queries[0].device

        img_metas_list = list()
        img_metas_list.append(img_metas)
        for i in range(1, self.num_frames):
            img_metas_batch = list()
            for j in range(queries[0].shape[0]):
                img_metas_batch.append(img_metas[j]['prev_img_metas'][i - 1])
            img_metas_list.append(img_metas_batch)

        temporal_queries_cls, temporal_queries_reg = self.forward_temporal_fusion(queries, pred_results, img_metas_list)
        cur_boxes = pred_results['boxes'][0].to(device)
        cur_scores = pred_results['scores'][0].to(device)
        cur_labels = pred_results['labels'][0].to(device)
        temporal_res = self.forward_ffn(temporal_queries_cls[-1], temporal_queries_reg[-1], cur_boxes)
        temporal_res['cur_scores'] = cur_scores
        temporal_res['cur_labels'] = cur_labels

        res_dict = dict(temporal_results=temporal_res, pred_results=pred_results)
        return [res_dict]

    def forward(self, queries, img_feats, img_metas, pred_results):
        res = multi_apply(self.forward_single, queries, [img_metas], pred_results)
        assert len(res) == 1, "only support one level features."
        return res

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, preds_dict, pred_flag=False):
        """Generate training targets.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.
            preds_dicts (tuple of dict): first index by layer (default 1)
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.

                - torch.Tensor: classification target.  [BS, num_proposals]
                - torch.Tensor: classification weights (mask)  [BS, num_proposals]
                - torch.Tensor: regression target. [BS, num_proposals, 8]
                - torch.Tensor: regression weights. [BS, num_proposals, 8]
        """

        list_of_pred_dict = []
        for batch_idx in range(len(gt_bboxes_3d)):
            pred_dict = {}
            for key in preds_dict[0].keys():
                pred_dict[key] = preds_dict[0][key][batch_idx:batch_idx + 1]
            list_of_pred_dict.append(pred_dict)

        assert len(gt_bboxes_3d) == len(list_of_pred_dict)

        pred_flag = [pred_flag for _ in range(len(gt_bboxes_3d))]
        res_tuple = multi_apply(self.get_targets_single, gt_bboxes_3d, gt_labels_3d, list_of_pred_dict, pred_flag)

        labels = torch.cat(res_tuple[0], dim=0)
        label_weights = torch.cat(res_tuple[1], dim=0)
        bbox_targets = torch.cat(res_tuple[2], dim=0)
        bbox_weights = torch.cat(res_tuple[3], dim=0)
        ious = torch.cat(res_tuple[4], dim=0)
        num_pos = np.sum(res_tuple[5])
        matched_ious = np.mean(res_tuple[6])
        gt_inds = list(res_tuple[7])
        return labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious, gt_inds

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, preds_dict, pred_flag=False):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.
            preds_dict (dict): dict of prediction result for a single sample
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.

                - torch.Tensor: classification target.  [1, num_proposals]
                - torch.Tensor: classification weights (mask)  [1, num_proposals]
                - torch.Tensor: regression target. [1, num_proposals, 8]
                - torch.Tensor: regression weights. [1, num_proposals, 8]
                - torch.Tensor: iou target. [1, num_proposals]
                - int: number of positive proposals
        """
        if not pred_flag:
            num_proposals = preds_dict['center'].shape[-1]

            # get pred boxes, carefully ! donot change the network outputs
            score = copy.deepcopy(preds_dict['heatmap'].detach())
            center = copy.deepcopy(preds_dict['center'].detach())
            height = copy.deepcopy(preds_dict['height'].detach())
            dim = copy.deepcopy(preds_dict['dim'].detach())
            rot = copy.deepcopy(preds_dict['rot'].detach())
            if 'vel' in preds_dict.keys():
                vel = copy.deepcopy(preds_dict['vel'].detach())
            else:
                vel = None
            boxes_dict = self.bbox_coder.decode(score, rot, dim, center, height, vel)
            bboxes_tensor = boxes_dict[0]['bboxes']
        else:
            num_proposals = preds_dict['boxes'].shape[1]
            bboxes_tensor = preds_dict['boxes'].squeeze(0)
            scores = preds_dict['scores']
            labels = preds_dict['labels'].long()
            one_hot = F.one_hot(labels, num_classes=self.num_classes)
            score = (one_hot * scores.unsqueeze(-1)).permute(0, 2, 1)

        gt_bboxes_tensor = gt_bboxes_3d.tensor.to(score.device)
        assign_result = self.bbox_assigner.assign(bboxes_tensor, gt_bboxes_tensor, gt_labels_3d, score, self.train_cfg)

        sampling_result = self.bbox_sampler.sample(assign_result, bboxes_tensor, gt_bboxes_tensor)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        assert len(pos_inds) + len(neg_inds) == num_proposals

        # create target for loss computation
        bbox_targets = torch.zeros([num_proposals, self.bbox_coder.code_size]).to(score.device)
        bbox_weights = torch.zeros([num_proposals, self.bbox_coder.code_size]).to(score.device)
        ious = assign_result.max_overlaps
        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)
        label_weights = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)

        if gt_labels_3d is not None:  # default label is -1
            labels += self.num_classes

        # both pos and neg have classification loss, only pos has regression and iou loss
        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_gt_bboxes)

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels_3d is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels_3d[sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
        return labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], ious[None], int(
            pos_inds.shape[0]), float(mean_iou), assign_result.gt_inds


    def loss_det(self, preds_dict, labels, label_weights, num_pos, bbox_weights, bbox_targets, matched_ious, ious,
                 prefix):
        layer_labels = labels.reshape(-1)
        layer_label_weights = label_weights.reshape(-1)
        layer_score = preds_dict['heatmap']
        layer_cls_score = layer_score.permute(0, 2, 1).reshape(-1, self.num_classes)
        layer_loss_cls = self.loss_cls(layer_cls_score, layer_labels, layer_label_weights, avg_factor=max(num_pos, 1))

        layer_center = preds_dict['center']
        layer_height = preds_dict['height']
        layer_rot = preds_dict['rot']
        layer_dim = preds_dict['dim']
        preds = torch.cat([layer_center, layer_height, layer_dim, layer_rot], dim=1).permute(0, 2, 1)
        if 'vel' in preds_dict.keys():
            layer_vel = preds_dict['vel']
            preds = torch.cat([layer_center, layer_height, layer_dim, layer_rot, layer_vel], dim=1).permute(0, 2, 1)
        code_weights = self.train_cfg.get('code_weights', None)
        layer_bbox_weights = bbox_weights
        layer_reg_weights = layer_bbox_weights * layer_bbox_weights.new_tensor(code_weights)
        layer_bbox_targets = bbox_targets
        layer_loss_bbox = self.loss_bbox(preds, layer_bbox_targets, layer_reg_weights, avg_factor=max(num_pos, 1))

        loss_dict = dict()
        loss_dict[f'{prefix}_loss_cls'] = layer_loss_cls
        loss_dict[f'{prefix}_loss_bbox'] = layer_loss_bbox

        if self.iou_enable:
            layer_iou = preds_dict['iou'].squeeze(1)
            layer_loss_iou = self.loss_iou(layer_iou, ious, bbox_weights.max(-1).values,
                                           avg_factor=max(num_pos, 1))
            loss_dict[f'{prefix}_loss_iou'] = layer_loss_iou

        loss_dict[f'{prefix}_matched_ious'] = layer_loss_cls.new_tensor(matched_ious)
        return loss_dict


    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts):
        """Loss function for CenterHead.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (list[list[dict]]): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """

        temporal_res = preds_dicts[0][0]['temporal_results']

        labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious, gt_inds = self.get_targets(
            gt_bboxes_3d, gt_labels_3d, [temporal_res])

        loss_dict = dict()
        loss_det = self.loss_det(temporal_res, labels, label_weights, num_pos, bbox_weights, bbox_targets,
                                 matched_ious, ious, prefix='det')

        loss_dict.update(loss_det)
        return loss_dict

    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False, for_roi=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.

        Returns:
            list[list[dict]]: Decoded bbox, scores and labels for each layer & each batch
        """
        rets = []
        preds_dicts = [[preds_dicts[0][0]['temporal_results']]]
        for layer_id, preds_dict in enumerate(preds_dicts):
            batch_size = preds_dict[0]['heatmap'].shape[0]
            batch_score = preds_dict[0]['heatmap'][..., -self.num_proposals:].sigmoid()
            if self.iou_enable:
                iou = preds_dict[0]['iou']
                iou = torch.clamp(iou, min=0.0, max=1.0)
                det_weight = iou
            else:
                det_weight = self.det_weight

            batch_score = self.pred_weight * batch_score + det_weight * preds_dict[0]['cur_scores'].unsqueeze(dim=1)

            batch_center = preds_dict[0]['center'][..., -self.num_proposals:]
            batch_height = preds_dict[0]['height'][..., -self.num_proposals:]
            batch_dim = preds_dict[0]['dim'][..., -self.num_proposals:]
            batch_rot = preds_dict[0]['rot'][..., -self.num_proposals:]
            batch_vel = None
            if 'vel' in preds_dict[0]:
                batch_vel = preds_dict[0]['vel'][..., -self.num_proposals:]

            temp = self.bbox_coder.decode(batch_score, batch_rot, batch_dim, batch_center, batch_height, batch_vel,
                                          filter=True)

            if self.test_cfg['dataset'] == 'nuScenes':
                self.tasks = [
                    dict(num_class=8, class_names=[], indices=[0, 1, 2, 3, 4, 5, 6, 7], radius=-1),
                    dict(num_class=1, class_names=['pedestrian'], indices=[8], radius=0.175),
                    dict(num_class=1, class_names=['traffic_cone'], indices=[9], radius=0.175),
                ]

            ret_layer = []
            for i in range(batch_size):
                boxes3d = temp[i]['bboxes']
                scores = temp[i]['scores']
                labels = temp[i]['labels']
                ## adopt circle nms for different categories
                if self.test_cfg['nms_type'] != None:
                    keep_mask = torch.zeros_like(scores)
                    for task in self.tasks:
                        task_mask = torch.zeros_like(scores)
                        for cls_idx in task['indices']:
                            task_mask += labels == cls_idx
                        task_mask = task_mask.bool()
                        if task['radius'] > 0:
                            if self.test_cfg['nms_type'] == 'circle':
                                boxes_for_nms = torch.cat([boxes3d[task_mask][:, :2], scores[:, None][task_mask]],
                                                          dim=1)
                                task_keep_indices = torch.tensor(
                                    circle_nms(
                                        boxes_for_nms.detach().cpu().numpy(),
                                        task['radius'],
                                    )
                                )
                            else:
                                boxes_for_nms = xywhr2xyxyr(
                                    img_metas[i]['box_type_3d'](boxes3d[task_mask][:, :7], 7).bev)
                                top_scores = scores[task_mask]
                                task_keep_indices = nms_gpu(
                                    boxes_for_nms,
                                    top_scores,
                                    thresh=task['radius'],
                                    pre_maxsize=self.test_cfg['pre_maxsize'],
                                    post_max_size=self.test_cfg['post_maxsize'],
                                )
                        else:
                            task_keep_indices = torch.arange(task_mask.sum())
                        if task_keep_indices.shape[0] != 0:
                            keep_indices = torch.where(task_mask != 0)[0][task_keep_indices]
                            keep_mask[keep_indices] = 1
                    keep_mask = keep_mask.bool()
                    ret = dict(bboxes=boxes3d[keep_mask], scores=scores[keep_mask], labels=labels[keep_mask])
                else:  # no nms
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                ret_layer.append(ret)
            rets.append(ret_layer)
        assert len(rets) == 1
        assert len(rets[0]) == 1
        res = [[
            img_metas[0]['box_type_3d'](rets[0][0]['bboxes'], box_dim=rets[0][0]['bboxes'].shape[-1]),
            rets[0][0]['scores'],
            rets[0][0]['labels'].int()
        ]]
        return res
