import torch
import time
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu
from mmdet.models import DETECTORS
from mmdet3d.core import (bbox3d2result, xywhr2xyxyr)
from .mvx_two_stage import MVXTwoStageDetector


@DETECTORS.register_module()
class QTNetDetector(MVXTwoStageDetector):
    def __init__(self, **kwargs):
        super(QTNetDetector, self).__init__(**kwargs)

    def forward_train(self,
                      queries=None,
                      pred_results=None,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        losses = dict()
        losses_pts = self.forward_pts_train(queries, pred_results, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses

    def forward_pts_train(self,
                          queries,
                          pred_results,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        num_frames = queries.shape[1]
        queries = [queries[:, i] for i in range(num_frames)]
        for key in pred_results.keys():
            pred_results[key] = [pred_results[key][:, i] for i in range(num_frames)]
        outs = self.pts_bbox_head([queries], None, img_metas, [pred_results])
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def forward_test(self, queries, img_metas, pred_results=None, img=None, **kwargs):
        img_feats = [None]

        return self.simple_test(queries[0], img_metas[0], pred_results[0], img_feats[0], **kwargs)

    def simple_test(self, queries, img_metas, pred_results=None, img_feats=None, rescale=False):
        bbox_list = [dict() for i in range(len(img_metas))]
        if queries is not None and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                queries, img_feats, img_metas, pred_results, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def simple_test_pts(self, x, x_img, img_metas, pred_results=None, rescale=False):
        x = [x[:, i] for i in range(x.shape[1])]
        pred = dict()
        for key in pred_results.keys():
            pred[key] = [pred_results[key][:, i] for i in range(pred_results[key].shape[1])]

        outs = self.pts_bbox_head([x], x_img, img_metas, [pred])
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results



