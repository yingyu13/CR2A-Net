import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.ops import DeformConv
from mmcv.cnn import ConvModule, bias_init_with_prob
from mmdet.models.builder import HEADS, build_loss
from obb.self_mmdet.core import (delta2dbbox, delta2dbbox_v3, multiclass_nms_rbbox)
from obb.self_mmdet.models.anchor_heads.cascade_anchor_head import CascadeAnchorHeadRbbox

class AdaptiveConv(nn.Module):
    """ Adaptive Conv is built based on Deformable Conv
    with precomputed offsets which derived from anchors"""

    def __init__(self, in_channels, out_channels, dilation=1, adapt=True):
        super(AdaptiveConv, self).__init__()
        self.adapt = adapt
        if self.adapt:
            assert dilation == 1
            self.conv = DeformConv(in_channels, out_channels, 3, padding=1)
        else:  # fallback to normal Conv2d
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilation,
                dilation=dilation)

    def init_weights(self):
        normal_init(self.conv, std=0.01)

    def forward(self, x, offset):
        if self.adapt:
            N, _, H, W = x.shape
            assert offset is not None
            assert H * W == offset.shape[1]
            # reshape [N, NA, 18] to (N, 18, H, W), 因为offset是每个channel上都一样的，所以这里不需要考虑C，N为batch_size
            offset = offset.permute(0, 2, 1).reshape(N, -1, H, W)
            x = self.conv(x, offset)
        else:
            assert offset is None
            x = self.conv(x)
        return x


@HEADS.register_module()
class CascadeRPNHead(CascadeAnchorHeadRbbox):

    def __init__(self,
                 in_channels,
                 stacked_convs=4,
                 feat_adapt=False,
                 dilation=1,
                 bridged_feature=False,
                 with_module=False,
                 hbb_trans='hbb2obb_v2',
                 conv_cfg=None,
                 norm_cfg=None,
                 **kwargs):
        """CascadeRPNHead
        Args:
            in_channels (int): Number of channels in the input feature map.
            feat_adapt (bool): Whether to use adaptive convolution.
            dilation (int): Dilation factor of rpn_conv
            bridged_feature (bool): Whether to use bridge feature.
        """  # noqa: W605
        super(CascadeRPNHead, self).__init__(2, in_channels, **kwargs)
        self.feat_adapt = feat_adapt
        self.dilation = dilation
        self.bridged_feature = bridged_feature
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_module = with_module
        self.hbb_trans = hbb_trans
        self._init_layers()

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.rpn_conv = AdaptiveConv(
            self.in_channels,
            self.feat_channels,
            dilation=self.dilation,
            adapt=self.feat_adapt)
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 5, 3, padding=1)

    def init_weights(self):
        self.rpn_conv.init_weights()
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

    def forward_single(self, x, offset):
        bridged_x = x
        x = self.relu(self.rpn_conv(x, offset))
        if self.bridged_feature:
            bridged_x = x  # update feature
        cls_feat = x
        reg_feat = x
        if self.with_cls:
            for cls_conv in self.cls_convs:
                cls_feat = cls_conv(cls_feat)
            cls_score = self.retina_cls(cls_feat)
        else:
            cls_score = None
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return bridged_x, cls_score, bbox_pred

    def loss(self,
             anchor_list,
             valid_flag_list,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_masks,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        losses = super(CascadeRPNHead, self).loss(
            anchor_list,
            valid_flag_list,
            cls_scores,
            bbox_preds,
            gt_bboxes,
            gt_masks,
            gt_labels,
            img_metas,
            cfg,
            gt_bboxes_ignore=gt_bboxes_ignore)
        if self.with_cls:
            return dict(
                loss_rpn_cls=losses['loss_cls'],
                loss_rpn_reg=losses['loss_reg'])
        return dict(loss_rpn_reg=losses['loss_reg'])

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        mlvl_bboxes = []
        mlvl_scores = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            anchors = mlvl_anchors[idx]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                scores = rpn_cls_score.softmax(dim=1)[:, 1]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 5)  # 这里注意
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                _, topk_inds = scores.topk(cfg.nms_pre)
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
                scores = scores[topk_inds]
            # proposals = delta2bbox(anchors, rpn_bbox_pred, self.target_means,
            #                       self.target_stds, img_shape)

            # rbbox_ex_anchors = hbb2obb_v2(anchors)
            rbbox_ex_anchors = anchors  # 这里注意，对于cascade_rpn来说，最后一个head的anchor是rotate的
            if self.with_module:
                bboxes = delta2dbbox(rbbox_ex_anchors, rpn_bbox_pred, self.target_means,
                                     self.target_stds, img_shape)
            else:
                bboxes = delta2dbbox_v3(rbbox_ex_anchors, rpn_bbox_pred, self.target_means,
                                        self.target_stds, img_shape)

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            # mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
            mlvl_bboxes[:, :4] /= mlvl_bboxes[:, :4].new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores).reshape(-1, 1)
        # print(mlvl_bboxes.shape)
        # print(mlvl_scores.shape)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        # det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
        #                                         cfg.score_thr, cfg.nms,
        #                                         cfg.max_per_img)
        det_bboxes, det_labels = multiclass_nms_rbbox(mlvl_bboxes, mlvl_scores,
                                                      cfg.score_thr, cfg.nms,
                                                      cfg.max_per_img)
        # for item in det_bboxes:
        #     print(item.shape)
        # for item in det_labels:
        #     print(item.shape)
        return det_bboxes, det_labels
