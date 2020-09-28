import torch
from obb.self_mmdet.core.bbox.transforms_rbbox import RotBox2Polys_torch


def anchor_offset(anchor_list, anchor_strides, featmap_sizes):
    """ Get offest for deformable conv based on anchor shape
    NOTE: currently support deformable kernel_size=3 and dilation=1
    Args:
        anchor_list (list[list[tensor])): [NI, NLVL, NA, 4] list of
            multi-level anchors
            anchor_strides (list): anchor stride of each level
    Returns:
        offset_list (list[tensor]): [NLVL, NA, 2, 18]: offset of 3x3 deformable
        kernel.
    """

    def _shape_offset(anchors, stride):
        # currently support kernel_size=3 and dilation=1
        ks = 3
        dilation = 1
        pad = (ks - 1) // 2
        idx = torch.arange(-pad, pad + 1, dtype=dtype, device=device)  # [-1, 0, 1]
        yy, xx = torch.meshgrid(idx, idx)  # return order matters
        # yy = tensor([[-1, -1, -1],
        #         [ 0,  0,  0],
        #         [ 1,  1,  1]])
        # xx = tensor([[-1,  0,  1],
        #         [-1,  0,  1],
        #         [-1,  0,  1]]
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        # xx = tensor([-1, 0, 1, -1,  0,  1, -1,  0,  1])
        # yy = tensor([-1, -1, -1, 0,  0,  0, 1,  1,  1])
        w = (anchors[:, 2] - anchors[:, 0] + 1) / stride  # anchor的w和h
        h = (anchors[:, 3] - anchors[:, 1] + 1) / stride
        w = w / (ks - 1) - dilation
        h = h / (ks - 1) - dilation
        offset_x = w[:, None] * xx  # (NA, ks**2)
        offset_y = h[:, None] * yy  # (NA, ks**2)
        return offset_x, offset_y

    def _ctr_offset(anchors, stride, featmap_size):
        feat_h, feat_w = featmap_size
        assert len(anchors) == feat_h * feat_w

        x = (anchors[:, 0] + anchors[:, 2]) * 0.5
        y = (anchors[:, 1] + anchors[:, 3]) * 0.5
        # compute centers on feature map
        x = (x - (stride - 1) * 0.5) / stride
        y = (y - (stride - 1) * 0.5) / stride
        # compute predefine centers
        xx = torch.arange(0, feat_w, device=anchors.device)
        yy = torch.arange(0, feat_h, device=anchors.device)
        yy, xx = torch.meshgrid(yy, xx)
        xx = xx.reshape(-1).type_as(x)
        yy = yy.reshape(-1).type_as(y)

        offset_x = x - xx  # (NA, )
        offset_y = y - yy  # (NA, )
        return offset_x, offset_y

    def ranchor_offset(anchors, stride, featmap_size):
        feat_h, feat_w = featmap_size
        assert len(anchors) == feat_h * feat_w

        anchors = RotBox2Polys_torch(anchors)  # 这个还挺好用的哈哈
        # print(anchors.shape)
        # print(featmap_size)
        x1 = anchors[:, 0]
        y1 = anchors[:, 1]
        x2 = anchors[:, 2]
        y2 = anchors[:, 3]
        x3 = anchors[:, 4]
        y3 = anchors[:, 5]
        x4 = anchors[:, 6]
        y4 = anchors[:, 7]
        x12_mid = (x1 + x2) * 0.5
        y12_mid = (y1 + y2) * 0.5
        x23_mid = (x2 + x3) * 0.5
        y23_mid = (y2 + y3) * 0.5
        x34_mid = (x3 + x4) * 0.5
        y34_mid = (y3 + y4) * 0.5
        x41_mid = (x4 + x1) * 0.5
        y41_mid = (y4 + y1) * 0.5
        x_ctr = (x12_mid + x34_mid) * 0.5
        y_ctr = (y12_mid + y34_mid) * 0.5
        # compute centers on feature map
        x1 = (x1 - (stride - 1) * 0.5) / stride
        y1 = (y1 - (stride - 1) * 0.5) / stride
        x2 = (x2 - (stride - 1) * 0.5) / stride
        y2 = (y2 - (stride - 1) * 0.5) / stride
        x3 = (x3 - (stride - 1) * 0.5) / stride
        y3 = (y3 - (stride - 1) * 0.5) / stride
        x4 = (x4 - (stride - 1) * 0.5) / stride
        y4 = (y4 - (stride - 1) * 0.5) / stride
        x12_mid = (x12_mid - (stride - 1) * 0.5) / stride
        y12_mid = (y12_mid - (stride - 1) * 0.5) / stride
        x23_mid = (x23_mid - (stride - 1) * 0.5) / stride
        y23_mid = (y23_mid - (stride - 1) * 0.5) / stride
        x34_mid = (x34_mid - (stride - 1) * 0.5) / stride
        y34_mid = (y34_mid - (stride - 1) * 0.5) / stride
        x41_mid = (x41_mid - (stride - 1) * 0.5) / stride
        y41_mid = (y41_mid - (stride - 1) * 0.5) / stride
        x_ctr = (x_ctr - (stride - 1) * 0.5) / stride
        y_ctr = (y_ctr - (stride - 1) * 0.5) / stride
        x1 = x1 - x_ctr
        y1 = y1 - y_ctr
        x2 = x2 - x_ctr
        y2 = y2 - y_ctr
        x3 = x3 - x_ctr
        y3 = y3 - y_ctr
        x4 = x4 - x_ctr
        y4 = y4 - y_ctr
        x12_mid = x12_mid - x_ctr
        y12_mid = y12_mid - y_ctr
        x23_mid = x23_mid - x_ctr
        y23_mid = y23_mid - y_ctr
        x34_mid = x34_mid - x_ctr
        y34_mid = y34_mid - y_ctr
        x41_mid = x41_mid - x_ctr
        y41_mid = y41_mid - y_ctr
        # currently support kernel_size=3 and dilation=1
        ks = 3
        dilation = 1
        pad = (ks - 1) // 2
        idx = torch.arange(-pad, pad + 1, dtype=dtype, device=device)  # [-1, 0, 1]
        yy, xx = torch.meshgrid(idx, idx)  # return order matters
        # yy = tensor([[-1, -1, -1],
        #         [ 0,  0,  0],
        #         [ 1,  1,  1]])
        # xx = tensor([[-1,  0,  1],
        #         [-1,  0,  1],
        #         [-1,  0,  1]]
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        # print(x1.shape) # 不错不错，就是这样
        # print(xx.shape)
        offset_x1 = x1 - xx[0]
        offset_y1 = y1 - yy[0]
        offset_x12_mid = x12_mid - xx[1]
        offset_y12_mid = y12_mid - yy[1]
        offset_x2 = x2 - xx[2]
        offset_y2 = y2 - yy[2]
        offset_x41_mid = x41_mid - xx[3]
        offset_y41_mid = y41_mid - yy[3]
        offset_x1_ctr = 0
        offset_y1_ctr = 0
        offset_x23_mid = x23_mid - xx[5]
        offset_y23_mid = y23_mid - yy[5]
        offset_x4 = x4 - xx[6]
        offset_y4 = y4 - yy[6]
        offset_x34_mid = x34_mid - xx[7]
        offset_y34_mid = y34_mid - yy[7]
        offset_x3 = x3 - xx[8]
        offset_y3 = y3 - yy[8]
        offset_x1 = offset_x1.reshape(-1, 1)
        offset_x2 = offset_x2.reshape(-1, 1)
        offset_x3 = offset_x3.reshape(-1, 1)
        offset_x4 = offset_x4.reshape(-1, 1)
        offset_x12_mid = offset_x12_mid.reshape(-1, 1)
        offset_x23_mid = offset_x23_mid.reshape(-1, 1)
        offset_x34_mid = offset_x34_mid.reshape(-1, 1)
        offset_x41_mid = offset_x41_mid.reshape(-1, 1)
        offset_x1_ctr = torch.zeros_like(offset_x1)
        offset_y1 = offset_y1.reshape(-1, 1)
        offset_y2 = offset_y2.reshape(-1, 1)
        offset_y3 = offset_y3.reshape(-1, 1)
        offset_y4 = offset_y4.reshape(-1, 1)
        offset_y12_mid = offset_y12_mid.reshape(-1, 1)
        offset_y23_mid = offset_y23_mid.reshape(-1, 1)
        offset_y34_mid = offset_y34_mid.reshape(-1, 1)
        offset_y41_mid = offset_y41_mid.reshape(-1, 1)
        offset_y1_ctr = torch.zeros_like(offset_y1)
        shape_offset_x = torch.stack(
            [offset_x1, offset_x12_mid, offset_x2, offset_x41_mid, offset_x1_ctr, offset_x23_mid, offset_x4,
             offset_x34_mid, offset_x3], dim=1).reshape(-1, ks ** 2)
        shape_offset_y = torch.stack(
            [offset_y1, offset_y12_mid, offset_y2, offset_y41_mid, offset_y1_ctr, offset_y23_mid, offset_y4,
             offset_y34_mid, offset_y3], dim=1).reshape(-1, ks ** 2)
        # compute predefine centers
        xx_ctr = torch.arange(0, feat_w, device=anchors.device)
        yy_ctr = torch.arange(0, feat_h, device=anchors.device)
        yy_ctr, xx_ctr = torch.meshgrid(yy_ctr, xx_ctr)
        xx_ctr = xx_ctr.reshape(-1).type_as(x_ctr)
        yy_ctr = yy_ctr.reshape(-1).type_as(y_ctr)
        ctr_offset_x = x_ctr - xx_ctr  # (NA, )
        ctr_offset_y = y_ctr - yy_ctr  # (NA, )

        # print(shape_offset_x.shape)
        # print(ctr_offset_x.shape)

        offset_x = shape_offset_x + ctr_offset_x[:, None]
        offset_y = shape_offset_y + ctr_offset_y[:, None]

        return offset_x, offset_y

    num_imgs = len(anchor_list)
    num_lvls = len(anchor_list[0])
    dtype = anchor_list[0][0].dtype
    device = anchor_list[0][0].device
    num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]

    offset_list = []
    for i in range(num_imgs):
        mlvl_offset = []
        for lvl in range(num_lvls):
            offset_x, offset_y = ranchor_offset(anchor_list[i][lvl],
                                                anchor_strides[lvl],
                                                featmap_sizes[lvl])

            # offset order (y0, x0, y1, x2, .., y8, x8, y9, x9)
            offset = torch.stack([offset_y, offset_x], dim=-1)
            offset = offset.reshape(offset.size(0), -1)  # [NA, 2*ks**2]
            # print(offset.shape)
            mlvl_offset.append(offset)
        offset_list.append(torch.cat(mlvl_offset))  # [totalNA, 2*ks**2]
    offset_list = images_to_levels(offset_list, num_level_anchors)
    return offset_list


def images_to_levels(target, num_level_anchors):
    """Convert targets by image to targets by feature level.
    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_level_anchors:
        end = start + n
        level_targets.append(target[:, start:end])
        start = end
    return level_targets