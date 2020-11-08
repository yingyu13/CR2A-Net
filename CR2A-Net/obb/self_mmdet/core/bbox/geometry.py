import torch
from mmdet.core.bbox import bbox_overlaps
# from bbox_v2 import bbox_overlaps_cython_v2
import numpy as np
import obb.DOTA_devkit.polyiou as polyiou
from obb.self_mmdet.core.bbox.transforms_rbbox import RotBox2Polys, poly2bbox, mask2poly, Tuplelist2Polylist

def rbbox_overlaps_cy_warp(rbboxes, query_boxes):
    # TODO: first calculate the hbb overlaps, for overlaps > 0, calculate the obb overlaps
    # import pdb
    # pdb.set_trace()
    box_device = query_boxes.device
    query_boxes_np = query_boxes.cpu().numpy().astype(np.float)

    # polys_np = RotBox2Polys(boxes_np)
    # TODO: change it to only use pos gt_masks
    # polys_np = mask2poly(gt_masks)
    # polys_np = np.array(Tuplelist2Polylist(polys_np)).astype(np.float)

    polys_np = RotBox2Polys(rbboxes).astype(np.float)
    query_polys_np = RotBox2Polys(query_boxes_np)

    h_bboxes_np = poly2bbox(polys_np)
    h_query_bboxes_np = poly2bbox(query_polys_np)

    # hious
    ious = bbox_overlaps(h_bboxes_np, h_query_bboxes_np)
    import pdb
    # pdb.set_trace()
    inds = np.where(ious > 0)
    for index in range(len(inds[0])):
        box_index = inds[0][index]
        query_box_index = inds[1][index]

        box = polys_np[box_index]
        query_box = query_polys_np[query_box_index]

        # calculate obb iou
        # import pdb
        # pdb.set_trace()
        overlap = polyiou.iou_poly(polyiou.VectorDouble(box), polyiou.VectorDouble(query_box))
        ious[box_index][query_box_index] = overlap

    return torch.from_numpy(ious).to(box_device)

def rbbox_overlaps_hybrid(boxes, query_boxes):
    # TODO: first calculate the hbb overlaps, for overlaps > 0, use the gpu_overlaps to calculate the obb overlaps
    # box_device = boxes.device
    pass

def rbbox_overlaps_cy(boxes_np, query_boxes_np):
    # TODO: first calculate the hbb overlaps, for overlaps > 0, calculate the obb overlaps

    polys_np = RotBox2Polys(boxes_np).astype(np.float)
    query_polys_np = RotBox2Polys(query_boxes_np).astype(np.float)

    h_bboxes_np = poly2bbox(polys_np).astype(np.float)
    h_query_bboxes_np = poly2bbox(query_polys_np).astype(np.float)

    # hious
    ious = bbox_overlaps(h_bboxes_np, h_query_bboxes_np)
    import pdb
    # pdb.set_trace()
    inds = np.where(ious > 0)
    for index in range(len(inds[0])):
        box_index = inds[0][index]
        query_box_index = inds[1][index]

        box = polys_np[box_index]
        query_box = query_polys_np[query_box_index]

        # calculate obb iou
        # import pdb
        # pdb.set_trace()
        overlap = polyiou.iou_poly(polyiou.VectorDouble(box), polyiou.VectorDouble(query_box))
        ious[box_index][query_box_index] = overlap

    return ious

