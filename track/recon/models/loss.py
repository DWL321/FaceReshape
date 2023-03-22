import torch
import torch.nn.functional as F
import numpy as np
from pytorch3d.ops.knn import knn_points

wflw_rigid_ids = np.concatenate((np.arange(33, 60, dtype=np.int64), np.array((60, 64, 68, 72), dtype=np.int64)), axis=0)
wflw_nonrigid_ids = np.concatenate(
    (np.setdiff1d(np.arange(33, 96, dtype=np.int64), wflw_rigid_ids), np.array(16, dtype=np.int64).reshape(-1)), axis=0
)
contour_rigid_ids = np.array((0, 1, 2, 3, 12, 13, 14, 15), dtype=np.int64)
contour_nonrigid_ids = np.setdiff1d(np.arange(0, 16, dtype=np.int64), contour_rigid_ids)

wflw_wts = np.ones((98), dtype=np.float32)
wflw_wts[[61, 62, 63, 65, 66, 67, 69, 70, 71, 73, 74, 75, 89, 90, 91, 93, 94, 95, 39, 40, 41, 47, 48, 49]] = 6.0


def subdivide_mid(input_ori):
    ## (b, n, d) -> (b, 2*n-1, d)
    input = input_ori.clone()
    b, n, d = input.shape
    mid_pts = (input[:, :-1] + input[:, 1:]) * 0.5
    subdivide_pts = torch.cat((input[:, :-1].unsqueeze(2), mid_pts.unsqueeze(2)), dim=2).reshape(b, -1, d)
    subdivide_pts = torch.cat((subdivide_pts, input[:, -1:]), dim=1)
    return subdivide_pts


def cal_lan_dist(p1, p2, lms_weight=None):
    ## (b, n, d), (b, n, d) -> (b, n)
    if lms_weight is None:
        return torch.sqrt(torch.sum((p1 - p2) ** 2, dim=2) + 1e-5)
    else:
        lms_weight = torch.as_tensor(lms_weight, device=p1.device).reshape(1, -1, 1)
        return torch.sqrt(torch.sum((p1 * lms_weight - p2 * lms_weight) ** 2, dim=2) + 1e-5)


def compute_color_loss(pred_imgs, gt_imgs, masks):
    ## (b, 3, h, w), (b, 3, h, w), (b, 1, h, w)
    ## return: L2,1 norm
    loss = torch.sqrt(torch.sum(torch.square(pred_imgs - gt_imgs), dim=1) + 1e-5) * masks.squeeze(1)
    loss = torch.sum(loss, dim=(1, 2)) / (torch.sum(masks.squeeze(1), dim=(1, 2)) + 1e-3)
    return torch.mean(loss)


def compute_lms_loss(
    pred_contour_lms, pred_wflw_lms, gt_wflw_lms, contour_lms_pose_detach=None, wflw_lms_pose_detach=None
):
    ## (b, 16, 2), (b, 98, 2), (b, 98, 2)
    ## return: L2,1 norm
    if contour_lms_pose_detach is None:
        contour_lms_pose_detach = pred_contour_lms.clone()
    if wflw_lms_pose_detach is None:
        wflw_lms_pose_detach = pred_wflw_lms.clone()
    wflw_contour_lms = gt_wflw_lms[:, :33].clone()
    wflw_contour_lms = subdivide_mid(wflw_contour_lms)
    _, _, gt_contour_lms = knn_points(pred_contour_lms.detach(), wflw_contour_lms, return_nn=True)
    gt_contour_lms = gt_contour_lms.squeeze(2)
    contour_rigid_dist = (
        cal_lan_dist(pred_contour_lms[:, contour_rigid_ids], gt_contour_lms[:, contour_rigid_ids]) * 1.0
    )
    contour_nonrigid_dist = (
        cal_lan_dist(contour_lms_pose_detach[:, contour_nonrigid_ids], gt_contour_lms[:, contour_nonrigid_ids]) * 1.0
    )
    wflw_rigid_dist = cal_lan_dist(pred_wflw_lms[:, wflw_rigid_ids], gt_wflw_lms[:, wflw_rigid_ids])
    wflw_nonrigid_dist = cal_lan_dist(
        wflw_lms_pose_detach[:, wflw_nonrigid_ids], gt_wflw_lms[:, wflw_nonrigid_ids], wflw_wts[wflw_nonrigid_ids]
    )
    return torch.mean(
        torch.cat((contour_rigid_dist, contour_nonrigid_dist, wflw_rigid_dist, wflw_nonrigid_dist), dim=-1)
    )


def compute_iris_loss(pred_wflw_lms, gt_iris_lms):
    ## (b, 108, 2), (b, 108, 2)
    ## return: L2,1 norm
    iris_dist = cal_lan_dist(pred_wflw_lms[:, 98:], gt_iris_lms)
    return torch.mean(iris_dist[iris_dist < 20.0])


def compute_reg_loss(code):
    return torch.mean(code**2)


def compute_id_loss(pred_id_feature, gt_id_feature):
    return torch.mean(1.0 - torch.sum(pred_id_feature * gt_id_feature, dim=1))


def compute_emo_loss(pred_emo_feature, gt_emo_feature):
    return torch.mean(torch.sqrt(torch.sum((pred_emo_feature - gt_emo_feature) ** 2, dim=1) + 1e-5))


def compute_para_loss(pred_para, gt_para, valid_gt=None):
    para_delta = pred_para - gt_para
    if valid_gt is None:
        return torch.mean(para_delta**2)
    else:
        return torch.mean(para_delta**2 * valid_gt) / torch.mean(valid_gt + 1e-6)
