"""This module contains functions for geometry transform and camera projection"""
import torch
import numpy as np
from kornia.geometry.transform import get_rotation_matrix2d
import torch.nn.functional as F


def euler2rot(euler_angle):
    batch_size = euler_angle.shape[0]
    theta = euler_angle[:, 0].reshape(-1, 1, 1)
    phi = euler_angle[:, 1].reshape(-1, 1, 1)
    psi = euler_angle[:, 2].reshape(-1, 1, 1)
    one = torch.ones((batch_size, 1, 1), dtype=torch.float32, device=euler_angle.device)
    zero = torch.zeros((batch_size, 1, 1), dtype=torch.float32, device=euler_angle.device)
    rot_x = torch.cat(
        (
            torch.cat((one, zero, zero), 1),
            torch.cat((zero, theta.cos(), theta.sin()), 1),
            torch.cat((zero, -theta.sin(), theta.cos()), 1),
        ),
        2,
    )
    rot_y = torch.cat(
        (
            torch.cat((phi.cos(), zero, -phi.sin()), 1),
            torch.cat((zero, one, zero), 1),
            torch.cat((phi.sin(), zero, phi.cos()), 1),
        ),
        2,
    )
    rot_z = torch.cat(
        (
            torch.cat((psi.cos(), -psi.sin(), zero), 1),
            torch.cat((psi.sin(), psi.cos(), zero), 1),
            torch.cat((zero, zero, one), 1),
        ),
        2,
    )
    return torch.bmm(rot_x, torch.bmm(rot_y, rot_z))


def inv_warp(warp_mat):
    inv_warp_mat = warp_mat.clone()
    inv_warp_mat[:, 0, 0] = 1.0 / warp_mat[:, 0, 0]
    inv_warp_mat[:, 1, 1] = 1.0 / warp_mat[:, 1, 1]
    inv_warp_mat[:, 0, 2] = -1.0 / warp_mat[:, 0, 0] * warp_mat[:, 0, 2]
    inv_warp_mat[:, 1, 2] = -1.0 / warp_mat[:, 1, 1] * warp_mat[:, 1, 2]
    return inv_warp_mat


def estimate_transform_2d(src_pts, dst_pts, mode="similarity"):
    # src_pts: (b, n, 2), dst_pts: (b, n, 2)
    # return: (b, 2, 3)
    if mode == "similarity":
        b, n = src_pts.shape[:2]
        x, y = src_pts[:, :, :1], src_pts[:, :, 1:]
        x_d, y_d = dst_pts[:, :, :1], dst_pts[:, :, 1:]
        vec_1, vec_0 = torch.ones_like(x), torch.zeros_like(x)
        A_1 = torch.cat((x, y, vec_1, vec_0), dim=-1)
        A_2 = torch.cat((y, -x, vec_0, vec_1), dim=-1)
        A_mat = torch.cat((A_1, A_2), dim=-1).reshape(b, 2 * n, 4)
        b_vec = torch.cat((x_d, y_d), dim=-1).reshape(b, 2 * n, 1)
        x_vec = torch.linalg.lstsq(A_mat, b_vec)[0]  # (b, 4, 1)
        return torch.cat(
            (x_vec[:, 0], x_vec[:, 1], x_vec[:, 2], -x_vec[:, 1], x_vec[:, 0], x_vec[:, 3]), dim=-1
        ).reshape(b, 2, 3)
    elif mode == "crop":
        b, n = src_pts.shape[:2]
        x, y = src_pts[:, :, :1], src_pts[:, :, 1:]
        x_d, y_d = dst_pts[:, :, :1], dst_pts[:, :, 1:]
        vec_1, vec_0 = torch.ones_like(x), torch.zeros_like(x)
        A_1 = torch.cat((x, vec_1, vec_0), dim=-1)
        A_2 = torch.cat((y, vec_0, vec_1), dim=-1)
        A_mat = torch.cat((A_1, A_2), dim=-1).reshape(b, n * 2, 3)
        b_vec = torch.cat((x_d, y_d), dim=-1).reshape(b, n * 2, 1)
        x_vec = torch.linalg.lstsq(A_mat, b_vec)[0]  # (b, 3, 1)
        zero_vec = torch.zeros_like(x_vec[:, 0])
        return torch.cat((x_vec[:, 0], zero_vec, x_vec[:, 1], zero_vec, x_vec[:, 0], x_vec[:, 2]), dim=-1).reshape(
            b, 2, 3
        )
    else:
        raise NotImplementedError


def proj_pts(rott_geo, cam_para):
    fx, fy, cx, cy = cam_para[:, 0:1], cam_para[:, 1:2], cam_para[:, 2:3], cam_para[:, 3:4]
    X = rott_geo[:, :, 0]
    Y = rott_geo[:, :, 1]
    Z = rott_geo[:, :, 2]
    fxX = fx * X
    fyY = fy * Y
    proj_x = -fxX / Z + cx
    proj_y = fyY / Z + cy
    return torch.cat((proj_x[:, :, None], proj_y[:, :, None]), 2)


def rot_trans_pts(geometry, rot, trans):
    return torch.bmm(geometry, rot.permute(0, 2, 1)) + trans.unsqueeze(1)


def forward_rott(geometry, euler_angle, trans):
    rot = euler2rot(euler_angle)
    return rot_trans_pts(geometry, rot, trans)


def extract_5p(lms, lms_mode="wflw"):
    if lms_mode == "wflw":
        lm_leye = torch.mean(lms[:, [60, 64]], dim=1, keepdim=True)
        lm_reye = torch.mean(lms[:, [68, 72]], dim=1, keepdim=True)
        lm_nose = lms[:, 54:55]
        lm_lmouth = lms[:, 76:77]
        lm_rmouth = lms[:, 82:83]
        return torch.cat((lm_leye, lm_reye, lm_nose, lm_lmouth, lm_rmouth), dim=1)
    if lms_mode == "mediapipe":
        lm_idx = np.array([4, 33, 133, 398, 263, 61, 291], dtype=np.int64)
        lm5p = torch.cat(
            (
                lms[:, lm_idx[0]].unsqueeze(1),
                torch.mean(lms[:, lm_idx[[1, 2]]], dim=1).unsqueeze(1),
                torch.mean(lms[:, lm_idx[[3, 4]]], dim=1).unsqueeze(1),
                lms[:, lm_idx[5]].unsqueeze(1),
                lms[:, lm_idx[6]].unsqueeze(1),
            ),
            dim=1,
        )
        return lm5p[:, [1, 2, 0, 3, 4], :]
    else:
        raise NotImplementedError


def estimate_transform_id(pts, lms_mode="wflw", crop_size=112):
    lms_5p = extract_5p(pts, lms_mode)
    dst_5p = np.array(
        [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
        dtype=np.float32,
    )
    dst_5p = dst_5p / 112.0 * crop_size
    dst_5p = torch.as_tensor(dst_5p, device=lms_5p.device).unsqueeze(0).expand(lms_5p.shape[0], -1, -1)
    return estimate_transform_2d(lms_5p, dst_5p)


def estimate_transform_pose(pts, in_train=True, crop_size=224):
    # pts: (b, n, 2)
    b, n = pts.shape[:2]
    left_top = torch.min(pts, dim=1)[0]
    right_bottom = torch.max(pts, dim=1)[0]
    left, right, top, bottom = left_top[..., 0], right_bottom[..., 0], left_top[..., 1], right_bottom[..., 1]
    old_size = (right - left + bottom - top) / 2.0
    if in_train and (np.random.randint(5) != 0):
        old_size = old_size * np.random.uniform(1.3, 1.7)
    else:
        old_size = old_size * 1.5
    center = torch.cat(((left + right).unsqueeze(-1) * 0.5, (top + bottom).unsqueeze(-1) * 0.5), dim=-1)
    if in_train and (np.random.randint(5) != 0):
        center[:, 0] += (torch.rand_like(center[:, 0]) * 2.0 - 1.0) * old_size * 0.2
        center[:, 1] += (torch.rand_like(center[:, 0]) * (-1.0)) * old_size * 0.2
    # else:
    #     center[:, 1] += (-.2)*old_size*.15
    old_hsize = old_size.unsqueeze(-1) * 0.5
    src_pts = torch.cat(
        (
            center[:, 0:1] - old_hsize,
            center[:, 1:] - old_hsize,
            center[:, 0:1] - old_hsize,
            center[:, 1:] + old_hsize,
            center[:, 0:1] + old_hsize,
            center[:, 1:] - old_hsize,
        ),
        dim=-1,
    ).reshape(b, 3, 2)
    dst_pts = torch.zeros_like(src_pts)
    dst_pts[:, 1, 1] = crop_size - 1.0
    dst_pts[:, 2, 0] = crop_size - 1.0
    t_mat = estimate_transform_2d(src_pts, dst_pts, "crop")
    warp_mat = t_mat.clone()
    if in_train and (np.random.randint(5) != 0):
        rot_center = torch.ones_like(center) * crop_size / 2.0
        rot_angle = (torch.rand_like(center[:, 0]) * 2.0 - 1.0) * 30.0
        rot_scale = torch.ones_like(center)
        rot_mat = get_rotation_matrix2d(rot_center, rot_angle, rot_scale)
        warp_mat[:, :2, :2] = torch.bmm(rot_mat[:, :2, :2], t_mat[:, :2, :2])
        warp_mat[:, :, 2:] = torch.bmm(rot_mat[:, :2, :2], t_mat[:, :2, 2:]) + rot_mat[:, :2, 2:]
    return warp_mat


def estimate_transform_exp(pts, in_train=True, crop_size=224):
    # pts: (b, n, 2)
    b, n = pts.shape[:2]
    left_top = torch.min(pts, dim=1)[0]
    right_bottom = torch.max(pts, dim=1)[0]
    left, right, top, bottom = left_top[..., 0], right_bottom[..., 0], left_top[..., 1], right_bottom[..., 1]
    old_size = (right - left + bottom - top) / 2.0
    if in_train and (np.random.randint(5) != 0):
        old_size = old_size * np.random.uniform(0.9, 1.3)
    else:
        old_size = old_size * 1.1
    center = torch.cat(((left + right).unsqueeze(-1) * 0.5, (top + bottom).unsqueeze(-1) * 0.5), dim=-1)
    if in_train and (np.random.randint(5) != 0):
        center[:, 0] += (torch.rand_like(center[:, 0]) * 2.0 - 1.0) * old_size * 0.2
        center[:, 1] += (torch.rand_like(center[:, 0]) * (-1.0)) * old_size * 0.2

    old_hsize = old_size.unsqueeze(-1) * 0.5
    src_pts = torch.cat(
        (
            center[:, 0:1] - old_hsize,
            center[:, 1:] - old_hsize,
            center[:, 0:1] - old_hsize,
            center[:, 1:] + old_hsize,
            center[:, 0:1] + old_hsize,
            center[:, 1:] - old_hsize,
        ),
        dim=-1,
    ).reshape(b, 3, 2)
    dst_pts = torch.zeros_like(src_pts)
    dst_pts[:, 1, 1] = crop_size - 1.0
    dst_pts[:, 2, 0] = crop_size - 1.0

    t_mat = estimate_transform_2d(src_pts, dst_pts)
    warp_mat = t_mat.clone()
    if in_train and (np.random.randint(5) != 0):
        rot_center = torch.ones_like(center) * crop_size / 2.0
        rot_angle = (torch.rand_like(center[:, 0]) * 2.0 - 1.0) * 30.0
        rot_scale = torch.ones_like(center)
        rot_mat = get_rotation_matrix2d(rot_center, rot_angle, rot_scale)
        warp_mat[:, :2, :2] = torch.bmm(rot_mat[:, :2, :2], t_mat[:, :2, :2])
        warp_mat[:, :, 2:] = torch.bmm(rot_mat[:, :2, :2], t_mat[:, :2, 2:]) + rot_mat[:, :2, 2:]
    return warp_mat


def warp_lms(input_lms, warp_mat):
    input_lms = torch.cat((input_lms, torch.ones_like(input_lms[:, :, :1])), dim=-1)
    return torch.bmm(input_lms, warp_mat.permute(0, 2, 1))


def compute_vertex_normals(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of vertices, 3]
    """
    assert vertices.ndimension() == 3
    assert faces.ndimension() == 3
    assert vertices.shape[0] == faces.shape[0]
    assert vertices.shape[2] == 3
    assert faces.shape[2] == 3

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    normals = torch.zeros(bs * nv, 3).to(device)

    ex_faces = (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    faces = faces + ex_faces  # expanded faces
    # print('exfaces : ', ex_faces.shape)
    vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]

    faces = faces.view(-1, 3)
    vertices_faces = vertices_faces.view(-1, 3, 3)
    # print(faces, vertices_faces)

    normals.index_add_(
        0,
        faces[:, 1].long(),
        torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1], vertices_faces[:, 0] - vertices_faces[:, 1]),
    )
    normals.index_add_(
        0,
        faces[:, 2].long(),
        torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2], vertices_faces[:, 1] - vertices_faces[:, 2]),
    )
    normals.index_add_(
        0,
        faces[:, 0].long(),
        torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0], vertices_faces[:, 2] - vertices_faces[:, 0]),
    )

    normals = F.normalize(normals, eps=1e-6, dim=1)
    normals = normals.reshape((bs, nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return normals
