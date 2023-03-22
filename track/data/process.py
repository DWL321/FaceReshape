import numpy as np


def calc_warp_params(ldmks, tgt_size=512, scale_coeff=0.9, lift_coeff=0.3):
    ldmks_area_center = (np.max(ldmks, axis=0) + np.min(ldmks, axis=0)) / 2
    ldmks_area_size = np.max(ldmks, axis=0) - np.min(ldmks, axis=0)
    ldmks_area_long_edge = ldmks_area_size.max() * scale_coeff
    ldmks_area_center[1] -= ldmks_area_long_edge * lift_coeff

    half_size = tgt_size / 2
    target_center = np.array((half_size, half_size), dtype=np.float32)
    warp_scale = half_size / ldmks_area_long_edge
    warp_trans = target_center - warp_scale * ldmks_area_center

    return warp_scale, warp_trans


def get_warp_mat(warp_scale, warp_trans):
    warp_mat = np.array(
        [
            [warp_scale, 0, warp_trans[0]],
            [0, warp_scale, warp_trans[1]],
        ],
        dtype=np.float32,
    )
    return warp_mat


def get_inv_warp_mat(warp_scale, warp_trans):
    inv_warp_mat = np.array(
        [
            [1 / warp_scale, 0, -warp_trans[0] / warp_scale],
            [0, 1 / warp_scale, -warp_trans[1] / warp_scale],
        ],
        dtype=np.float32,
    )
    return inv_warp_mat


def warp_cam_intris(cam_intris, warp_scale, warp_trans):
    new_cam_intris = cam_intris.clone()  ## NOTE clone before in-place operations
    new_cam_intris[:2] *= warp_scale  ## fx, fy
    new_cam_intris[2:] = new_cam_intris[2:] * warp_scale + warp_trans  ## cx, cy
    return new_cam_intris
