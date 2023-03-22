import os
from os.path import join as pjoin

import torch


class ReconParams:
    """Recon params stored in `recon_params.pt`"""

    def __init__(self, dir_or_path):
        if os.path.isdir(dir_or_path):
            file_path = pjoin(dir_or_path, "recon_params.pt")
        else:
            file_path = dir_or_path

        recon_params = torch.load(file_path, map_location="cpu")

        # fmt: off
        self.flame_shape_params = recon_params["flame_shape_params"]  ## shape (n_frames, 300)
        self.flame_expr_params = recon_params["flame_exp_params"]    ## shape (n_frames, 9)
        self.flame_pose_params = recon_params["flame_pose_params"]    ## shape (n_frames, 100)
        self.cam_extris = recon_params["cam_extris"]                  ## shape (n_frames, 6) --- rot, trans
        self.cam_intris = recon_params["cam_intris"]                  ## shape (n_frames, 4) --- fx, fy, cx, cy
        # fmt: on

    def __len__(self):
        return len(self.flame_expr_params)

    def to(self, device):
        self.flame_shape_params = self.flame_shape_params.to(device)
        self.flame_pose_params = self.flame_pose_params.to(device)
        self.flame_expr_params = self.flame_expr_params.to(device)
        self.cam_extris = self.cam_extris.to(device)

    def get_cam_extris(self, indices):
        rot = self.cam_extris[indices, :3]
        trans = self.cam_extris[indices, 3:]

        return rot, trans

    def get_flame_params(self, indices):
        """see params of model/flame.py#FLAME.forward_geo(...)"""
        shape_params = self.flame_shape_params[indices]
        expr_params = self.flame_expr_params[indices]
        jaw_pose_params = self.flame_pose_params[indices][:, :3]
        eye_pose_params = self.flame_pose_params[indices][:, 3:]

        batch_size = len(indices)
        device = self.flame_shape_params.device
        pose_params = torch.zeros((batch_size, 6), device=device)
        neck_pose_params = torch.zeros((batch_size, 3), device=device)

        return shape_params, expr_params, jaw_pose_params, pose_params, eye_pose_params, neck_pose_params
