import os
from os.path import join as pjoin

import kornia.augmentation as Aug
import numpy as np
import torch
import torch.nn as nn
from kornia.geometry.transform import warp_affine
from pytorch3d.io import load_obj

from ..flame import FLAME, flame_config
from ..utils.geo_transform import compute_vertex_normals, estimate_transform_pose, forward_rott, inv_warp, warp_lms
from .models.feature_extractor import EmoCnnModule, PoseLitCnnModule, ShapeCnnModule
from .models.loss import compute_iris_loss, compute_lms_loss, compute_para_loss, compute_reg_loss
from .models.regressor import ExpMapper, PoseLitMapper, ShapeMapper
from .models.render_uv_nvdiff import MeshRenderer

from ..utils.storage import download_pretrained_models


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def eval_with_batchnorm(nets):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        for m in net.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()
                m.track_running_stats = False
                m.weight.requires_grad = False
                m.bias.requires_grad = False
            if isinstance(m, torch.nn.BatchNorm1d):
                m.eval()
                m.track_running_stats = False
                m.weight.requires_grad = False
                m.bias.requires_grad = False


class FaceRecon(nn.Module):
    def __init__(self, pipeline_config):
        super().__init__()

        root = os.path.expanduser("~/.idrfacetrack")
        emo_encoder_path = pjoin(root, "pretrained/recon_modules/Emotion_recog.ckpt")
        shape_encoder_path = pjoin(root, "pretrained/recon_modules/mica.tar")
        poselit_encoder_path = pjoin(root, "pretrained/recon_modules/deca_model.tar")

        if (
            not os.path.exists(flame_config.flame_model_path)
            or not os.path.exists(flame_config.template_mesh_file)
            or not os.path.exists(emo_encoder_path)
        ):
            download_pretrained_models()

        self.config = pipeline_config
        self.render_size = pipeline_config.render_size
        self.flame_model = FLAME(flame_config)
        self.emo_encoder = EmoCnnModule(emo_encoder_path)
        self.shape_encoder = ShapeCnnModule(shape_encoder_path)
        self.poselit_encoder = PoseLitCnnModule(poselit_encoder_path)
        self.exp_mapper = ExpMapper(flame_config.n_exp + 3)  ## exp_para & jaw para
        self.shape_mapper = ShapeMapper(shape_encoder_path)
        self.poselit_mapper = PoseLitMapper()

        self.model_keys = [
            "flame_model",
            "shape_encoder",
            "emo_encoder",
            "poselit_encoder",
            "shape_mapper",
            "exp_mapper",
            "poselit_mapper",
        ]
        self.state_keys = self.model_keys[2:]

        self.optable_encoder_keys = []
        self.optable_mapper_keys = []

        if pipeline_config.train_shape:
            self.optable_mapper_keys.append("shape_mapper")
        if pipeline_config.train_poselit:
            self.optable_encoder_keys.append("poselit_encoder")
            self.optable_mapper_keys.append("poselit_mapper")
        if pipeline_config.train_exp:
            self.optable_encoder_keys.append("emo_encoder")
            self.optable_mapper_keys.append("exp_mapper")

        self.aug = Aug.AugmentationSequential(
            Aug.ColorJiggle(brightness=0.6, contrast=0.5, saturation=0.15, hue=0.15, p=0.5),
            Aug.RandomGrayscale(p=0.1),
            Aug.RandomGaussianNoise(std=0.12, p=0.1),
            Aug.RandomGaussianBlur((5, 5), (0.1, 2.0), p=0.1),
            Aug.RandomMotionBlur(5, 30.0, 0.5, p=0.1),
            same_on_batch=False,
        )

        self.color_jitter = Aug.ColorJiggle(brightness=0.8, contrast=0.5, saturation=0.15, hue=0.15, p=0.5)
        cam_para = torch.tensor((512.0, 512.0, self.render_size / 2.0, self.render_size / 2.0)).float().reshape(1, -1)
        self.register_buffer("cam_para", cam_para)
        _, faces, aux = load_obj(flame_config.template_mesh_file, load_textures=False)
        uv_coords = aux.verts_uvs[None, ...]  # (1, V, 2)
        uv_coords[..., 1] = 1.0 - uv_coords[..., 1]
        uv_faces = faces.textures_idx[None, ...]  # (1, F, 3)
        tri_faces = faces.verts_idx[None, ...]  # (1, F, 3)
        self.register_buffer("uv_coords", uv_coords)
        self.register_buffer("uv_faces", uv_faces)
        self.register_buffer("tri_faces", tri_faces)
        self.renderer = MeshRenderer(
            cam_para.reshape(-1).numpy(), [self.render_size, self.render_size], pipeline_config.gpu_ids
        )

        mediapipe_upper_ids = np.array(
            (
                162,
                21,
                54,
                103,
                67,
                109,
                10,
                338,
                297,
                332,
                284,
                251,
                389,
                139,
                71,
                68,
                104,
                69,
                108,
                151,
                337,
                299,
                333,
                298,
                301,
            ),
            dtype=np.int64,
        )
        self.mediapipe_inner_ids = np.setdiff1d(np.arange(0, 478, dtype=np.int64), mediapipe_upper_ids)

        self.agg_id_code = None
        self.agg_tex_code = None
        self.agg_light_code = None
        self.max_agg_num = 100
        self.agg_num = 0

    ### for exp conversion
    def convert_exp(self, poselit_code, exp_code):
        jaw_pose, eyes_pose = poselit_code[:, 6:9], poselit_code[:, 9:15]
        geometry = self.flame_model.forward_geo(
            exp_code.new_zeros((exp_code.shape[0], self.n_shape)), exp_code, jaw_pose, eye_pose_params=eyes_pose
        )
        frontal_geo = geometry[:, self.frontal_verts_ids].reshape(exp_code.shape[0], -1)
        return torch.mm(frontal_geo - self.frontal_mean, self.frontal_base)

    def load_model(self, ckpt_path):
        if (ckpt_path is not None) and os.path.isfile(ckpt_path):
            state_dicts = torch.load(ckpt_path, map_location="cpu")
            for state_key in self.state_keys:
                if state_key in state_dicts:
                    self.__getattr__(state_key).load_state_dict(state_dicts[state_key])
        # self.emo_encoder.freezer()

        ### fix shape code extractor pretrained by MICA
        fixed_module_keys = []
        if not self.config.train_shape:
            fixed_module_keys.extend(["shape_encoder", "shape_mapper"])
        if not self.config.train_exp:
            fixed_module_keys.extend(["emo_encoder", "exp_mapper"])
        if not self.config.train_poselit:
            fixed_module_keys.extend(["poselit_encoder", "poselit_mapper"])
        fixed_modules = []
        for module_key in fixed_module_keys:
            fixed_modules.append(self.__getattr__(module_key))
            self.__getattr__(module_key).eval()
        set_requires_grad(fixed_modules, False)
        # eval_with_batchnorm([self.shape_encoder, self.shape_mapper, self.id_recognet, self.emo_recognet])
        eval_with_batchnorm(fixed_modules)

    def get_model_dicts(self):
        state_dicts = {}
        for state_key in self.state_keys:
            state_dicts[state_key] = self.__getattr__(state_key).state_dict()
        return state_dicts

    def parameters_to_optimize(self):
        parameters_encoder = list()
        parameters_mapper = list()
        for state_key in self.optable_encoder_keys:
            parameters_encoder += list(self.__getattr__(state_key).parameters())
        for state_key in self.optable_mapper_keys:
            parameters_mapper += list(self.__getattr__(state_key).parameters())
        return [
            {"params": parameters_encoder, "lr": self.config.lr_encoder},
            {"params": parameters_mapper, "lr": self.config.lr_mapper},
        ]

    def forward(self, data_input, in_train=True, with_vis=False):
        codedict = self.encode(data_input, in_train)
        output = self.decode(codedict, with_vis)
        vis_dict = {}
        if with_vis:
            vis_dict["rendered_geo"] = output["rendered_geo"]
            vis_dict["input_imgs"] = codedict["imgs"]
            vis_dict["gt_lms"] = torch.cat((codedict["lms"], codedict["iris"]), dim=1)
            vis_dict["pred_lms"] = torch.cat(
                (output["contour_lms"], output["wflw_lms"][:, 33:96], output["wflw_lms"][:, 98:]), dim=1
            ).detach()
        losses = self.comput_losses(codedict, output)
        return losses, vis_dict

    def forward_test(self, imgs, ldmks):
        img_size = imgs.shape[2:]
        # inner_lms = ldmks[:, self.mediapipe_inner_ids]
        inner_lms = ldmks.clone()

        exp_code = self.exp_mapper(self.emo_encoder(imgs, inner_lms, in_train=False))

        pose_warp_mat = estimate_transform_pose(inner_lms, False, self.render_size)
        posed_imgs = warp_affine(imgs, pose_warp_mat, (self.render_size, self.render_size))

        poselit_code = self.poselit_mapper(self.poselit_encoder(posed_imgs, None))
        cam_para = torch.tensor((512.0, 512.0, 112.0, 112.0), dtype=torch.float32)
        scale = float(pose_warp_mat[0, 0, 0])
        tx, ty = float(pose_warp_mat[0, 0, 2]), float(pose_warp_mat[0, 1, 2])
        cam_para[:2] /= scale
        cam_para[2] = (cam_para[2] - tx) / scale
        cam_para[3] = (cam_para[3] - ty) / scale

        euler_angle, trans, jaw_pose, eyes_pose, exp_para = (
            poselit_code[:, :3],
            poselit_code[:, 3:6],
            exp_code[:, -3:],
            poselit_code[:, 6:12],
            exp_code[:, :-3],
        )

        shape_code = self.shape_mapper(self.shape_encoder(imgs, ldmks))

        trans[:, 2] -= 0.8  ## init trans: -0.8m

        result = (  ## shape_code, exp_para, flame_pose_params, cam_poses, cam_para
            shape_code.detach().cpu(),
            exp_para.detach().cpu(),
            torch.cat([jaw_pose, eyes_pose], dim=1).detach().cpu(),
            torch.cat((euler_angle, trans), dim=1).detach().cpu(),
            cam_para,
        )

        ## visualization
        batch_size = euler_angle.shape[0]
        geometry = self.flame_model.forward_geo(shape_code, exp_para, jaw_pose, eye_pose_params=eyes_pose)
        rott_geo = forward_rott(geometry, euler_angle, trans)
        vertex_normals = compute_vertex_normals(geometry, self.tri_faces.expand(batch_size, -1, -1))
        pred_contour_lms, pred_wflw_lms, _, _ = self.flame_model.forward_landmarks(
            rott_geo, self.cam_para.expand(batch_size, -1), None
        )

        rendered_geo = self.renderer.forward_test(
            rott_geo,
            self.tri_faces.expand(batch_size, -1, -1),
            self.uv_coords.expand(batch_size, -1, -1),
            self.uv_faces.expand(batch_size, -1, -1),
            vertex_normals,
            cam_para,
            img_size,
        )

        vis_dict = {}
        vis_dict["rendered_geo"] = rendered_geo
        vis_dict["pred_lms"] = warp_lms(
            torch.cat((pred_contour_lms, pred_wflw_lms[:, 33:96], pred_wflw_lms[:, 98:]), dim=1).detach(),
            inv_warp(pose_warp_mat),
        )

        return result, vis_dict

    def encode(self, data_input, in_train=True):

        input_images, input_lms = data_input["imgs"], data_input["lms"]

        # cv2.imwrite('input_ori.jpg', (input_images[0]*255).byte().permute(1,2,0).cpu().numpy()[:, :, [2,1,0]])

        shape_code = self.shape_mapper(self.shape_encoder(input_images, input_lms))
        codedict = {}

        input_images = self.aug(input_images)

        exp_code = self.exp_mapper(self.emo_encoder(input_images, input_lms, in_train))
        pose_warp_mat = estimate_transform_pose(input_lms, in_train, self.render_size)
        posed_imgs = warp_affine(input_images, pose_warp_mat, (self.render_size, self.render_size))
        posed_lms = warp_lms(input_lms, pose_warp_mat)
        poselit_code = self.poselit_mapper(self.poselit_encoder(posed_imgs, None))

        codedict["iris"] = warp_lms(data_input["iris"], pose_warp_mat)
        codedict["shape"] = shape_code
        codedict["exp"] = exp_code
        codedict["poselit"] = poselit_code
        codedict["imgs"] = posed_imgs
        codedict["lms"] = posed_lms
        codedict["shape_gt"] = data_input["shape_codes"]
        codedict["exp_gt"] = data_input["exp_codes"]
        codedict["jaw_gt"] = data_input["jaw_poses"]
        codedict["valid_gt"] = data_input["valid_gt"]

        return codedict

    def decode(self, codedict, with_vis=False):
        poselit_code = codedict["poselit"]
        euler_angle, trans, jaw_pose, eyes_pose, exp_para = (
            poselit_code[:, :3],
            poselit_code[:, 3:6],
            codedict["exp"][:, -3:],
            poselit_code[:, 6:12],
            codedict["exp"][:, :-3],
        )
        trans[:, 2] += -0.8  ### init trans: -0.8m

        batch_size = euler_angle.shape[0]
        ### landmark loss donot influence shape and exp

        shape_detach = codedict["shape"].detach()
        exp_detach = exp_para.detach()
        valid_shape = codedict["valid_gt"]
        shape_mixed = codedict["shape"] * (1.0 - valid_shape) + shape_detach * valid_shape
        exp_mixed = exp_para * (1.0 - valid_shape) + exp_detach * valid_shape

        geometry = self.flame_model.forward_geo(shape_mixed, exp_mixed, jaw_pose, eye_pose_params=eyes_pose)
        rott_geo = forward_rott(geometry, euler_angle, trans)
        rott_geo_pose_detach = forward_rott(geometry, euler_angle.detach(), trans.detach())
        vertex_normals = compute_vertex_normals(geometry, self.tri_faces.expand(batch_size, -1, -1))

        (
            pred_contour_lms,
            pred_wflw_lms,
            pred_contour_lms_pose_detach,
            pred_wflw_lms_pose_detach,
        ) = self.flame_model.forward_landmarks(rott_geo, self.cam_para.expand(batch_size, -1), rott_geo_pose_detach)

        output = {}
        output["contour_lms"] = pred_contour_lms
        output["wflw_lms"] = pred_wflw_lms
        output["contour_lms_pose_detach"] = pred_contour_lms_pose_detach
        output["wflw_lms_pose_detach"] = pred_wflw_lms_pose_detach
        if with_vis:
            with torch.no_grad():
                rendered_geo = self.renderer.render_geo(
                    rott_geo,
                    self.tri_faces.expand(batch_size, -1, -1),
                    self.uv_coords.expand(batch_size, -1, -1),
                    self.uv_faces.expand(batch_size, -1, -1),
                    vertex_normals,
                )
            output["rendered_geo"] = rendered_geo

        return output

    def comput_losses(self, codedict, output):
        losses = {}
        losses["lms"] = compute_lms_loss(
            output["contour_lms"],
            output["wflw_lms"],
            codedict["lms"],
            output["contour_lms_pose_detach"],
            output["wflw_lms_pose_detach"],
        )
        losses["iris"] = compute_iris_loss(output["wflw_lms_pose_detach"], codedict["iris"])
        losses["reg_exp"] = compute_reg_loss(codedict["exp"][:, :-3])
        losses["reg_shape"] = compute_reg_loss(codedict["shape"])

        losses["shape"] = compute_para_loss(codedict["shape"], codedict["shape_gt"], codedict["valid_gt"])
        losses["exp"] = compute_para_loss(codedict["exp"][:, :-3], codedict["exp_gt"], codedict["valid_gt"])
        losses["jaw"] = compute_para_loss(codedict["exp"][:, -3:], codedict["jaw_gt"], codedict["valid_gt"])
        return losses
