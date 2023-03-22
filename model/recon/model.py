import os
from os.path import join as pjoin

import cv2
import natsort
import numpy as np
import torch
from torchvision.utils import draw_keypoints

from ..utils.log_utils import log, info
from ..utils.storage import download_pretrained_models
from ..utils.utils import tqdm
from .config.default import pipeline_config, trainer_config
from .pipeline import FaceRecon


def to_byte_img(x):
    return torch.clip((x.detach() * 255.0), 0, 255.0).byte()


class Recon(object):
    def __init__(self, ckpt_path="~/.idrfacetrack/pretrained/recon/epoch_22.tar"):
        torch.set_grad_enabled(False)
        torch.backends.cudnn.enabled = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        trainer_config.gpu_ids = pipeline_config.gpu_ids = [0]
        self.device = "cuda"

        self.model = self.create_model(ckpt_path)

    def create_model(self, ckpt_path):
        ckpt_path = os.path.expanduser(ckpt_path)
        if not os.path.exists(ckpt_path):
            download_pretrained_models()
        model = FaceRecon(pipeline_config)
        log(f"loading model from '{ckpt_path}'")
        model.load_model(ckpt_path)
        model.to(self.device)
        model.eval()
        return model

    def process_folder(
        self,
        data_dir,
        img_folder="imgs",
        ldmks_folder="landmarks",
        recon_params_file="recon_params.pt",
        max_frames=10000,
        fps=25,
        debug=False,
        img_ext=".jpg",
    ):
        width = max(len(img_folder), len(ldmks_folder), len(recon_params_file))
        info("face recon")
        info(f"{data_dir}")
        info(f"├── {img_folder:{width}} --> reading images")
        info(f"├── {ldmks_folder:{width}} --> reading landmarks")
        info(f"└── {recon_params_file:{width}} <-- saving recon results to")

        imgs_dir = pjoin(data_dir, img_folder)
        ldmks_dir = pjoin(data_dir, ldmks_folder)

        debug_video = None

        results = []

        img_names = natsort.natsorted([f for f in os.listdir(imgs_dir) if f.endswith(img_ext)])
        img_names = img_names[:max_frames]
        assert len(img_names) > 0

        for img_name in tqdm(img_names, desc="face recon"):
            img_path = pjoin(imgs_dir, img_name)
            img = cv2.imread(img_path)

            ldmks_path = pjoin(ldmks_dir, img_name.replace(img_ext, ".lms"))
            ldmks = np.loadtxt(ldmks_path)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            recon_out, listed_imgs = self.face_recon(img_rgb, ldmks)
            results.append(recon_out)

            if debug:
                if debug_video is None:
                    os.makedirs(pjoin(data_dir, "debug"), exist_ok=True)
                    debug_video = cv2.VideoWriter(
                        pjoin(data_dir, "debug", "facerecon.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (listed_imgs.shape[1], listed_imgs.shape[0]),
                    )

                debug_video.write(listed_imgs)

        if debug_video is not None:
            debug_video.release()

        shape_code, exp_para, flame_pose_params, cam_poses, cam_para = list(zip(*results))
        recon_params = {
            "flame_shape_params": torch.vstack(shape_code),
            "flame_expr_params": torch.vstack(exp_para),
            "flame_pose_params": torch.vstack(flame_pose_params),
            "cam_extris": torch.vstack(cam_poses),
            "cam_intris": torch.vstack(cam_para),
        }
        torch.save(recon_params, pjoin(data_dir, recon_params_file))

    def face_recon(self, img, ldmks):
        imgs = torch.as_tensor(img).unsqueeze(0).to(self.device).permute(0, 3, 1, 2).contiguous().float() / 255.0
        ldmks = torch.as_tensor(ldmks).unsqueeze(0).to(self.device).float()

        with torch.no_grad():
            recon_out, vis_dict = self.model.forward_test(imgs, ldmks)

        imgs, lms, rendered_geo = (
            imgs[0],
            vis_dict["pred_lms"],
            vis_dict["rendered_geo"][0],
        )
        lms_imgs = to_byte_img(imgs.clone())
        lms_imgs = draw_keypoints(lms_imgs, lms, colors="blue").to(self.device)
        lms_imgs = lms_imgs.float() / 255.0
        listed_imgs = torch.cat((lms_imgs.unsqueeze(2), rendered_geo.unsqueeze(2)), dim=2).reshape(
            3, lms_imgs.shape[1], -1
        )
        listed_imgs = (torch.clamp(listed_imgs, 0.0, 1.0) * 255).byte().permute(1, 2, 0)[:, :, [2, 1, 0]].cpu().numpy()

        return recon_out, listed_imgs
