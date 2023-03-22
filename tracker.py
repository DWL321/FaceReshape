import os
import shutil
from os.path import join as pjoin

import cv2
import numpy as np
import torch

from model.landmarks import Detector
from model.matting import BgMatter
from model.recon import Recon
from model.utils.utils import tqdm
from model.utils.log_utils import log


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


def extract_imgs(data_dir, vid_path, img_folder, flip_code=2, max_frames=10000):
    cap = cv2.VideoCapture(vid_path)

    imgs_dir = pjoin(data_dir, img_folder)
    os.makedirs(imgs_dir, exist_ok=True)

    vid_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = min(vid_length, max_frames)
    real_frame_count = 0

    for frame_id in tqdm(range(max_frames), desc="extract imgs"):
        ret, frame = cap.read()

        if not ret:
            break
        real_frame_count += 1

        if flip_code <= 1:
            frame = cv2.flip(frame, flip_code)

        img_path = os.path.join(imgs_dir, f"{frame_id:05d}.jpg")
        cv2.imwrite(img_path, frame)

    cap.release()

    log(f"extracted {real_frame_count} frames")


class Track:
    def __init__(self):
        self.video = None
        self.image = None
        self.result_root = None
        self.img_folder = "imgs"
        self.max_frames = 100000
        self.ldmks_folder = "landmarks"
        self.fps = 30
        self.recon_params_file = "recon_params.pt"
        self.fg_img_folder = "fg_imgs"

    def track_video(self, video, result_root):
        self.video = video
        self.result_root = result_root
        self.image = None

    def track_image(self, image, result_root):
        self.image = image
        self.result_root = result_root
        self.video = None

    def get_recon(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        os.makedirs(self.result_root, exist_ok=True)

        if self.video and not self.image:
            ## extract imgs
            extract_imgs(self.result_root, self.video, self.img_folder, flip_code=2, max_frames=self.max_frames)
        elif not self.video and self.image:
            os.makedirs(pjoin(self.result_root, self.img_folder), exist_ok=True)
            shutil.copy(self.image, pjoin(self.result_root, self.img_folder, "00000.jpg"))

        ## detect wflw landmarks
        ldmks_detector = Detector()
        ldmks_detector.process_folder(
            self.result_root,
            self.img_folder,
            self.ldmks_folder,
            max_frames=self.max_frames,
            fps=self.fps,
            debug=self.video,
        )

        ## face recon
        recon = Recon()
        recon.process_folder(
            self.result_root,
            self.img_folder,
            self.ldmks_folder,
            self.recon_params_file,
            max_frames=self.max_frames,
            fps=self.fps,
            debug=self.video,
        )

        # # background matting
        # matter = BgMatter()
        # matter.process_folder(self.result_root, self.img_folder, self.fg_img_folder, max_frames=self.max_frames)
