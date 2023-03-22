import math
import os
from os.path import join as pjoin

import cv2
import natsort
import numpy as np
import torch
from insightface.app import FaceAnalysis

from ..utils.log_utils import log, info
from ..utils.storage import download_pretrained_models
from ..utils.tensor_utils import normalize
from ..utils.utils import tqdm
from .Config import cfg
from .SLPT import Sparse_alignment_network
from .utils import crop_v2, transform_pixel_v2


def draw_landmark(landmark, image):
    for (x, y) in (landmark + 0.5).astype(np.int32):
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    return image


def crop_img(img, bbox, transform):
    x1, y1, x2, y2 = (bbox[:4] + 0.5).astype(np.int32)

    w = x2 - x1 + 1
    h = y2 - y1 + 1
    cx = x1 + w // 2
    cy = y1 + h // 2
    center = np.array([cx, cy])

    scale = max(math.ceil(x2) - math.floor(x1), math.ceil(y2) - math.floor(y1)) / 200.0

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    input, trans = crop_v2(img, center, scale * 1.15, (256, 256))

    input = transform(input).unsqueeze(0)

    return input, trans


class Detector:
    def __init__(self, slpt_ckpt_path="~/.idrfacetrack/pretrained/detection/WFLW_6_layer.pth"):
        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        self.face_detector, self.slpt_model = self.create_models(slpt_ckpt_path)

    def create_models(self, slpt_ckpt_path):
        ## face detector
        log("loading face detector")
        face_detector = FaceAnalysis(allowed_modules=["detection"], providers=["CUDAExecutionProvider"])
        face_detector.prepare(ctx_id=0, det_size=(224, 224))

        ## SLPT landmark detector
        init_points = "~/.idrfacetrack/pretrained/detection/wflw_init_98.npz"
        init_points = os.path.expanduser(init_points)
        if not os.path.exists(init_points):
            download_pretrained_models()

        slpt_model = Sparse_alignment_network(init_points)
        slpt_model = torch.nn.DataParallel(slpt_model, device_ids=cfg.GPUS).cuda()

        slpt_ckpt_path = os.path.expanduser(slpt_ckpt_path)
        log(f"loading face landmark detector SLPT from '{slpt_ckpt_path}'")

        checkpoint = torch.load(slpt_ckpt_path)
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in slpt_model.module.state_dict().keys()}
        slpt_model.module.load_state_dict(pretrained_dict)
        slpt_model.eval()

        return face_detector, slpt_model

    def get_landmarks(self, img):
        faces = self.face_detector.get(img, max_num=1)

        if (faces is None) or (len(faces) == 0):
            return None

        bbox = faces[0].bbox
        if bbox is None:
            return None

        bbox[0] = int(bbox[0] + 0.5)
        bbox[2] = int(bbox[2] + 0.5)
        bbox[1] = int(bbox[1] + 0.5)
        bbox[3] = int(bbox[3] + 0.5)

        alignment_input, trans = crop_img(img.copy(), bbox, normalize)

        with torch.no_grad():
            raw_output = self.slpt_model(alignment_input.cuda())
        output = raw_output[2][0, -1, :, :].cpu().numpy()
        landmarks = transform_pixel_v2(output * cfg.MODEL.IMG_SIZE, trans, inverse=True)

        return landmarks

    def process_folder(self, data_dir, img_folder="imgs", ldmks_folder="landmarks", max_frames=10000, fps=25, debug=False, img_ext=".jpg"):

        width = max(len(img_folder), len(ldmks_folder))
        info("detect landmarks")
        info(f"{data_dir}")
        info(f"├── {img_folder:{width}} --> reading images from")
        info(f"└── {ldmks_folder:{width}} <-- saving landmarks to")

        imgs_dir = pjoin(data_dir, img_folder)
        ldmks_dir = pjoin(data_dir, ldmks_folder)
        os.makedirs(ldmks_dir, exist_ok=True)

        debug_video = None

        img_names = natsort.natsorted([f for f in os.listdir(imgs_dir) if f.endswith(img_ext)])
        img_names = img_names[:max_frames]
        assert len(img_names) > 0

        for img_name in tqdm(img_names, desc="detect landmarks"):
            img_path = pjoin(imgs_dir, img_name)
            img = cv2.imread(img_path)

            landmarks = self.get_landmarks(img)
            if landmarks is None:
                continue

            ldmks_path = pjoin(ldmks_dir, img_name.replace(img_ext, ".lms"))
            np.savetxt(ldmks_path, landmarks, fmt="%f")

            if debug:
                os.makedirs(pjoin(data_dir, "debug"), exist_ok=True)
                frame = draw_landmark(landmarks, img)
                if debug_video is None:
                    debug_video = cv2.VideoWriter(
                        pjoin(data_dir, "debug", "landmarks.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (frame.shape[1], frame.shape[0]),
                    )

                debug_video.write(frame)

        if debug_video is not None:
            debug_video.release()
