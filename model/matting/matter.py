import os
from os.path import join as pjoin

import cv2
import natsort
import torch

from ..utils.log_utils import info
from ..utils.storage import download_pretrained_models
from ..utils.utils import tqdm
from .model import MattingNetwork


class BgMatter:
    def __init__(self, ckpt_path="~/.idrfacetrack/pretrained/matting/rvm_resnet50.pth"):
        self.model = self.create_models(ckpt_path)

        self.green = torch.tensor([0.0, 1.0, 0.0]).view(3, 1, 1).cuda()

    def create_models(self, ckpt_path):
        ckpt_path = os.path.expanduser(ckpt_path)
        if not os.path.exists(ckpt_path):
            download_pretrained_models()

        model = MattingNetwork("resnet50").eval().cuda()
        model.load_state_dict(torch.load(ckpt_path))
        return model

    def process_folder(self, data_dir, img_folder="imgs", fg_img_folder="fg_imgs", max_frames=10000, fps=25, img_ext=".jpg"):

        width = max(len(img_folder), len(fg_img_folder))
        info("background matting")
        info(f"{data_dir}")
        info(f"├── {img_folder:{width}} --> reading images from")
        info(f"└── {fg_img_folder:{width}} <-- saving foreground images to")

        imgs_dir = pjoin(data_dir, img_folder)
        fg_img_dir = pjoin(data_dir, fg_img_folder)
        os.makedirs(fg_img_dir, exist_ok=True)

        # out_vid = None
        # save_vid_path = os.path.join(args.debug_path, "fg.avi")

        with torch.no_grad():
            first_img = True
            rec = [None] * 4  # Initial recurrent states.

            img_names = natsort.natsorted([f for f in os.listdir(imgs_dir) if f.endswith(img_ext)])
            img_names = img_names[:max_frames]
            assert len(img_names) > 0

            for img_name in tqdm(img_names, desc="background matting"):
                img_path = pjoin(imgs_dir, img_name)
                img = cv2.imread(img_path)

                fg_img, pha, rec = self.extract_fg(img, first_img, rec)
                if first_img:
                    first_img = False

                fg_img_path = pjoin(fg_img_dir, img_name)
                cv2.imwrite(fg_img_path, fg_img)
                # if out_vid is None:
                #     out_vid = cv2.VideoWriter(
                #         save_vid_path, cv2.VideoWriter_fourcc("M", "J", "P", "G"), 30, (img.shape[1], img.shape[0])
                #     )
                # out_vid.write(img)

                alpha = (pha[0] * 255).detach().byte().permute(1, 2, 0).cpu().numpy()
                fg_img_alpha_path = pjoin(fg_img_dir, img_name.replace(img_ext, ".alpha.png"))
                cv2.imwrite(fg_img_alpha_path, alpha)

    def extract_fg(self, img, first_img=True, rec=[None] * 4, downsample_ratio=0.25):
        src = torch.as_tensor(img).cuda()[:, :, [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).float() / 255.0
        if first_img:
            for _ in range(10):
                fgr, pha, *rec = self.model(src.cuda(), *rec, downsample_ratio)
        else:
            for _ in range(2):
                fgr, pha, *rec = self.model(src.cuda(), *rec, downsample_ratio)

        fg_img = fgr * pha + self.green * (1 - pha)
        fg_img = (fg_img[0] * 255).detach().byte().permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
        return fg_img, pha, rec
