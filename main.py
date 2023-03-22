import torch
import argparse
import os
import os.path as osp
import cv2
import numpy as np
from PIL import Image
from reshape import Reshape
from tracker import Track
from track.data.params import ReconParams


def parse_args():
    parser = argparse.ArgumentParser(description="Video Demo")

    parser.add_argument("mode", type=str, default="video", help="video or image")
    parser.add_argument("--src_video", type=str, default="dataset/videos/3.mp4", help="the source face video")
    parser.add_argument("--src_image", type=str, default="dataset/videos/5.jpg", help="the target dace image")
    parser.add_argument("--tar_image", type=str, default="dataset/videos/5.jpg", help="the target face image")
    parser.add_argument("--result_root", default="result", type=str, help="the result file path")
    parser.add_argument("-- tar", type=str, default="thinner", help="thinner or fatter or shorter or longer")

    args = parser.parse_args()
    return args


def get_recon_params(root, recon_params_file):
    recon_params = ReconParams(osp.join(root, recon_params_file))
    recon_params.flame_shape_params.cuda()
    recon_params.flame_expr_params.cuda()
    recon_params.flame_pose_params.cuda()
    recon_params.cam_extris.cuda()
    recon_params.cam_intris.cuda()
    return recon_params


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.result_root, exist_ok=True)

    track_model = Track()
    if args.mode == "video":
        track_model.track_video(args.src_video, args.result_root + "/src")
        track_model.track_image(args.tar_image, args.result_root + "/tar")
    elif args.mode == "image":
        track_model.track_image(args.src_image, args.result_root + "/src")
    else:
        print("wrong mode")
        raise SystemExit()
    track_model.get_recon()

    rehape_model = Reshape("pretrained/reshape/verts.txt")
    src_img = cv2.imread(osp.join(args.result_root, "src", "imgs", "00000.jpg"))
    H = src_img.shape[0]
    W = src_img.shape[1]
    rehape_model.set_grid(H, W)
    src_recon = get_recon_params(osp.join(args.result_root, "src"), "recon_params.pt")
    if args.mode == "video":
        tar_recon = get_recon_params(osp.join(args.result_root, "tar"), "recon_params.pt")
        rehape_model.set_recon_parm(src_recon, tar_recon)
        # 变形视频:初始化参数,输入图片文件夹路径 保存路径 视频名称
        rehape_model.shape_swap(args.result_root + "/src", args.result_root + "/result", args.video)
    else:
        rehape_model.set_recon_parm(src_recon)
        image = np.array(Image.open(args.src_image))
        rehape_model.reshape(image, args.result_root + "/result", "output.jpg", args.tar)
