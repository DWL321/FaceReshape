import argparse

from .model import Detector


def parse_args():
    parser = argparse.ArgumentParser()

    # face detector
    parser.add_argument("-m", "--trained_model", type=str, default="~/.idrfacetrack/pretrained/detection/yunet_final.pth")

    parser.add_argument("--confidence_threshold", default=0.7, type=float, help="confidence_threshold")
    parser.add_argument("--top_k", default=5000, type=int, help="top_k")
    parser.add_argument("--nms_threshold", default=0.3, type=float, help="nms_threshold")
    parser.add_argument("--keep_top_k", default=750, type=int, help="keep_top_k")
    parser.add_argument("--vis_thres", default=0.3, type=float, help="visualization_threshold")
    parser.add_argument("--base_layers", default=16, type=int, help="the number of the output of the first layer")
    parser.add_argument("--device", default="cuda")

    # landmark detector
    parser.add_argument("--ckpt", type=str, default="~/.idrfacetrack/pretrained/detection/WFLW_6_layer.pth")

    parser.add_argument("--dataDir", help="data directory", type=str, default="./")
    parser.add_argument("--prevModelDir", help="prev Model directory", type=str, default=None)
    parser.add_argument("--img_step", default=1, type=int, help="the image file to be detected")

    parser.add_argument("--data_dir", type=str, help="the data directory")
    parser.add_argument("--img_folder", default="imgs", type=str, help="the image file to be detected")
    parser.add_argument("--ldmks_folder", default="landmarks", type=str, help="save landmarks to this folder")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--no_video", action="store_true")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    detector = Detector(args.ckpt)
    detector.process_folder(args.data_dir, args.img_folder, args.ldmks_folder, fps=args.fps, debug=not args.no_video)
