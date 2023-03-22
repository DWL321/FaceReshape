import argparse

from track.recon.model import Recon


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="the data directory")
    parser.add_argument("--img_folder", default="imgs", type=str, help="the image file to be detected")
    parser.add_argument("--ldmks_folder", default="landmarks", type=str, help="landmarks folder")
    parser.add_argument(
        "--recon_params_file", default="recon_params.pt", type=str, help="save recon results to this file"
    )
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--no_video", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    recon = Recon()
    recon.process_folder(
        args.data_dir, args.img_folder, args.ldmks_folder, args.recon_params_file, fps=args.fps, debug=not args.no_video
    )
