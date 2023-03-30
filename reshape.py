import os
import os.path as osp
import matplotlib
import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from model.flame.flame import FLAME
from model.utils.renderer import NvDiffRenderer
from model.reshape.geo_transform import forward_rott
from model.reshape.mls import mls_deformation
from model.reshape.util import forward_transform
from model.reshape.energy import opt
import torch.nn.functional as F


def get_flame_model(verts_file):
    flame_model = FLAME()

    # 仅保留面部
    valid_index = []
    with open(verts_file) as file_read:
        # 将文件存在lines中
        lines = file_read.readlines()
    # 循环读取文件
    for line in lines:
        valid_index = valid_index + [int(line)]
    vs_isvalid = np.zeros((6000), dtype=np.bool)
    for i in range(len(valid_index)):
        vs_isvalid[valid_index[i]] = True
    temp_tris = torch.tensor([])
    for face in flame_model.faces_tensor:
        if vs_isvalid[face[0]] and vs_isvalid[face[1]] and vs_isvalid[face[2]]:
            temp_tris = torch.cat((temp_tris, face.unsqueeze(0)), 0)
    flame_model.faces_tensor = temp_tris.clone()

    return flame_model


def findAllContours(mask, sample_radius=7.0):
    # cs,hs=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_TC89_L1)
    cs, hs = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    conts = []
    for cont in cs:
        cont = cont.reshape(-1, 2)
        lens = cont[1:] - cont[:-1]
        lens = np.concatenate((lens, (cont[0] - cont[-1]).reshape(1, 2)), axis=0).astype(np.float32)
        lens = np.linalg.norm(lens, axis=-1)
        sels = []
        dis = 0
        for ind, tmp in enumerate(lens):
            if dis + tmp < sample_radius:
                sels.append(False)
                dis = dis + tmp
            else:
                sels.append(True)
                dis = 0
        sels = np.array(sels[-1:] + sels[:-1])
        cont = cont[sels]
        if len(cont) > 3:
            conts.append(cont)
    conts.sort(key=lambda tupl: tupl.shape[0], reverse=True)
    return [conts[0]]


def findAllContours_tar(mask):
    # cs,hs=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_TC89_L1)
    cs, hs = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    conts = []
    for cont in cs:
        cont = cont.reshape(-1, 2)
        conts.append(cont)
    conts.sort(key=lambda tupl: tupl.shape[0], reverse=True)
    return [conts[0]]


# 固定边界点
def add_edge_points(H, W, conts, tar_contps):
    H_edge_points_ = np.arange(0, H, 50, dtype=np.int32)
    H_edge_points_l = []
    H_edge_points_r = []
    for j in H_edge_points_:
        H_edge_points_l = H_edge_points_l + [[j, 0]]
        H_edge_points_r = H_edge_points_r + [[j, W - 1]]
    W_edge_points_ = np.arange(1, W, 50, dtype=np.int32)
    W_edge_points_t = []
    W_edge_points_b = []
    for j in W_edge_points_:
        W_edge_points_t = W_edge_points_t + [[0, j]]
        W_edge_points_b = W_edge_points_b + [[H - 1, j]]
    conts = np.concatenate((conts, [H_edge_points_l], [H_edge_points_r], [W_edge_points_b], [W_edge_points_t]), axis=1)
    tar_contps = np.concatenate(
        (tar_contps, H_edge_points_l, H_edge_points_r, W_edge_points_b, W_edge_points_t), axis=0
    )
    return conts, tar_contps


def bilinear(transform, vx, vy, image):
    # 双线性插值
    transform_ = (
        (
            F.interpolate(transform.unsqueeze(0), scale_factor=1, mode="bilinear", align_corners=True)
            .reshape(2, transform.shape[1], transform.shape[2])
            .type(torch.int)
        )
        .cpu()
        .numpy()
    )
    aug = np.ones_like(image)
    aug[vx, vy] = image[tuple(transform_)]

    return aug


# 拼接结果视频
def get_mls_video(W, H, result_root):
    fps = 25  # 视频每秒25帧
    size = (W, H)  # 需要转为视频的图片的尺寸
    video = cv2.VideoWriter(result_root + "/result_mls.avi", cv2.VideoWriter_fourcc("I", "4", "2", "0"), fps, size)
    ind = 0
    while True:
        name = osp.join(result_root, "result_mls", "%05d.jpg" % ind)
        if osp.isfile(name):
            img = cv2.imread(name)
            video.write(img)
            ind = ind + 1
        else:
            break

    video.release()


# 拼接对比视频
def get_compare_video(src_root, result_root, input_video):
    reader1 = cv2.VideoCapture(input_video)
    reader2 = cv2.VideoCapture(osp.join(result_root, "result_mls.avi"))
    width = int(reader1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(reader1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        osp.join(result_root, "compare_src_mls.avi"),
        cv2.VideoWriter_fourcc("I", "4", "2", "0"),
        25,
        (width, height // 2),
    )

    print(reader1.isOpened())
    print(reader2.isOpened())
    have_more_frame1 = True
    have_more_frame2 = True
    while have_more_frame1 and have_more_frame2:
        have_more_frame1, frame1 = reader1.read()
        have_more_frame2, frame2 = reader2.read()
        if not (have_more_frame1 and have_more_frame2):
            break
        frame1 = cv2.resize(frame1, (width // 2, height // 2))
        frame2 = cv2.resize(frame2, (width // 2, height // 2))
        img = np.hstack((frame1, frame2))
        writer.write(img)

    writer.release()
    reader1.release()
    reader2.release()


# 调整flame_shape_code参数编辑脸型
def change_shape_code(shape, mode):
    if mode == "fatter":
        shape[0] = shape[0] * 0.001
        shape[1] = shape[1] * 0.0001
        shape[2] = shape[2] * 4
        shape[3] = shape[3] * 0.05
    if mode == "longer":
        shape[0] = shape[0] * 1
        shape[1] = shape[1] * 0.1
        shape[2] = shape[2] * 8
        shape[3] = shape[3] * 1
    if mode == "thinner":
        shape[0] = shape[0] * 2
        shape[1] = shape[1] * 4
        shape[2] = shape[2] * 4
        shape[3] = shape[3] * 2
    if mode == "shorter":
        shape[0] = shape[0] * 6
        shape[1] = shape[1] * 0.5
        shape[2] = shape[2] * 4
        shape[3] = shape[3] * 0.5
    return shape


class Reshape:
    def __init__(self, verts_file):
        self.flame_model = get_flame_model(verts_file)
        self.grid = False
        self.recon_s = False
        self.recon_t = False

    def set_grid(self, H, W):
        self.H = H
        self.W = W
        self.render = NvDiffRenderer((H, W))
        # Define deformation grid
        gridX = np.arange(self.W, dtype=np.int32)
        gridY = np.arange(self.H, dtype=np.int32)
        self.vy, self.vx = np.meshgrid(gridX, gridY)
        self.grid = True

    def set_recon_parm(self, src_recon, tar_recon=None):
        self.src_recon = src_recon
        self.src_cxy = torch.tensor(src_recon.cam_intris[:, 2:]).cuda()
        if self.src_recon.flame_shape_params.shape[0] > 1:
            self.src_shape = self.src_recon.flame_shape_params.mean(0)
        self.recon_s = True
        if tar_recon:
            self.tar_recon = tar_recon
            self.tar_shape = self.tar_recon.flame_shape_params[0]
            self.recon_t = True

    def check_init(self):
        if self.grid and self.recon_s:
            return True
        else:
            if not self.grid:
                print("please set gird")
            if not self.recon_s:
                print("please set source recon parameters")
            return False

    # 获得变形后稀疏轮廓点坐标
    def get_tar_contps(self, conts_ws, conts_fs, tar_vs, sid, tar_contps_mask):
        tar_contps_init = []
        for cont_ws, cont_fs in zip(conts_ws, conts_fs):
            cont_fs = cont_fs.cpu().detach().numpy()
            cont_ws = cont_ws.unsqueeze(-1)
            ps = (tar_vs[0, self.flame_model.faces_tensor[cont_fs].detach().numpy()] * cont_ws.cpu()).sum(-2)
            tar_contps_init.append(ps)
        tar_contps_init = torch.cat(tar_contps_init, dim=0)
        tar_contps_init = (
            forward_transform(
                tar_contps_init[None],
                self.src_recon.cam_extris[sid : sid + 1, 0:3],
                self.src_recon.cam_extris[sid : sid + 1, 3:6],
                self.src_recon.cam_intris[sid, 0].cpu(),
                self.src_cxy[sid].cpu(),
            )[0]
            .cpu()
            .numpy()
        )
        tar_contps_init = tar_contps_init.transpose((1, 0))
        tar_contps_init = np.array([tar_contps_init[0]] + [tar_contps_init[1]])
        tar_contps_init = tar_contps_init.transpose((1, 0)).astype(np.int32)

        tar_contps = []
        for point in tar_contps_init:
            dis = (tar_contps_mask[:, 1] - point[1]) ** 2 + (tar_contps_mask[:, 0] - point[0]) ** 2
            dis = dis.tolist()
            min_index = dis.index(min(dis))
            tar_contps = tar_contps + [[tar_contps_mask[min_index][0], tar_contps_mask[min_index][1]]]
            tar_contps_mask[min_index] = [-100, -100]
        tar_contps = np.array(tar_contps)
        return tar_contps

    def reshape_one_pic(self, image, sid, result_root, result_name, gray):
        if not self.recon_t:
            print("please set target recon parameters")
        if not self.check_init() or not self.recon_t:
            print("reshape failed")
            return None

        # 得到源人脸特征点的三维坐标(id+表情+姿态)
        self.src_recon.flame_shape_params[sid] = self.src_shape
        vs = self.flame_model.forward_geo(*(self.src_recon.get_flame_params([sid])))

        # 得到源人脸的mask，坐标值归一化
        verts = forward_rott(vs, *(self.src_recon.get_cam_extris([sid]))).cuda()
        render_mask = self.render.get_mask(
            verts, self.flame_model.faces_tensor, self.src_recon.cam_intris[sid].unsqueeze(0)
        )
        rast_out = self.render.rasterize(
            verts, self.flame_model.faces_tensor, self.src_recon.cam_intris[sid].unsqueeze(0)
        )

        # 得到原轮廓坐标
        conts = findAllContours(render_mask[0].cpu().numpy().astype(np.uint8))

        # 得到原轮廓索引
        conts_ws = []
        conts_fs = []
        for cont in conts:
            cont = torch.from_numpy(cont.astype(np.int64))
            cont_ws = rast_out[0, cont[:, 1], cont[:, 0], :2]
            cont_ws = torch.cat((cont_ws, (1.0 - cont_ws.sum(-1)).reshape(-1, 1)), dim=-1)
            cont_fs = rast_out[0, cont[:, 1], cont[:, 0], -1].long() - 1
            conts_ws.append(cont_ws)
            conts_fs.append(cont_fs)
        # 得到目标人脸特征点的三维坐标
        self.src_recon.flame_shape_params[sid] = self.tar_shape
        tar_vs = self.flame_model.forward_geo(*(self.src_recon.get_flame_params([sid])))

        # tar_mask start
        verts = forward_rott(tar_vs, *(self.src_recon.get_cam_extris([sid]))).cuda()
        render_mask = self.render.get_mask(
            verts, self.flame_model.faces_tensor, self.src_recon.cam_intris[sid].unsqueeze(0)
        )
        rast_out = self.render.rasterize(
            verts, self.flame_model.faces_tensor, self.src_recon.cam_intris[sid].unsqueeze(0)
        )
        # 得到变形后轮廓坐标
        tar_contps_mask = findAllContours_tar(render_mask[0].cpu().numpy().astype(np.uint8))
        tar_contps_mask = np.array(tar_contps_mask).reshape(-1, 2)

        # tar_mask end

        # 获得变形后稀疏轮廓点坐标
        tar_contps = self.get_tar_contps(conts_ws, conts_fs, tar_vs, sid, tar_contps_mask)

        # 固定边界点
        conts, tar_contps = add_edge_points(self.W, self.H, conts, tar_contps)

        # 二维图像变形
        transform = mls_deformation(conts, tar_contps, self.vy, self.vx)
        transform = opt(self.H, self.W, gray, transform, [self.vx, self.vy])
        aug = bilinear(transform, self.vx, self.vy, image)
        matplotlib.image.imsave(osp.join(result_root, result_name), aug)
        return transform

    def shape_swap(self, src_root, result_root, input_video):
        img_folder = "imgs"
        os.makedirs(result_root, exist_ok=True)
        os.makedirs(osp.join(result_root, "result_mls"), exist_ok=True)
        src_pic_num = 0
        for file in os.listdir(osp.join(src_root, img_folder)):
            if file.endswith(".jpg"):
                src_pic_num += 1

        if src_pic_num % 2 == 0:
            sid = src_pic_num - 1
            image = np.array(Image.open(osp.join(src_root, img_folder, "%05d.jpg" % sid)))
            gray = cv2.imread(osp.join(result_root, "../mask", "%05d.jpg" % sid), 0)
            self.reshape_one_pic(image, sid, result_root + "/result_mls", "%05d.jpg" % sid, gray[None])

        for sid in tqdm(range(0, src_pic_num, 2)):
            image = np.array(Image.open(osp.join(src_root, img_folder, "%05d.jpg" % sid)))
            gray = cv2.imread(osp.join(result_root, "../mask", "%05d.jpg" % sid), 0)
            transform = self.reshape_one_pic(image, sid, result_root + "/result_mls", "%05d.jpg" % sid, gray[None])
            if sid:
                image = np.array(Image.open(osp.join(src_root, img_folder, "%05d.jpg" % (sid - 1))))
                aug = bilinear((transform + last_transform) / 2, self.vx, self.vy, image)
                matplotlib.image.imsave(osp.join(result_root, "result_mls", "%05d.jpg" % (sid - 1)), aug)
            last_transform = transform

        # 图片转视频对比效果
        # 拼接mls结果视频
        get_mls_video(self.W, self.H, result_root)

        # 拼接对比视频
        get_compare_video(src_root, result_root, input_video)

    def reshape(self, image, gray, result_root, result_name, mode):
        if not self.check_init():
            print("reshape failed")
            return None

        os.makedirs(result_root, exist_ok=True)

        # 得到源人脸特征点的三维坐标(id+表情+姿态)
        vs = self.flame_model.forward_geo(*(self.src_recon.get_flame_params([0])))

        # 得到源人脸的mask，坐标值归一化
        verts = forward_rott(vs, *(self.src_recon.get_cam_extris([0]))).cuda()
        render_mask = self.render.get_mask(
            verts, self.flame_model.faces_tensor, self.src_recon.cam_intris[0].unsqueeze(0)
        )
        rast_out = self.render.rasterize(
            verts, self.flame_model.faces_tensor, self.src_recon.cam_intris[0].unsqueeze(0)
        )

        # 得到原轮廓坐标
        conts = findAllContours(render_mask[0].cpu().numpy().astype(np.uint8))

        # 得到原轮廓索引
        conts_ws = []
        conts_fs = []
        for cont in conts:
            cont = torch.from_numpy(cont.astype(np.int64))
            cont_ws = rast_out[0, cont[:, 1], cont[:, 0], :2]
            cont_ws = torch.cat((cont_ws, (1.0 - cont_ws.sum(-1)).reshape(-1, 1)), dim=-1)
            cont_fs = rast_out[0, cont[:, 1], cont[:, 0], -1].long() - 1
            conts_ws.append(cont_ws)
            conts_fs.append(cont_fs)
        # 得到变形后人脸特征点的三维坐标
        self.src_recon.flame_shape_params[0] = change_shape_code(self.src_recon.flame_shape_params[0], mode)
        tar_vs = self.flame_model.forward_geo(*(self.src_recon.get_flame_params([0])))

        # tar_mask start
        verts = forward_rott(tar_vs, *(self.src_recon.get_cam_extris([0]))).cuda()
        render_mask = self.render.get_mask(
            verts, self.flame_model.faces_tensor, self.src_recon.cam_intris[0].unsqueeze(0)
        )
        rast_out = self.render.rasterize(
            verts, self.flame_model.faces_tensor, self.src_recon.cam_intris[0].unsqueeze(0)
        )
        # 得到变形后轮廓坐标
        tar_contps_mask = findAllContours_tar(render_mask[0].cpu().numpy().astype(np.uint8))
        tar_contps_mask = np.array(tar_contps_mask).reshape(-1, 2)

        # tar_mask end

        # 获得变形后稀疏轮廓点坐标
        tar_contps = self.get_tar_contps(conts_ws, conts_fs, tar_vs, 0, tar_contps_mask)

        # 固定边界点
        conts, tar_contps = add_edge_points(self.W, self.H, conts, tar_contps)

        # 二维图像变形
        transform = mls_deformation(conts, tar_contps, self.vy, self.vx)
        transform = opt(self.H, self.W, gray, transform, [self.vx, self.vy])
        aug = bilinear(transform, self.vx, self.vy, image)
        matplotlib.image.imsave(osp.join(result_root, result_name), aug)
