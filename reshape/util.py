import torch
import torch.nn as nn
import torch.nn.functional as F

# from pytorch3d.ops.knn import knn_points


def compute_tri_normal(geometry, tris):
    tri_1 = tris[:, 0]
    tri_2 = tris[:, 1]
    tri_3 = tris[:, 2]
    vert_1 = torch.index_select(geometry, 1, tri_1)
    vert_2 = torch.index_select(geometry, 1, tri_2)
    vert_3 = torch.index_select(geometry, 1, tri_3)
    nnorm = torch.cross(vert_2 - vert_1, vert_3 - vert_1, 2)
    normal = nn.functional.normalize(nnorm)
    return normal


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


def Rmat2eulurangle(rot):
    batch_size = rot.size(0)
    ang_y = torch.asin(-rot[:, 2, 0]).unsqueeze(1)
    ang_x = torch.atan2(rot[:, 2, 1], rot[:, 2, 2]).unsqueeze(1)
    ang_z = torch.atan2(rot[:, 1, 0], rot[:, 0, 0]).unsqueeze(1)
    return torch.cat((ang_x, ang_y, ang_z), dim=1)


def rot_trans_pts(geometry, rot, trans):
    return torch.bmm(geometry.cpu(), rot.permute(0, 2, 1)) + trans.unsqueeze(1)


def cal_lap_loss(tensor_list, weight_list):
    lap_kernel = torch.Tensor((-0.5, 1.0, -0.5)).unsqueeze(0).unsqueeze(0).float().to(tensor_list[0].device)
    loss_lap = 0
    for i in range(len(tensor_list)):
        in_tensor = tensor_list[i]
        in_tensor = in_tensor.view(-1, 1, in_tensor.shape[-1])
        out_tensor = F.conv1d(in_tensor, lap_kernel)
        loss_lap += torch.mean(out_tensor**2) * weight_list[i]
    return loss_lap


def proj_pts(rott_geo, focal_length, cxy):
    cx, cy = cxy[0], cxy[1]
    X = rott_geo[:, :, 0]
    Y = rott_geo[:, :, 1]
    Z = rott_geo[:, :, 2]
    fxX = focal_length * X
    fyY = focal_length * Y
    proj_x = -fxX / Z + cx
    proj_y = fyY / Z + cy
    return torch.cat((proj_x[:, :, None], proj_y[:, :, None], Z[:, :, None]), 2)


def forward_rott(geometry, euler_angle, trans):
    rot = euler2rot(euler_angle)
    rott_geo = rot_trans_pts(geometry, rot, trans)
    return rott_geo


def forward_transform(geometry, euler_angle, trans, focal_length, cxy):
    rot = euler2rot(euler_angle)
    rott_geo = rot_trans_pts(geometry, rot, trans)
    proj_geo = proj_pts(rott_geo, focal_length, cxy)
    return proj_geo


def subdivide_mid(input_ori):
    input = input_ori.clone()
    b, n, d = input.shape
    mid_pts = (input[:, :-1] + input[:, 1:]) * 0.5
    subdivide_pts = torch.cat((input[:, :-1].unsqueeze(2), mid_pts.unsqueeze(2)), dim=2).reshape(b, -1, d)
    subdivide_pts = torch.cat((subdivide_pts, input[:, -1:]), dim=1)
    return subdivide_pts


def cal_col_loss(pred_img, gt_img, img_mask):
    pred_img = pred_img.float()
    loss = torch.sqrt(torch.sum(torch.square(pred_img - gt_img), 3)) * img_mask / 255
    loss = torch.sum(loss, dim=(1, 2)) / torch.sum(img_mask, dim=(1, 2))
    loss = torch.mean(loss)
    return loss
