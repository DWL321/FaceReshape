import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.transform import warp_affine

from ...utils.geo_transform import estimate_transform_exp, estimate_transform_id, estimate_transform_pose
from .backbone import Arcface, iresnet50, resnet50


def freeze_layers(layers):
    for layer in layers:
        for block in layer.parameters():
            block.requires_grad = False


class EmoCnnModule(nn.Module):
    """
    Emotion Recognition module which uses a conv net as its backbone. Currently Resnet-50 is supported.
    """

    def __init__(self, pretrained_path):
        super().__init__()
        self.backbone = resnet50(num_classes=8631, include_top=False)
        if pretrained_path is not None:
            ckpt = torch.load(pretrained_path, map_location="cpu")
            self.load_state_dict(ckpt["state_dict"], strict=False)
        self.crop_size = 224

    def forward(self, imgs, lms=None, in_train=True):
        # imgs: (b, 3, h, w), lms: (b, l, 2)
        if lms is None:
            return self.backbone(imgs)
        trans_mat = estimate_transform_exp(lms, in_train, self.crop_size)
        warped_imgs = warp_affine(imgs, trans_mat, (self.crop_size, self.crop_size))
        # cv2.imwrite('exp_imgs.jpg', (warped_imgs[0]*255).byte().permute(1,2,0).cpu().numpy()[:, :, [2,1,0]])
        return self.backbone(warped_imgs)

    def freezer(self):
        freeze_layers(
            [
                self.backbone.layer1,
                self.backbone.layer2,
                self.backbone.layer3,
                self.backbone.conv1,
                self.backbone.bn1,
            ]
        )


class IdCnnModule(nn.Module):
    """
    Identity Recognition module to extract identity feature.
    """

    def __init__(self, pretrained_path):
        super().__init__()
        self.crop_size = 112
        self.backbone = iresnet50(False, fp16=False)
        if pretrained_path is not None:
            # "/dellnas/users/users_share/zhangdehao/models/models_face_tracking/arcface_backbone.pth"
            self.backbone.load_state_dict(torch.load(pretrained_path, map_location="cpu"))

    def forward(self, imgs, lms, lms_mode="wflw"):
        # imgs: (b, 3, h, w), lms: (b, l, 2)
        trans_mat = estimate_transform_id(lms, lms_mode=lms_mode, crop_size=self.crop_size)
        warped_imgs = warp_affine(imgs, trans_mat, (self.crop_size, self.crop_size))
        # cv2.imwrite('tex_imgs.jpg', (warped_imgs[0]*255).byte().permute(1,2,0).cpu().numpy()[:, :, [2,1,0]])
        return F.normalize(self.backbone(2.0 * warped_imgs - 1.0))

    def freezer(self):
        freeze_layers(
            [
                self.backbone.layer1,
                self.backbone.layer2,
                self.backbone.layer3,
                self.backbone.conv1,
                self.backbone.bn1,
                self.backbone.prelu,
            ]
        )


class ShapeCnnModule(nn.Module):
    """
    Shape module from MICA to extract identity feature.
    """

    def __init__(self, pretrained_path):
        super().__init__()
        self.crop_size = 112
        # self.backbone = iresnet50(False, fp16=False)
        self.backbone = Arcface()
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        assert "arcface" in checkpoint
        self.backbone.load_state_dict(checkpoint["arcface"])

    def forward(self, imgs, lms, lms_mode="wflw"):
        # imgs: (b, 3, h, w), lms: (b, l, 2)
        trans_mat = estimate_transform_id(lms, lms_mode=lms_mode, crop_size=self.crop_size)
        warped_imgs = warp_affine(imgs, trans_mat, (self.crop_size, self.crop_size))
        # cv2.imwrite('shape_imgs.jpg', (warped_imgs[0]*255).byte().permute(1,2,0).cpu().numpy()[:, :, [2,1,0]])
        return F.normalize(self.backbone(2.0 * warped_imgs - 1.0))


class PoseLitCnnModule(nn.Module):
    """
    Deca Resnet module to extract pose feature (jaw, eyeballs, 6d pose) and SH lighting.
    """

    def __init__(self, pretrained_path):
        super().__init__()
        self.encoder = resnet50(include_top=False)
        if pretrained_path is not None:
            ckpt = torch.load(pretrained_path, map_location="cpu")["E_flame"]
            self.load_state_dict(ckpt, strict=False)
        self.crop_size = 224

    def forward(self, imgs, lms=None):
        # imgs: (b, 3, h, w), lms: (b, l, 2)
        if lms is None:
            return self.encoder(imgs)
        trans_mat = estimate_transform_pose(lms, self.crop_size)
        warped_imgs = warp_affine(imgs, trans_mat, (self.crop_size, self.crop_size))
        return self.encoder(warped_imgs)
