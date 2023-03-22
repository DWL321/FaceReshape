import torch
import torch.nn as nn

from .backbone import Generator, MappingNetwork


class ShapeMapper(nn.Module):
    def __init__(self, pretrained_path):
        super().__init__()
        self.mapper = Generator()
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        assert "flameModel" in checkpoint
        self.mapper.load_state_dict(checkpoint["flameModel"], strict=False)

    def forward(self, id_feature):
        return self.mapper(id_feature)


class ExpMapper(nn.Module):
    def __init__(self, dim_exp):
        super().__init__()
        self.mapper = MappingNetwork(2048, 128, dim_exp)

    def forward(self, emo_feature):
        return self.mapper(emo_feature)


class PoseLitMapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.mapper = MappingNetwork(2048, 128, 12)  ## 6(dof) + 2*3(eyes)

    def forward(self, poselit_feature):
        return self.mapper(poselit_feature)


class TexMapper(nn.Module):
    def __init__(self, dim_tex):
        super().__init__()
        self.mapper = MappingNetwork(512, 128, dim_tex)

    def forward(self, tex_feature):
        return self.mapper(tex_feature)
