import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn as nn

from ...utils.renderer import calc_ndc_proj_mat


class SH:
    def __init__(self):
        self.a = [np.pi, 2 * np.pi / np.sqrt(3.0), 2 * np.pi / np.sqrt(8.0)]
        self.c = [1 / np.sqrt(4 * np.pi), np.sqrt(3.0) / np.sqrt(4 * np.pi), 3 * np.sqrt(5.0) / np.sqrt(12 * np.pi)]


class MeshRenderer(nn.Module):
    ## TODO extract the nvdiffrast part and use ...utils.renderer.NvDiffRenderer
    def __init__(self, cam_para, img_size=[224, 224], gpu_ids=[]):
        super(MeshRenderer, self).__init__()
        self.img_size = img_size
        self.h, self.w = img_size
        ndc_proj = torch.as_tensor(calc_ndc_proj_mat(cam_para[None], img_size)[0])
        self.register_buffer("ndc_proj", ndc_proj)
        self.glctxs = {}
        for gpu_id in gpu_ids:
            device = torch.device("cuda:" + str(gpu_id))
            self.glctxs[device] = dr.RasterizeGLContext(device=device, output_db=True)
        init_lit = torch.zeros((1, 9, 1)).float()
        init_lit[0, 0, 0] = 0.5
        init_lit[0, 2, 0] = 0.3
        self.register_buffer("init_lit", init_lit)
        self.SH = SH()

    def render_geo(self, vertex, tri, uvs, faces_uvs, vertex_normals):
        """
        vertex: (b, nv, 3), tri: (b, nf, 3), tex: (b, u, v, 3), lights: (b, 27),
        uvs: (b, nv, 2), faces_uvs: (b, nf, 3), vertex_normals: (b, nv, 3)
        Return:
        color_img: (b, h, w, 3), mask: (b, 1, h, w), rast_out: (b, h, w, 4)
        """
        tri = tri.int()
        faces_uvs = faces_uvs.int()
        glctx = self.glctxs[vertex.device]
        # trans to homogeneous coordinates of 3d vertices, the direction of y is the same as v
        if vertex.shape[-1] == 3:
            vertex = torch.cat((vertex, torch.ones_like(vertex[..., :1])), dim=-1)
        vertex_ndc = vertex @ self.ndc_proj.t()
        rast_out, _ = dr.rasterize(glctx, vertex_ndc.contiguous(), tri[0], resolution=self.img_size)
        uv_out = dr.interpolate(uvs[0], rast_out, faces_uvs[0])[0]
        uv_normals = dr.interpolate(vertex_normals, rast_out, tri[0])[0]
        uv_normals = torch.nn.functional.normalize(uv_normals, dim=-1)
        geo_img = self.compute_light_color(
            torch.ones_like(uv_normals) * 0.6,
            uv_normals.detach(),
            uv_normals.new_zeros(uv_normals.shape[0], 27),
            self.img_size,
        ).permute(0, 3, 1, 2)
        return geo_img

    def forward(self, vertex, tri, tex, lights, uvs, faces_uvs, vertex_normals):
        """
        vertex: (b, nv, 3), tri: (b, nf, 3), tex: (b, u, v, 3), lights: (b, 27),
        uvs: (b, nv, 2), faces_uvs: (b, nf, 3), vertex_normals: (b, nv, 3)
        Return:
        color_img: (b, h, w, 3), mask: (b, 1, h, w), rast_out: (b, h, w, 4)
        """
        tri = tri.int()
        faces_uvs = faces_uvs.int()
        glctx = self.glctxs[vertex.device]
        # trans to homogeneous coordinates of 3d vertices, the direction of y is the same as v
        if vertex.shape[-1] == 3:
            vertex = torch.cat((vertex, torch.ones_like(vertex[..., :1])), dim=-1)
        vertex_ndc = vertex @ self.ndc_proj.t()
        rast_out, _ = dr.rasterize(glctx, vertex_ndc.contiguous(), tri[0], resolution=self.img_size)
        mask = (rast_out[..., 3] > 0).float().unsqueeze(1)
        uv_out = dr.interpolate(uvs[0], rast_out, faces_uvs[0])[0]
        uv_normals = dr.interpolate(vertex_normals, rast_out, tri[0])[0]
        uv_normals = torch.nn.functional.normalize(uv_normals, dim=-1)
        tex_img = dr.texture(tex, uv_out)
        color_img = self.compute_light_color(tex_img, uv_normals, lights, self.img_size).permute(0, 3, 1, 2)
        geo_img = self.compute_light_color(
            torch.ones_like(tex_img) * 0.6, uv_normals.detach(), torch.zeros_like(lights), self.img_size
        ).permute(0, 3, 1, 2)
        return color_img, mask, rast_out, geo_img

    def compute_light_color(self, texture, normal, gamma, img_size):
        h, w = img_size
        gamma = gamma.reshape(-1, 9, 3) + self.init_lit
        normal = normal.reshape(-1, h * w, 3)
        texture = texture.reshape(-1, h * w, 3)
        a, c = self.SH.a, self.SH.c
        Y = torch.cat(
            [
                a[0] * c[0] * torch.ones_like(normal[..., :1]),
                -a[1] * c[1] * normal[..., 1:2],
                a[1] * c[1] * normal[..., 2:],
                -a[1] * c[1] * normal[..., :1],
                a[2] * c[2] * normal[..., :1] * normal[..., 1:2],
                -a[2] * c[2] * normal[..., 1:2] * normal[..., 2:],
                0.5 * a[2] * c[2] / np.sqrt(3.0) * (3 * normal[..., 2:] ** 2 - 1),
                -a[2] * c[2] * normal[..., :1] * normal[..., 2:],
                0.5 * a[2] * c[2] * (normal[..., :1] ** 2 - normal[..., 1:2] ** 2),
            ],
            dim=-1,
        )
        color = torch.bmm(Y, gamma) * texture
        return color.reshape(-1, h, w, 3)

    def forward_test(self, vertex, tri, uvs, faces_uvs, vertex_normals, cam_para, img_size):
        """
        vertex: (b, nv, 3), tri: (b, nf, 3), tex: (b, u, v, 3), lights: (b, 27),
        uvs: (b, nv, 2), faces_uvs: (b, nf, 3), vertex_normals: (b, nv, 3)
        Return:
        color_img: (b, h, w, 3), mask: (b, 1, h, w), rast_out: (b, h, w, 4)
        """
        ndc_proj = torch.as_tensor(calc_ndc_proj_mat(cam_para[None], img_size)[0]).to(vertex.device)
        tri = tri.int()
        faces_uvs = faces_uvs.int()
        glctx = self.glctxs[vertex.device]
        # trans to homogeneous coordinates of 3d vertices, the direction of y is the same as v
        if vertex.shape[-1] == 3:
            vertex = torch.cat((vertex, torch.ones_like(vertex[..., :1])), dim=-1)
        vertex_ndc = vertex @ ndc_proj.t()
        rast_out, _ = dr.rasterize(glctx, vertex_ndc.contiguous(), tri[0], resolution=img_size)
        uv_out = dr.interpolate(uvs[0], rast_out, faces_uvs[0])[0]
        uv_normals = dr.interpolate(vertex_normals, rast_out, tri[0])[0]
        uv_normals = torch.nn.functional.normalize(uv_normals, dim=-1)
        geo_img = self.compute_light_color(
            torch.ones_like(uv_normals) * 0.6,
            uv_normals.detach(),
            uv_normals.new_zeros(uv_normals.shape[0], 27),
            img_size,
        ).permute(0, 3, 1, 2)
        return geo_img
