import numpy as np
import torch

import nvdiffrast.torch as dr


def calc_ndc_proj_mat(cam_intris, img_size, n=0.1, f=3.0):
    """
    cam_intris: shape (batch, 4)
    n -- near, f -- far
    """
    if len(cam_intris.shape) < 2:
        raise NotImplementedError("cam_intris should be of shape (batch_size, 4)")

    h, w = img_size
    # fmt: off
    ndc_proj_mat = [
        [
            [2 * fx / (w - 1),                0 , 1 - 2 * cx / (w - 1),                     0 ],
            [              0 , -2 * fy / (h - 1), 1 - 2 * cy / (h - 1),                     0 ],
            [              0 ,                0 ,   -(f + n) / (f - n), -(2 * f * n) / (f - n)],
            [              0 ,                0 ,                  -1 ,                     0 ],
        ]
        for fx, fy, cx, cy in cam_intris.tolist()
    ]
    # fmt: on
    return np.array(ndc_proj_mat, dtype=np.float32)


class NvDiffRenderer:
    def __init__(self, img_size=(512, 512), device="cuda"):
        super().__init__()

        self.img_size = img_size
        self.device = device

        self.glctx = None

    def render(self, verts, tris, tex, uvs, tris_uvs, cam_intris, img_size=None):
        """
        Parameters:
            verts -- torch.tensor, size (B, N, 3)
            tris  -- torch.tensor, size (B, M, 3) or (M, 3), triangles
        """
        rast_out = self.rasterize(verts, tris, cam_intris, img_size)

        # add interpolate and texture operater
        uv_out, _ = dr.interpolate(uvs, rast_out, tris_uvs)
        tex_sampling = dr.texture(tex, uv_out)
        mask = (rast_out[..., 3] > 0).float().unsqueeze(-1)

        return tex_sampling, mask

    def rasterize(self, verts, tris, cam_intris, img_size=None):
        """
        Parameters:
            verts -- torch.Tensor, size (B, N, 3)
            tris  -- torch.Tensor, size (B, M, 3) or (M, 3), triangles
        """
        img_size = img_size if img_size else self.img_size
        ndc_proj = torch.as_tensor(calc_ndc_proj_mat(cam_intris, img_size), device=self.device)

        ## converts to homogeneous coordinates of 3d verts, the direction of y is the same as v
        if verts.shape[-1] == 3:
            verts = torch.cat([verts, torch.ones([*verts.shape[:2], 1]).to(self.device)], dim=-1)
        verts_ndc = (verts @ ndc_proj.permute(0, 2, 1)).contiguous()

        tris = tris.int().to(self.device)

        if self.glctx is None:
            self.glctx = dr.RasterizeGLContext(device=self.device)

        rast_out, _rast_db = dr.rasterize(self.glctx, verts_ndc, tris, resolution=self.img_size)  # type: ignore
        ## rast_out.shape (bs, height, width, 4), where 4 means (u, v, z/w, triangle_id)

        return rast_out

    def get_mask(self, verts, tris, cam_intris, img_size=None):
        rast_out = self.rasterize(verts, tris, cam_intris, img_size)
        mask = (rast_out[..., 3] > 0).float()
        return mask
