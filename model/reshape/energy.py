import torch
import torch.nn as nn
import numpy as np


class Energy(nn.Module):
    def __init__(self, options, render_mask, transform, source_mesh, correction_strength, dtype=torch.float32):
        """
        :param source_mesh: torch.tensor, 2 x Hm x Wm
        :param correction_strength:  torch.tensor, Hm x Wm
        """
        super(Energy, self).__init__()

        self.opt = options

        self.source_mesh = torch.tensor(source_mesh, dtype=dtype)
        self.target_mesh = torch.tensor(transform, dtype=dtype)
        self.mask = torch.tensor(render_mask, dtype=dtype)
        self.mesh = torch.tensor(source_mesh, dtype=dtype)
        self.correction_strength = torch.tensor(correction_strength, dtype=dtype)

        self.source_mesh = nn.Parameter(self.source_mesh, requires_grad=False)
        self.target_mesh = nn.Parameter(self.target_mesh, requires_grad=False)
        self.mask = nn.Parameter(self.mask, requires_grad=False)
        self.mesh = nn.Parameter(self.mesh, requires_grad=True)
        self.correction_strength = nn.Parameter(self.correction_strength, requires_grad=False)

    def forward(self):
        # face energy term
        face_energy = self.mask * (self.mesh - self.target_mesh) ** 2
        face_energy = face_energy.mean()

        # Line-Bending Term
        coordinate_padding_u = torch.zeros_like(self.correction_strength[1:, :]).unsqueeze(0)
        mesh_diff_u = self.mesh[:, 1:, :] - self.mesh[:, :-1, :]
        mesh_diff_u = torch.cat((mesh_diff_u, coordinate_padding_u), dim=0)
        source_mesh_diff_u = self.source_mesh[:, 1:, :] - self.source_mesh[:, :-1, :]
        unit_source_mesh_diff_u = source_mesh_diff_u / torch.norm(source_mesh_diff_u, dim=0).unsqueeze(0)
        unit_source_mesh_diff_u = torch.cat((unit_source_mesh_diff_u, coordinate_padding_u), dim=0)
        line_bending_u_loss = torch.square(torch.norm(torch.cross(mesh_diff_u, unit_source_mesh_diff_u, dim=0), dim=0))

        coordinate_padding_v = torch.zeros_like(self.correction_strength[:, 1:]).unsqueeze(0)
        mesh_diff_v = self.mesh[:, :, 1:] - self.mesh[:, :, :-1]
        mesh_diff_v = torch.cat((mesh_diff_v, coordinate_padding_v), dim=0)
        source_mesh_diff_v = self.source_mesh[:, :, 1:] - self.source_mesh[:, :, :-1]
        unit_source_mesh_diff_v = source_mesh_diff_v / torch.norm(source_mesh_diff_v, dim=0).unsqueeze(0)
        unit_source_mesh_diff_v = torch.cat((unit_source_mesh_diff_v, coordinate_padding_v), dim=0)
        line_bending_v_loss = torch.square(torch.norm(torch.cross(mesh_diff_v, unit_source_mesh_diff_v, dim=0), dim=0))

        line_bending_term = (line_bending_u_loss.mean() + line_bending_v_loss.mean()) / 2

        # Regularization Term
        regularization_u_loss = torch.square(torch.norm(mesh_diff_u, dim=0))
        regularization_v_loss = torch.square(torch.norm(mesh_diff_v, dim=0))
        regularization_term = (regularization_u_loss.mean() + regularization_v_loss.mean()) / 2

        # boundary constraint
        left = (self.mesh[0, :, 0] - self.source_mesh[0, :, 0]) ** 2
        right = (self.mesh[0, :, self.mesh.shape[2] - 1] - self.source_mesh[0, :, self.source_mesh.shape[2] - 1]) ** 2
        top = (self.mesh[1, 0, :] - self.source_mesh[1, 0, :]) ** 2
        bottom = (self.mesh[1, self.mesh.shape[1] - 1, :] - self.source_mesh[1, self.source_mesh.shape[1] - 1, :]) ** 2
        boundary_constraint = left.mean() + right.mean() + top.mean() + bottom.mean()

        energy = (
            self.opt["face_energy"] * face_energy
            + self.opt["line_bending"] * line_bending_term
            + self.opt["regularization"] * regularization_term
            + self.opt["boundary_constraint"] * boundary_constraint
        )

        return energy


def opt(H, W, render_mask, transform, uniform_mesh):
    import torch.optim as optim

    correction_strength = torch.ones([H, W])
    options = {"face_energy": 10, "line_bending": 2, "regularization": 2, "boundary_constraint": 2}
    model = Energy(options, render_mask, transform, uniform_mesh, correction_strength)
    model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=100)

    for i in range(100):
        optimizer.zero_grad()
        loss = model.forward()
        loss.backward()
        # import pdb

        # pdb.set_trace()
        optimizer.step()

    print(i, loss)
    mesh = model.mesh
    return mesh
