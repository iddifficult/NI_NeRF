import torch
import numpy as np


def get_coordinate(w,h,l,size_voxel):
    
    wsize,hsize,lsize = size_voxel
    x = torch.tensor(np.linspace(-(w-1)/2*wsize, (w-1)/2*wsize, w))
    y = torch.tensor(np.linspace(-(h-1)/2*hsize, (h-1)/2*hsize, h))
    z = torch.tensor(np.linspace(-(l-1)/2*lsize, (l-1)/2*lsize, l))

    # 使用 torch.meshgrid() 生成坐标网格
    X, Y, Z = torch.meshgrid(x, y, z,indexing='ij')
    coordinate = torch.stack([X, Y, Z], dim=-1)

    return coordinate

def render(u,rays,dis):

        dist_weght = (dis[..., 1:] - dis[..., :-1])
        dist_weght = torch.cat([dist_weght, torch.Tensor([1e-10]).cuda()], -1)
        dist_weght = dist_weght[ None, :] * torch.norm(rays[..., None, :], dim=-1)
        pro_prd = torch.sum(u*dist_weght, dim=-1)

        return pro_prd