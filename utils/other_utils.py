import torch


def barycentric_coordinates_tri(torch_xyz, torch_p):
    """
    计算批量三角形重心坐标
    torch_xyz: 三角形的3个顶点，形状为 (3, N, 3) 的 tensor，其中 N 是三角形的数量
    torch_p: 目标点，形状为 (N, 3) 的 tensor，其中 N 是采样点的数量
    返回：形状为 (N, 3) 的 tensor，表示每个点的重心坐标
    """
    # 取出三角形的三个顶点坐标
    A = torch_xyz[0]  # 形状为 (N, 3)
    B = torch_xyz[1]  # 形状为 (N, 3)
    C = torch_xyz[2]  # 形状为 (N, 3)
    
    # 目标点的坐标
    P = torch_p  # 形状为 (N, 3)

    # 计算向量
    v0 = B - A  # 形状为 (N, 3)
    v1 = C - A  # 形状为 (N, 3)
    v2 = P - A   # 形状为 (N, 3)

    # 计算面积相关系数
    d00 = torch.sum(v0 * v0, dim=1)  # 形状为 (N,)
    d01 = torch.sum(v0 * v1, dim=1)  # 形状为 (N,)
    d11 = torch.sum(v1 * v1, dim=1)  # 形状为 (N,)
    d20 = torch.sum(v2 * v0, dim=1)  # 形状为 (N,)
    d21 = torch.sum(v2 * v1, dim=1)  # 形状为 (N,)

    denom = d00 * d11 - d01 * d01  # 形状为 (N,)

    # 计算重心坐标
    v = (d11 * d20 - d01 * d21) / denom  # 形状为 (N,)
    w = (d00 * d21 - d01 * d20) / denom  # 形状为 (N,)
    u = 1.0 - v - w  # 形状为 (N,)

    return torch.stack([u, v, w], dim=-1)  # 返回 (N, 3)



