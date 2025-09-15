import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial import cKDTree

def chamfer_distance(x, y):
    # x: [B, N, 3]
    # y: [B, M, 3]
    dist = torch.cdist(x, y)      # [B, N, M]
    x_y = dist.min(dim=2)[0]      # [B, N]
    y_x = dist.min(dim=1)[0]     # [B, M]
    cd_x = x_y.mean(dim=1)        # [B]
    cd_y = y_x.mean(dim=1)        # [B]
    total_cd = cd_x + cd_y        # [B]
    return total_cd.mean(), (cd_x.mean(), cd_y.mean())


def chamfer_distance2(x, y):
    """
    計算 Chamfer Distance
    返回: (總距離, (x到y的距離, y到x的距離))
    """
    x = x.to(device)
    y = y.to(device)
    batch_size = x.size(0)

    xx = torch.sum(x ** 2, dim=2, keepdim=True)  # [B, N, 1]
    yy = torch.sum(y ** 2, dim=2, keepdim=True)  # [B, M, 1]
    xy = torch.bmm(x, y.transpose(1, 2))  # [B, N, M]
    dist = xx - 2 * xy + yy.transpose(1, 2)  # [B, N, M]
    dist = torch.sqrt(dist.clamp(min=1e-7))  # 避免 sqrt 負值，取平方根
    # 找到最近距離
    dist_x = torch.min(dist, dim=2, keepdim=True)[0]  # [B, N, 1]
    dist_y = torch.min(dist.transpose(1, 2), dim=2, keepdim=True)[0].transpose(1, 2)  # [B, M, 1] -> [B, 1, M] -> [B, M, 1]
    
    cd_x = torch.mean(dist_x, dim=1)
    cd_y = torch.mean(dist_y, dim=1)

    total_cd = cd_x+ cd_y

    return torch.mean(total_cd), (torch.mean(cd_x), torch.mean(cd_y))


def compute_local_density(points, k=8, method='knn'):
    """
    计算点云的局部密度
    
    Args:
        points: 点云张量 [B, N, 3] 或 [N, 3]
        k: 用于密度计算的邻居数量
        method: 密度计算方法 ('knn', 'gaussian', 'inverse_distance')
    
    Returns:
        density: 每个点的局部密度 [B, N] 或 [N]
    """
    if points.dim() == 3:
        batch_size, num_points, _ = points.shape
        densities = []
        
        for b in range(batch_size):
            density = compute_single_density(points[b], k, method)
            densities.append(density)
        
        return torch.stack(densities, dim=0)
    else:
        return compute_single_density(points, k, method)


def compute_single_density(points, k=8, method='knn'):
    """
    计算单个点云的局部密度
    
    Args:
        points: 点云张量 [N, 3]
        k: 邻居数量
        method: 密度计算方法
    
    Returns:
        density: 每个点的局部密度 [N]
    """
    device = points.device
    points_np = points.detach().cpu().numpy()
    
    # 使用 KDTree 查找邻居
    tree = cKDTree(points_np)
    distances, indices = tree.query(points_np, k=k+1)  # +1 因为包含自己
    distances = distances[:, 1:]  # 排除自己
    
    if method == 'knn':
        # 基于 k 邻居平均距离的密度
        mean_distances = np.mean(distances, axis=1)
        density = 1.0 / (mean_distances + 1e-8)
        
    elif method == 'gaussian':
        # 基于高斯核的密度
        sigma = np.mean(distances[:, -1])  # 使用最远邻居距离作为 sigma
        weights = np.exp(-distances**2 / (2 * sigma**2))
        density = np.sum(weights, axis=1)
        
    elif method == 'inverse_distance':
        # 基于距离倒数的密度
        inv_distances = 1.0 / (distances + 1e-8)
        density = np.sum(inv_distances, axis=1)
    
    else:
        raise ValueError(f"Unknown density method: {method}")
    
    return torch.tensor(density, dtype=torch.float32, device=device)


def density_chamfer_distance(x, y, k=8, density_method='knn', alpha=1.0, beta=1.0):
    """
    计算 Density Chamfer Distance (DCD)
    
    Args:
        x: 源点云 [B, N, 3]
        y: 目标点云 [B, M, 3]
        k: 用于密度计算的邻居数量
        density_method: 密度计算方法 ('knn', 'gaussian', 'inverse_distance')
        alpha: 距离项权重
        beta: 密度项权重
    
    Returns:
        total_dcd: 总的密度倒角距离
        (dcd_x, dcd_y): x到y和y到x的密度倒角距离
    """
    x = x.to(x.device)
    y = y.to(y.device)
    # 添加形狀檢查
    if x.dim() != 3 or y.dim() != 3:
        raise ValueError(f"Expected 3D tensors, got x: {x.shape}, y: {y.shape}")
    batch_size = x.size(0)
    
    # 计算局部密度
    density_x = compute_local_density(x, k=k, method=density_method)  # [B, N]
    density_y = compute_local_density(y, k=k, method=density_method)  # [B, M]
    
    # 归一化密度到 [0, 1] 范围
    density_x = F.normalize(density_x, p=1, dim=1)
    density_y = F.normalize(density_y, p=1, dim=1)
    
    # 计算点对点距离矩阵
    xx = torch.sum(x ** 2, dim=2, keepdim=True)  # [B, N, 1]
    yy = torch.sum(y ** 2, dim=2, keepdim=True)  # [B, M, 1]
    xy = torch.bmm(x, y.transpose(1, 2))  # [B, N, M]
    dist_matrix = xx - 2 * xy + yy.transpose(1, 2)  # [B, N, M]
    dist_matrix = torch.sqrt(dist_matrix.clamp(min=1e-7))
    
    # 从 x 到 y 的密度倒角距离
    min_dist_x2y, min_idx_x2y = torch.min(dist_matrix, dim=2)  # [B, N]
    
    # 获取对应的密度值
    batch_idx = torch.arange(batch_size).unsqueeze(1).expand(-1, x.size(1))  # [B, N]
    corresponding_density_y = density_y[batch_idx, min_idx_x2y]  # [B, N]
    
    # 计算密度加权的距离
    density_weight_x2y = (density_x + corresponding_density_y) / 2.0  # 平均密度作为权重
    weighted_dist_x2y = min_dist_x2y * (alpha + beta * density_weight_x2y)
    dcd_x2y = torch.mean(weighted_dist_x2y, dim=1)  # [B]
    
    # 从 y 到 x 的密度倒角距离
    min_dist_y2x, min_idx_y2x = torch.min(dist_matrix.transpose(1, 2), dim=2)  # [B, M]
    
    # 获取对应的密度值
    batch_idx = torch.arange(batch_size).unsqueeze(1).expand(-1, y.size(1))  # [B, M]
    corresponding_density_x = density_x[batch_idx, min_idx_y2x]  # [B, M]
    
    # 计算密度加权的距离
    density_weight_y2x = (density_y + corresponding_density_x) / 2.0
    weighted_dist_y2x = min_dist_y2x * (alpha + beta * density_weight_y2x)
    dcd_y2x = torch.mean(weighted_dist_y2x, dim=1)  # [B]
    
    # 总的密度倒角距离
    total_dcd = torch.mean(dcd_x2y + dcd_y2x)
    
    return total_dcd, (torch.mean(dcd_x2y), torch.mean(dcd_y2x))


def adaptive_density_chamfer_distance(x, y, k=8, density_method='knn', adaptive_weight=True):
    """
    自适应密度倒角距离 - 根据点云特性自动调整权重
    
    Args:
        x: 源点云 [B, N, 3]
        y: 目标点云 [B, M, 3]
        k: 邻居数量
        density_method: 密度计算方法
        adaptive_weight: 是否使用自适应权重
    
    Returns:
        total_dcd: 总的密度倒角距离
        details: 详细信息字典
    """
    x = x.to(x.device)
    y = y.to(y.device)
    
    # 计算传统的 Chamfer Distance 作为基准
    xx = torch.sum(x ** 2, dim=2, keepdim=True)
    yy = torch.sum(y ** 2, dim=2, keepdim=True)
    xy = torch.bmm(x, y.transpose(1, 2))
    dist_matrix = xx - 2 * xy + yy.transpose(1, 2)
    dist_matrix = torch.sqrt(dist_matrix.clamp(min=1e-7))
    
    traditional_cd_x = torch.mean(torch.min(dist_matrix, dim=2)[0], dim=1)
    traditional_cd_y = torch.mean(torch.min(dist_matrix.transpose(1, 2), dim=2)[0], dim=1)
    traditional_cd = torch.mean(traditional_cd_x + traditional_cd_y)
    
    # 计算密度
    density_x = compute_local_density(x, k=k, method=density_method)
    density_y = compute_local_density(y, k=k, method=density_method)
    
    # 计算密度变异系数（衡量密度分布的不均匀程度）
    cv_x = torch.std(density_x, dim=1) / (torch.mean(density_x, dim=1) + 1e-8)
    cv_y = torch.std(density_y, dim=1) / (torch.mean(density_y, dim=1) + 1e-8)
    avg_cv = torch.mean(cv_x + cv_y)
    
    if adaptive_weight:
        # 根据密度变异系数自适应调整权重
        # 密度变化大的点云需要更强的密度权重
        alpha = 1.0
        beta = torch.clamp(avg_cv, min=0.1, max=2.0)
    else:
        alpha, beta = 1.0, 1.0
    
    # 计算 DCD
    dcd, (dcd_x, dcd_y) = density_chamfer_distance(x, y, k, density_method, alpha, beta)
    
    details = {
        'traditional_cd': traditional_cd.item(),
        'density_weighted_cd': dcd.item(),
        'density_cv_x': torch.mean(cv_x).item(),
        'density_cv_y': torch.mean(cv_y).item(),
        'adaptive_weights': {'alpha': alpha, 'beta': beta.item() if torch.is_tensor(beta) else beta},
        'improvement_ratio': (traditional_cd - dcd) / traditional_cd * 100
    }
    
    return dcd, details


def estimate_normals(points, k=8):
    """
    使用 KDTree 估算法向量
    points: (N, 3) numpy array
    k: 鄰居點數量
    回傳: (N, 3) numpy array, 每個點的法向量
    """
    tree = cKDTree(points)
    normals = []
    for i in range(len(points)):
        _, idxs = tree.query(points[i], k + 1)  # 取自己+K個鄰居
        neighbors = points[idxs[1:]]  # 移除自身，只保留鄰居
        cov = np.cov(neighbors.T)
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        normal = eig_vecs[:, 0]  # 最小特徵值對應向量
        # 統一方向（可根據你場景調整, 例如 z>0）
        if normal[2] < 0:
            normal = -normal
        normals.append(normal)
    return np.stack(normals, axis=0)


def normal_consistency_loss(pred_points, gt_points, k=8):
    """
    計算法向量一致性損失
    pred_points, gt_points: Tensor, 形狀 (B, N, 3)
    回傳: scalar Tensor (loss)
    """
    batch_size, num_points, _ = pred_points.shape
    total_loss = 0.0
    for b in range(batch_size):
        pred_np = pred_points[b].detach().cpu().numpy()
        gt_np = gt_points[b].detach().cpu().numpy()
        pred_normals = estimate_normals(pred_np, k)  # (N, 3)
        gt_normals = estimate_normals(gt_np, k)      # (N, 3)
        pred_normals_tensor = torch.tensor(pred_normals, dtype=torch.float32, device=pred_points.device)
        gt_normals_tensor = torch.tensor(gt_normals, dtype=torch.float32, device=gt_points.device)
        cos_sim = F.cosine_similarity(pred_normals_tensor, gt_normals_tensor, dim=-1)  # shape: (N,)
        total_loss += 1 - cos_sim.mean()
    return total_loss / batch_size


def torch_normal_consistency_loss(pred_points, gt_points, k=8):
    # pred_points/gt_points: [B, N, 3]
    B, N, _ = pred_points.shape
    def get_normals(pc):
        # pc: [N, 3]
        dists = torch.cdist(pc.unsqueeze(0), pc.unsqueeze(0))[0]   # [N, N]
        knn_idx = dists.topk(k+1, dim=-1, largest=False).indices[:, 1:]  # [N, k]
        # neighbors = torch.stack([pc[idx] for idx in knn_idx], dim=0)  # [N, k, 3]
        neighbors = pc[knn_idx] # (N, k, 3)
        mean_neigh = neighbors.mean(dim=1, keepdim=True)             # [N, 1, 3]
        cov = (neighbors - mean_neigh).transpose(1,2) @ (neighbors - mean_neigh)  # [N, 3, 3]
        eigvals, eigvecs = torch.linalg.eigh(cov)                    # [N, 3], [N, 3, 3]
        normals = eigvecs[:, :, 0]                                   # [N, 3]，對應最小特徵值
        return normals
    loss = 0
    for b in range(B):
        normal_pred = get_normals(pred_points[b])
        normal_gt = get_normals(gt_points[b])
        cos_sim = F.cosine_similarity(normal_pred, normal_gt, dim=-1)
        loss += (1 - cos_sim).mean()
    return loss / B



def compute_scale_consistency_loss(recon_points, target_points):
    """計算尺度一致性損失"""
    # 計算點雲的尺度
    recon_scale = torch.sqrt(torch.sum(recon_points**2, dim=-1)).max(dim=-1)[0]
    target_scale = torch.sqrt(torch.sum(target_points**2, dim=-1)).max(dim=-1)
    
    # 尺度比例損失
    scale_ratio = recon_scale / (target_scale + 1e-8)
    scale_loss = F.mse_loss(scale_ratio, torch.ones_like(scale_ratio))
    
    return scale_loss


def compute_scale_consistency_loss_v2(recon_points, target_points):
    """增強的尺度損失 - 多維度約束"""
    # 1. 整體尺度約束
    recon_scale = torch.norm(recon_points, dim=-1).max(dim=-1)[0]
    target_scale = torch.norm(target_points, dim=-1).max(dim=-1)[0]
    global_scale_loss = F.mse_loss(recon_scale, target_scale)
    
    # 2. 各軸向尺度約束（重要！防止圓形化）
    recon_ranges = recon_points.max(dim=1)[0] - recon_points.min(dim=1)[0]  # [B, 3]
    target_ranges = target_points.max(dim=1)[0] - target_points.min(dim=1)[0]
    axis_scale_loss = F.mse_loss(recon_ranges, target_ranges)
    
    return global_scale_loss + axis_scale_loss * 2.0  # 強調軸向約束


def compute_latent_regularization(latent_z):
    """潛在空間正則化"""
    # L2正則化
    l2_reg = torch.mean(torch.sum(latent_z**2, dim=-1))
    
    # 稀疏性正則化
    l1_reg = torch.mean(torch.sum(torch.abs(latent_z), dim=-1))
    
    return 0.8 * l2_reg + 0.2 * l1_reg


def compute_latent_regularization_v2(latent_z):
    """輕量化正則化 - 減少過度約束"""
    # 只使用L2正則化，移除L1稀疏性約束
    l2_reg = torch.mean(torch.sum(latent_z**2, dim=-1))
    return 0.5 * l2_reg  # 權重減半


def compute_local_structure_loss(recon_points, target_points, k=16):
    """計算局部結構保持損失"""
    try:
        B, N, _ = recon_points.shape
        
        # 計算每個點的k近鄰
        recon_dist = torch.cdist(recon_points, recon_points)  # [B, N, N]
        target_dist = torch.cdist(target_points, target_points)
        
        # 獲取k近鄰距離
        recon_knn_dist = torch.topk(recon_dist, k+1, dim=-1, largest=False)[0][:, :, 1:]  # 排除自己
        target_knn_dist = torch.topk(target_dist, k+1, dim=-1, largest=False)[:, :, 1:]
        
        # 計算距離差異
        structure_loss = F.mse_loss(recon_knn_dist, target_knn_dist)
        
        return structure_loss
        
    except Exception as e:
        return torch.tensor(0.0).to(recon_points.device)


def compute_multiscale_structure_loss(recon_points, target_points):
    """多尺度結構保持損失"""
    total_loss = 0
    scales = [8, 16, 32]  # 不同鄰域大小
    
    for k in scales:
        try:
            # 計算不同尺度的局部結構
            recon_dist = torch.cdist(recon_points, recon_points)
            target_dist = torch.cdist(target_points, target_points)
            
            recon_knn = torch.topk(recon_dist, k+1, dim=-1, largest=False)[0][:, :, 1:]
            target_knn = torch.topk(target_dist, k+1, dim=-1, largest=False)[0][:, :, 1:]
            
            scale_loss = F.mse_loss(recon_knn, target_knn)
            total_loss += scale_loss / len(scales)
        except:
            continue
    
    return total_loss

def compute_multiscale_structure_loss_v2(recon_points, target_points):
    """簡化的多尺度結構損失"""
    total_loss = 0
    scales = [8, 16]  # 減少到2個尺度，降低計算量
    
    for k in scales:
        try:
            recon_dist = torch.cdist(recon_points, recon_points)
            target_dist = torch.cdist(target_points, target_points)
            
            recon_knn = torch.topk(recon_dist, min(k+1, recon_points.shape[1]), 
                                 dim=-1, largest=False)[0][:, :, 1:]
            target_knn = torch.topk(target_dist, min(k+1, target_points.shape[1]), 
                                  dim=-1, largest=False)[0][:, :, 1:]
            
            # 確保形狀匹配
            min_neighbors = min(recon_knn.shape[-1], target_knn.shape[-1])
            recon_knn = recon_knn[:, :, :min_neighbors]
            target_knn = target_knn[:, :, :min_neighbors]
            
            scale_loss = F.mse_loss(recon_knn, target_knn)
            total_loss += scale_loss / len(scales)
        except:
            continue
    
    return total_loss

def compute_boundary_preservation_loss(recon_points, target_points):
    """邊界保持損失 - 對雙層結構重要"""
    try:
        # 計算點到質心的距離分布
        recon_center = torch.mean(recon_points, dim=1, keepdim=True)
        target_center = torch.mean(target_points, dim=1, keepdim=True)
        
        recon_radial = torch.norm(recon_points - recon_center, dim=-1)
        target_radial = torch.norm(target_points - target_center, dim=-1)
        
        # 保持徑向分布
        radial_loss = F.mse_loss(recon_radial, target_radial)
        
        # 保持高度分布（Z軸）
        recon_z = recon_points[:, :, 2]
        target_z = target_points[:, :, 2]
        height_loss = F.mse_loss(recon_z, target_z)
        
        return radial_loss + height_loss * 2.0  # 強調高度保持
        
    except:
        return torch.tensor(0.0).to(recon_points.device)
    

def point_cloud_uniform_loss(pc, k=8, alpha=0.7):
    """
    提升均勻性損失：結合k鄰域標準差均值與最大值
    均勻性是所有點的k鄰域距離的變異數愈小愈好
    alpha（0~1）：控制最大值懲罰強度，值大則懲罰越不均情況
    """
    # pc: [B,N,3]
    dist = torch.cdist(pc, pc)  # [B,N,N]
    knn_dist, _ = dist.topk(k+1, dim=-1, largest=False)  # [B,N,k+1]
    knn_dist = knn_dist[:,:,1:]  # 排除自己本身
    std_dist = knn_dist.std(dim=-1)    # [B,N]
    mean_loss = std_dist.mean()
    max_loss = std_dist.max()
    
    return mean_loss + max_loss * alpha


def stratified_uniform_loss(pc, k=8):
    "層內均勻性：在每個層內各自施以均勻懲罰（而不是全域）"
    # pc: [B, N, 3]
    z_coords = pc[:, :, 2]
    z_median = torch.median(z_coords, dim=1, keepdim=True)[0]
    upper_mask = z_coords > z_median
    lower_mask = z_coords <= z_median

    loss = 0.0
    for mask in [upper_mask, lower_mask]:
        pc_sub = torch.where(mask.unsqueeze(-1), pc, torch.zeros_like(pc))
        # 選出本層
        n_sub = mask.sum(dim=1).clamp(min=2)
        # 只優化真有點的batch
        # knn loss（只對非零的部分）
        for b in range(pc.shape[0]):
            sel = pc_sub[b][mask[b]]
            if sel.shape[0] > 1:
                dist = torch.cdist(sel.unsqueeze(0), sel.unsqueeze(0))[0]
                knn = dist.topk(min(k+1, sel.shape[0]), largest=False)[0][:, 1:]
                std = knn.std(dim=-1).mean()
                loss += std
    return loss / (2 * pc.shape[0])   # 兩層分開平均



def compute_layer_balance_loss(recon_points, target_points):
    """
    層間密度平衡損失 - 確保上下層點數相近
    """
    try:
        # 按Z軸中位數分上下層
        z_coords = recon_points[:, :, 2]  # [B, N]
        z_median = torch.median(z_coords, dim=1, keepdim=True)[0]  # [B, 1]
        
        # 分層計算點數
        upper_mask = z_coords > z_median  # [B, N]
        lower_mask = z_coords <= z_median
        
        upper_count = upper_mask.float().sum(dim=1)  # [B]
        lower_count = lower_mask.float().sum(dim=1)
        
        # 目標：上下層點數比例接近1:1
        target_ratio = torch.ones_like(upper_count)
        actual_ratio = upper_count / (lower_count + 1e-8)
        
        balance_loss = F.mse_loss(actual_ratio, target_ratio)
        
        return balance_loss
        
    except:
        return torch.tensor(0.0, device=recon_points.device, requires_grad=True)



# def test(x=696):
#     print(f"testing... {x}")
