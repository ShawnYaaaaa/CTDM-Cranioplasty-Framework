import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial import cKDTree

def chamfer_distance(x, y):
    dist = torch.cdist(x, y)      # [B, N, M]
    x_y = dist.min(dim=2)[0]      # [B, N]
    y_x = dist.min(dim=1)[0]     # [B, M]
    cd_x = x_y.mean(dim=1)        # [B]
    cd_y = y_x.mean(dim=1)        # [B]
    total_cd = cd_x + cd_y        # [B]
    return total_cd.mean(), (cd_x.mean(), cd_y.mean())


def chamfer_distance2(x, y):
    """
    Computes the Chamfer Distance between two point clouds.
    Returns: (total_distance, (x_to_y_distance, y_to_x_distance))
    """
    x = x.to(device)
    y = y.to(device)
    batch_size = x.size(0)

    xx = torch.sum(x ** 2, dim=2, keepdim=True)  # [B, N, 1]
    yy = torch.sum(y ** 2, dim=2, keepdim=True)  # [B, M, 1]
    xy = torch.bmm(x, y.transpose(1, 2))  # [B, N, M]
    dist = xx - 2 * xy + yy.transpose(1, 2)  # [B, N, M]
    dist = torch.sqrt(dist.clamp(min=1e-7))
    dist_x = torch.min(dist, dim=2, keepdim=True)[0]  # [B, N, 1]
    dist_y = torch.min(dist.transpose(1, 2), dim=2, keepdim=True)[0].transpose(1, 2)  # [B, M, 1] -> [B, 1, M] -> [B, M, 1]
    
    cd_x = torch.mean(dist_x, dim=1)
    cd_y = torch.mean(dist_y, dim=1)

    total_cd = cd_x+ cd_y

    return torch.mean(total_cd), (torch.mean(cd_x), torch.mean(cd_y))


def compute_local_density(points, k=8, method='knn'):
    """
    Computes the local density of a point cloud.

    Args:
        points: Point cloud tensor of shape [B, N, 3] or [N, 3]
        k: Number of neighbors used for density estimation
        method: Density estimation method ('knn', 'gaussian', 'inverse_distance')

    Returns:
        density: Per-point local density of shape [B, N] or [N]
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
    Computes the local density for a single point cloud.

    Args:
        points: Point cloud tensor of shape [N, 3]
        k: Number of neighbors
        method: Density estimation method

    Returns:
        density: Per-point local density of shape [N]
    """
    device = points.device
    points_np = points.detach().cpu().numpy()
    
    tree = cKDTree(points_np)
    distances, indices = tree.query(points_np, k=k+1)
    distances = distances[:, 1:]
    
    if method == 'knn':
        mean_distances = np.mean(distances, axis=1)
        density = 1.0 / (mean_distances + 1e-8)
        
    elif method == 'gaussian':
        sigma = np.mean(distances[:, -1])
        weights = np.exp(-distances**2 / (2 * sigma**2))
        density = np.sum(weights, axis=1)
        
    elif method == 'inverse_distance':
        inv_distances = 1.0 / (distances + 1e-8)
        density = np.sum(inv_distances, axis=1)
    
    else:
        raise ValueError(f"Unknown density method: {method}")
    
    return torch.tensor(density, dtype=torch.float32, device=device)


def density_chamfer_distance(x, y, k=8, density_method='knn', alpha=1.0, beta=1.0):
    """
    Computes the Density-aware Chamfer Distance (DCD) between two point clouds.

    Args:
        x: Source point cloud of shape [B, N, 3]
        y: Target point cloud of shape [B, M, 3]
        k: Number of neighbors for density estimation
        density_method: Density estimation method ('knn', 'gaussian', 'inverse_distance')
        alpha: Weight for the distance term
        beta: Weight for the density term

    Returns:
        total_dcd: Total density-aware Chamfer distance
        (dcd_x, dcd_y): Density-weighted distances from x to y and y to x
    """
    x = x.to(x.device)
    y = y.to(y.device)
    if x.dim() != 3 or y.dim() != 3:
        raise ValueError(f"Expected 3D tensors, got x: {x.shape}, y: {y.shape}")
    batch_size = x.size(0)
    
    density_x = compute_local_density(x, k=k, method=density_method)  # [B, N]
    density_y = compute_local_density(y, k=k, method=density_method)  # [B, M]
    
    density_x = F.normalize(density_x, p=1, dim=1)
    density_y = F.normalize(density_y, p=1, dim=1)
    
    xx = torch.sum(x ** 2, dim=2, keepdim=True)  # [B, N, 1]
    yy = torch.sum(y ** 2, dim=2, keepdim=True)  # [B, M, 1]
    xy = torch.bmm(x, y.transpose(1, 2))  # [B, N, M]
    dist_matrix = xx - 2 * xy + yy.transpose(1, 2)  # [B, N, M]
    dist_matrix = torch.sqrt(dist_matrix.clamp(min=1e-7))
    
    min_dist_x2y, min_idx_x2y = torch.min(dist_matrix, dim=2)  # [B, N]
    
    batch_idx = torch.arange(batch_size).unsqueeze(1).expand(-1, x.size(1))  # [B, N]
    corresponding_density_y = density_y[batch_idx, min_idx_x2y]  # [B, N]
    
    density_weight_x2y = (density_x + corresponding_density_y) / 2.0
    weighted_dist_x2y = min_dist_x2y * (alpha + beta * density_weight_x2y)
    dcd_x2y = torch.mean(weighted_dist_x2y, dim=1)  # [B]
    
    min_dist_y2x, min_idx_y2x = torch.min(dist_matrix.transpose(1, 2), dim=2)  # [B, M]
    
    batch_idx = torch.arange(batch_size).unsqueeze(1).expand(-1, y.size(1))  # [B, M]
    corresponding_density_x = density_x[batch_idx, min_idx_y2x]  # [B, M]
    
    density_weight_y2x = (density_y + corresponding_density_x) / 2.0
    weighted_dist_y2x = min_dist_y2x * (alpha + beta * density_weight_y2x)
    dcd_y2x = torch.mean(weighted_dist_y2x, dim=1)  # [B]
    
    total_dcd = torch.mean(dcd_x2y + dcd_y2x)
    
    return total_dcd, (torch.mean(dcd_x2y), torch.mean(dcd_y2x))


def adaptive_density_chamfer_distance(x, y, k=8, density_method='knn', adaptive_weight=True):
    """
    Computes the Adaptive Density-aware Chamfer Distance, automatically adjusting
    weights based on the statistical properties of the input point clouds.

    Args:
        x: Source point cloud of shape [B, N, 3]
        y: Target point cloud of shape [B, M, 3]
        k: Number of neighbors
        density_method: Density estimation method
        adaptive_weight: Whether to use adaptive weight adjustment

    Returns:
        total_dcd: Total density-aware Chamfer distance
        details: Dictionary containing diagnostic information
    """
    x = x.to(x.device)
    y = y.to(y.device)
    
    xx = torch.sum(x ** 2, dim=2, keepdim=True)
    yy = torch.sum(y ** 2, dim=2, keepdim=True)
    xy = torch.bmm(x, y.transpose(1, 2))
    dist_matrix = xx - 2 * xy + yy.transpose(1, 2)
    dist_matrix = torch.sqrt(dist_matrix.clamp(min=1e-7))
    
    traditional_cd_x = torch.mean(torch.min(dist_matrix, dim=2)[0], dim=1)
    traditional_cd_y = torch.mean(torch.min(dist_matrix.transpose(1, 2), dim=2)[0], dim=1)
    traditional_cd = torch.mean(traditional_cd_x + traditional_cd_y)
    
    density_x = compute_local_density(x, k=k, method=density_method)
    density_y = compute_local_density(y, k=k, method=density_method)
    
    cv_x = torch.std(density_x, dim=1) / (torch.mean(density_x, dim=1) + 1e-8)
    cv_y = torch.std(density_y, dim=1) / (torch.mean(density_y, dim=1) + 1e-8)
    avg_cv = torch.mean(cv_x + cv_y)
    
    if adaptive_weight:
        alpha = 1.0
        beta = torch.clamp(avg_cv, min=0.1, max=2.0)
    else:
        alpha, beta = 1.0, 1.0
    
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
    Estimates surface normals using a KDTree-based local neighborhood analysis.

    Args:
        points: Input point cloud as a numpy array of shape (N, 3)
        k: Number of neighbor points used for covariance estimation

    Returns:
        normals: Estimated normals as a numpy array of shape (N, 3)
    """
    tree = cKDTree(points)
    normals = []
    for i in range(len(points)):
        _, idxs = tree.query(points[i], k + 1)
        neighbors = points[idxs[1:]]
        cov = np.cov(neighbors.T)
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        normal = eig_vecs[:, 0]
        if normal[2] < 0:
            normal = -normal
        normals.append(normal)
    return np.stack(normals, axis=0)


def normal_consistency_loss(pred_points, gt_points, k=8):
    """
    Computes normal consistency loss between predicted and ground-truth point clouds
    using numpy-based normal estimation.

    Args:
        pred_points: Predicted point cloud tensor of shape (B, N, 3)
        gt_points: Ground-truth point cloud tensor of shape (B, N, 3)

    Returns:
        loss: Scalar tensor representing the mean normal inconsistency
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
    """
    Computes normal consistency loss entirely in PyTorch without numpy conversion,
    enabling end-to-end gradient flow.

    Args:
        pred_points: Predicted point cloud tensor of shape [B, N, 3]
        gt_points: Ground-truth point cloud tensor of shape [B, N, 3]

    Returns:
        loss: Scalar tensor representing the mean normal inconsistency
    """
    B, N, _ = pred_points.shape
    def get_normals(pc):
        dists = torch.cdist(pc.unsqueeze(0), pc.unsqueeze(0))[0]   # [N, N]
        knn_idx = dists.topk(k+1, dim=-1, largest=False).indices[:, 1:]  # [N, k]
        neighbors = pc[knn_idx] # (N, k, 3)
        mean_neigh = neighbors.mean(dim=1, keepdim=True)             # [N, 1, 3]
        cov = (neighbors - mean_neigh).transpose(1,2) @ (neighbors - mean_neigh)  # [N, 3, 3]
        eigvals, eigvecs = torch.linalg.eigh(cov)                    # [N, 3], [N, 3, 3]
        normals = eigvecs[:, :, 0]                                   # [N, 3]
        return normals
    loss = 0
    for b in range(B):
        normal_pred = get_normals(pred_points[b])
        normal_gt = get_normals(gt_points[b])
        cos_sim = F.cosine_similarity(normal_pred, normal_gt, dim=-1)
        loss += (1 - cos_sim).mean()
    return loss / B



def compute_scale_consistency_loss(recon_points, target_points):
    """Computes scale consistency loss between reconstructed and target point clouds."""
    recon_scale = torch.sqrt(torch.sum(recon_points**2, dim=-1)).max(dim=-1)[0]
    target_scale = torch.sqrt(torch.sum(target_points**2, dim=-1)).max(dim=-1)
    
    scale_ratio = recon_scale / (target_scale + 1e-8)
    scale_loss = F.mse_loss(scale_ratio, torch.ones_like(scale_ratio))
    
    return scale_loss


def compute_scale_consistency_loss_v2(recon_points, target_points):
    """Enhanced scale consistency loss with multi-dimensional constraints to prevent shape degeneracy."""
    recon_scale = torch.norm(recon_points, dim=-1).max(dim=-1)[0]
    target_scale = torch.norm(target_points, dim=-1).max(dim=-1)[0]
    global_scale_loss = F.mse_loss(recon_scale, target_scale)
    
    recon_ranges = recon_points.max(dim=1)[0] - recon_points.min(dim=1)[0]  # [B, 3]
    target_ranges = target_points.max(dim=1)[0] - target_points.min(dim=1)[0]
    axis_scale_loss = F.mse_loss(recon_ranges, target_ranges)
    
    return global_scale_loss + axis_scale_loss * 2.0


def compute_latent_regularization(latent_z):
    """Computes latent space regularization combining L2 and L1 sparsity penalties."""
    l2_reg = torch.mean(torch.sum(latent_z**2, dim=-1))
    
    l1_reg = torch.mean(torch.sum(torch.abs(latent_z), dim=-1))
    
    return 0.8 * l2_reg + 0.2 * l1_reg


def compute_latent_regularization_v2(latent_z):
    """Lightweight latent space regularization using only L2 penalty to reduce over-constraint."""
    l2_reg = torch.mean(torch.sum(latent_z**2, dim=-1))
    return 0.5 * l2_reg


def compute_local_structure_loss(recon_points, target_points, k=16):
    """Computes local structural preservation loss based on k-nearest neighbor distance distributions."""
    try:
        B, N, _ = recon_points.shape
        
        recon_dist = torch.cdist(recon_points, recon_points)  # [B, N, N]
        target_dist = torch.cdist(target_points, target_points)
        
        recon_knn_dist = torch.topk(recon_dist, k+1, dim=-1, largest=False)[0][:, :, 1:]
        target_knn_dist = torch.topk(target_dist, k+1, dim=-1, largest=False)[:, :, 1:]
        
        structure_loss = F.mse_loss(recon_knn_dist, target_knn_dist)
        
        return structure_loss
        
    except Exception as e:
        return torch.tensor(0.0).to(recon_points.device)


def compute_multiscale_structure_loss(recon_points, target_points):
    """Computes multi-scale structural preservation loss across multiple neighborhood sizes."""
    total_loss = 0
    scales = [8, 16, 32]
    
    for k in scales:
        try:
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
    """Simplified multi-scale structural loss with reduced scales for improved computational efficiency."""
    total_loss = 0
    scales = [8, 16]
    
    for k in scales:
        try:
            recon_dist = torch.cdist(recon_points, recon_points)
            target_dist = torch.cdist(target_points, target_points)
            
            recon_knn = torch.topk(recon_dist, min(k+1, recon_points.shape[1]), 
                                 dim=-1, largest=False)[0][:, :, 1:]
            target_knn = torch.topk(target_dist, min(k+1, target_points.shape[1]), 
                                  dim=-1, largest=False)[0][:, :, 1:]
            
            min_neighbors = min(recon_knn.shape[-1], target_knn.shape[-1])
            recon_knn = recon_knn[:, :, :min_neighbors]
            target_knn = target_knn[:, :, :min_neighbors]
            
            scale_loss = F.mse_loss(recon_knn, target_knn)
            total_loss += scale_loss / len(scales)
        except:
            continue
    
    return total_loss

def compute_boundary_preservation_loss(recon_points, target_points):
    """Computes boundary preservation loss critical for maintaining double-layer point cloud structures."""
    try:
        recon_center = torch.mean(recon_points, dim=1, keepdim=True)
        target_center = torch.mean(target_points, dim=1, keepdim=True)
        
        recon_radial = torch.norm(recon_points - recon_center, dim=-1)
        target_radial = torch.norm(target_points - target_center, dim=-1)
        
        radial_loss = F.mse_loss(recon_radial, target_radial)
        
        recon_z = recon_points[:, :, 2]
        target_z = target_points[:, :, 2]
        height_loss = F.mse_loss(recon_z, target_z)
        
        return radial_loss + height_loss * 2.0
        
    except:
        return torch.tensor(0.0).to(recon_points.device)
    

def point_cloud_uniform_loss(pc, k=8, alpha=0.7):
    """
    Computes a uniformity loss that penalizes uneven point distributions by combining
    the mean and maximum of per-point k-neighbor distance standard deviations.

    Args:
        pc: Input point cloud tensor of shape [B, N, 3]
        k: Number of neighbors for local distribution analysis
        alpha: Weight controlling the penalty strength on the maximum deviation term
    """
    dist = torch.cdist(pc, pc)  # [B,N,N]
    knn_dist, _ = dist.topk(k+1, dim=-1, largest=False)  # [B,N,k+1]
    knn_dist = knn_dist[:,:,1:]
    std_dist = knn_dist.std(dim=-1)    # [B,N]
    mean_loss = std_dist.mean()
    max_loss = std_dist.max()
    
    return mean_loss + max_loss * alpha


def stratified_uniform_loss(pc, k=8):
    """Computes intra-layer uniformity loss by applying separate uniformity penalties
    to the upper and lower strata of the point cloud partitioned by the median Z-coordinate."""
    z_coords = pc[:, :, 2]
    z_median = torch.median(z_coords, dim=1, keepdim=True)[0]
    upper_mask = z_coords > z_median
    lower_mask = z_coords <= z_median

    loss = 0.0
    for mask in [upper_mask, lower_mask]:
        pc_sub = torch.where(mask.unsqueeze(-1), pc, torch.zeros_like(pc))
        n_sub = mask.sum(dim=1).clamp(min=2)
        for b in range(pc.shape[0]):
            sel = pc_sub[b][mask[b]]
            if sel.shape[0] > 1:
                dist = torch.cdist(sel.unsqueeze(0), sel.unsqueeze(0))[0]
                knn = dist.topk(min(k+1, sel.shape[0]), largest=False)[0][:, 1:]
                std = knn.std(dim=-1).mean()
                loss += std
    return loss / (2 * pc.shape[0])



def compute_layer_balance_loss(recon_points, target_points):
    """
    Computes an inter-layer density balance loss to ensure approximately equal
    point distribution between the upper and lower halves of the point cloud.
    """
    try:
        z_coords = recon_points[:, :, 2]  # [B, N]
        z_median = torch.median(z_coords, dim=1, keepdim=True)[0]  # [B, 1]
        
        upper_mask = z_coords > z_median  # [B, N]
        lower_mask = z_coords <= z_median
        
        upper_count = upper_mask.float().sum(dim=1)  # [B]
        lower_count = lower_mask.float().sum(dim=1)
        
        target_ratio = torch.ones_like(upper_count)
        actual_ratio = upper_count / (lower_count + 1e-8)
        
        balance_loss = F.mse_loss(actual_ratio, target_ratio)
        
        return balance_loss
        
    except:
        return torch.tensor(0.0, device=recon_points.device, requires_grad=True)