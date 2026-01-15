import numpy as np
import torch

def get_batch_indices(total_size, batch_size, shuffle=True):
    """Generate batch indices for one epoch."""
    if shuffle:
        indices = torch.randperm(total_size)
    else:
        indices = torch.arange(total_size)
    
    for i in range(0, total_size, batch_size):
        yield indices[i:min(i + batch_size, total_size)]

def create_batches(data, batch_size, shuffle=True):
    """
    Create mini-batches from data.
    
    Args:
        data: Tensor of shape (N, features)
        batch_size: Size of each batch
        shuffle: Whether to shuffle data before batching
        
    Returns:
        List of batch indices
    """
    n_samples = data.shape[0]
    indices = torch.randperm(n_samples) if shuffle else torch.arange(n_samples)
    
    # Create batch indices
    batch_indices = []
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices.append(indices[start_idx:end_idx])
    
    return batch_indices

def resample_collocation_points(points, t, x, y, perm, device):
    """
    Resample collocation points for better coverage of the domain.
    This is important for PINNs to avoid overfitting to specific regions.
    
    Args:
        points: Current points tensor
        t, x, y: Domain parameters
        perm: Permeability data
        device: torch device
        
    Returns:
        New points tensor, updated perm_vec
    """
    # Resample with same distribution as original
    # You may want to implement adaptive sampling based on loss
    n_points = points.shape[0]
    
    # Assuming uniform sampling - modify based on your domain
    new_points = points.clone()
    # Add your resampling logic here
    
    new_perm_vec = torch.tensor(
        perm[list(x.cpu().detach().numpy().astype(int)), 
             list(y.cpu().detach().numpy().astype(int)), -1].astype(np.float32)
    ).to(device)
    
    return new_points, new_perm_vec

def compute_pde_residuals(model, points_batch, perm_vec_batch):
    """
    Compute PDE residuals with proper gradient tracking.
    
    KEY FIXES:
    1. Separate variable extraction with requires_grad=True
    2. Use retain_graph=True for multiple grad calls
    3. Ensure all tensors are on same device
    """
    pwat = 2.0
    poil = 4.0
    kwat = 1.0
    koil = 0.3
    
    # Extract components and ensure gradient tracking
    # CRITICAL: Must extract BEFORE forward pass and set requires_grad=True
    t = points_batch[:, 0:1].clone().detach().requires_grad_(True)
    x = points_batch[:, 1:2].clone().detach().requires_grad_(True)
    y = points_batch[:, 2:3].clone().detach().requires_grad_(True)
    
    # Reconstruct input with gradient-enabled variables
    model_input = torch.cat([t, x, y], dim=1)
    
    # Forward pass
    model_res = model(model_input)
    
    # Ones vector for gradient computation
    ones = torch.ones_like(model_res[:, 0], requires_grad=False)
    
    # ========================================================================
    # GRADIENT COMPUTATIONS WITH PROPER FLAGS
    # ========================================================================
    
    # Water saturation equation: ∂S_w/∂t + ∂u_wx/∂x + ∂u_wy/∂y = 0
    dswat_dt = torch.autograd.grad(
        outputs=model_res[:, 2],
        inputs=t,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        allow_unused=False  # Should be used
    )[0]
    
    duwat_x_dx = torch.autograd.grad(
        outputs=model_res[:, 5],
        inputs=x,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        allow_unused=False
    )[0]
    
    duwat_y_dy = torch.autograd.grad(
        outputs=model_res[:, 6],
        inputs=y,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        allow_unused=False
    )[0]
    
    r1 = 0.1 * dswat_dt + duwat_x_dx + duwat_y_dy
    
    # Oil saturation equation: ∂S_o/∂t + ∂u_ox/∂x + ∂u_oy/∂y = 0
    dsoil_dt = torch.autograd.grad(
        outputs=model_res[:, 1],
        inputs=t,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        allow_unused=False
    )[0]
    
    duoil_x_dx = torch.autograd.grad(
        outputs=model_res[:, 3],
        inputs=x,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        allow_unused=False
    )[0]
    
    duoil_y_dy = torch.autograd.grad(
        outputs=model_res[:, 4],
        inputs=y,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        allow_unused=False
    )[0]
    
    r2 = 0.1 * dsoil_dt + duoil_x_dx + duoil_y_dy
    
    # Darcy's law for water - x component
    dpres_dx = torch.autograd.grad(
        outputs=model_res[:, 0],
        inputs=x,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        allow_unused=False
    )[0]
    
    r3_x1 = model_res[:, 5] + kwat * perm_vec_batch * \
            model_res[:, 2]**pwat * dpres_dx
    
    # Darcy's law for water - y component
    dpres_dy = torch.autograd.grad(
        outputs=model_res[:, 0],
        inputs=y,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        allow_unused=False
    )[0]
    
    r3_x2 = model_res[:, 6] + kwat * perm_vec_batch * \
            model_res[:, 2]**pwat * dpres_dy
    
    # Darcy's law for oil - x component (reuse dpres_dx)
    r4_x1 = model_res[:, 3] + koil * (1/3) * perm_vec_batch * \
            model_res[:, 1]**poil * dpres_dx
    
    # Darcy's law for oil - y component (reuse dpres_dy)
    r4_x2 = model_res[:, 4] + koil * (1/3) * perm_vec_batch * \
            model_res[:, 1]**poil * dpres_dy
    
    # Saturation constraint: S_w + S_o = 1
    r5 = model_res[:, 2] + model_res[:, 1] - ones
    
    return r1, r2, r3_x1, r3_x2, r4_x1, r4_x2, r5, model_res


def get_batch_indices(total_size, batch_size, shuffle=True):
    """Generate batch indices."""
    if shuffle:
        indices = torch.randperm(total_size)
    else:
        indices = torch.arange(total_size)
    
    for i in range(0, total_size, batch_size):
        yield indices[i:min(i + batch_size, total_size)]