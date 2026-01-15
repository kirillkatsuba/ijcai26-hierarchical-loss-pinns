import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import clear_output, display
from diffusion_equation import compute_solution
import pandas as pd
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

def resample_collocation_points(points, t, x, y, pwat, poil, kwat, koil, perm, device):
    """
    Resample collocation points for better coverage of the domain.
    This is important for PINNs to avoid overfitting to specific regions.
    
    Args:
        points: Current points tensor
        t, x, y, pwat, poil, kwat, koil: Domain parameters
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
    
    # Extract components and ensure gradient tracking
    # CRITICAL: Must extract BEFORE forward pass and set requires_grad=True
    t = points_batch[:, 0:1].clone().detach().requires_grad_(True)
    x = points_batch[:, 1:2].clone().detach().requires_grad_(True)
    y = points_batch[:, 2:3].clone().detach().requires_grad_(True)
    
    # Other parameters (no gradients needed)
    pwat = points_batch[:, 3:4]
    poil = points_batch[:, 4:5]
    kwat = points_batch[:, 5:6]
    koil = points_batch[:, 6:7]
    
    # Reconstruct input with gradient-enabled variables
    model_input = torch.cat([t, x, y, pwat, poil, kwat, koil], dim=1)
    
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

def get_plot_data(pwat=1.5, poil=2.0, kwat=1.5, koil = 0.3):
    perm = np.load('perm_2.npy')
    nx0, nx1 = perm.shape
    nx2 = 1
    perm = np.reshape(perm, (nx0, nx1, nx2))
    poro = 0.1 + np.zeros((nx0, nx1, nx2))

    dx0 = 1.0 / nx0
    dx1 = 1.0 / nx1
    dx2 = 1.0 / nx2
    vr = 0.3
    
    if isinstance(pwat, float):
        pwat_list = pwat * torch.ones(64 * 64)
        poil_list = poil * torch.ones(64 * 64)
        kwat_list = kwat * torch.ones(64 * 64)
        koil_list = koil * torch.ones(64 * 64)

    pmin = 0.0
    pmax = 1.0

    dt = 0.15e-1
    niter = 100


    swat = np.zeros((nx0, nx1, nx2))
    soil = np.ones((nx0, nx1, nx2))


    pres, swat, soil = compute_solution(perm, poro,
                                        dx0, dx1, dx2, dt * niter, niter,
                                        pwat, kwat, poil, koil, vr,
                                        pmin=0.0, pmax=1.0)
    return pres, swat, soil

def plot_validation(model, PRES, SWAT, SOIL, device='cpu', pwat=1.5, poil=2.0, kwat=1.5, koil = 0.3):
    perm = np.load('perm_2.npy')
    nx0, nx1 = perm.shape
    nx2 = 1
    perm = np.reshape(perm, (nx0, nx1, nx2))
    poro = 0.1 + np.zeros((nx0, nx1, nx2))

    dx0 = 1.0 / nx0
    dx1 = 1.0 / nx1
    dx2 = 1.0 / nx2
    vr = 0.3
    
    if isinstance(pwat, float):
        pwat_list = pwat * torch.ones(64 * 64)
        poil_list = poil * torch.ones(64 * 64)
        kwat_list = kwat * torch.ones(64 * 64)
        koil_list = koil * torch.ones(64 * 64)

    pmin = 0.0
    pmax = 1.0

    dt = 0.15e-1
    niter = 100

    pres, swat, soil = PRES, SWAT, SOIL
    
    time_for_model = (niter * dt) * torch.ones(64 * 64)
    x_for_model = dx0 * torch.arange(64)
    y_for_model = dx1 * torch.arange(64)
    cartesian_points = torch.cartesian_prod(x_for_model, y_for_model)

    model_prediction = model(
        torch.stack(
            (time_for_model, 
             cartesian_points[:, 0], 
             cartesian_points[:, 1],
             pwat_list,
             poil_list,
             kwat_list,
             koil_list), -1).to(device))
    
    model_prediction = model_prediction.cpu().detach().numpy()
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    
    # 0,0 – Water saturation, simulator
    ax = axes[0, 0]
    im = ax.imshow(swat[:, :, :, -1])
    ax.set_title('Water saturation, simulator')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(im, ax=ax)
    
    # 0,1 – Water saturation, PINN
    ax = axes[0, 1]
    im = ax.imshow(model_prediction[:, 2].reshape(nx0, nx1, nx2))
    ax.set_title('Water saturation, PINN')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(im, ax=ax)
    
    # 0,2 – Pressure, simulator
    ax = axes[0, 2]
    im = ax.imshow(pres[:, :, :, -1])
    ax.set_title('Pressure, simulator')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(im, ax=ax)
    
    # 1,0 – Pressure, PINN
    ax = axes[1, 0]
    im = ax.imshow(model_prediction[:, 0].reshape(nx0, nx1, nx2))
    ax.set_title('Pressure, PINN')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(im, ax=ax)
    
    # 1,1 – Water saturation at t = 0 (scatter)
    ax = axes[1, 1]
    ax.set_title('Water saturation at t = 0')
    ax.scatter(np.linspace(0.0, 1.0, nx0), swat[:, 0, 0, -1], label='Simulator')
    ax.scatter(
        np.linspace(0.0, 1.0, nx0),
        model_prediction[:, 2].reshape(nx0, nx1, nx2)[:, 0, 0],
        label='PINN'
    )
    ax.grid()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    
    # 1,2 – Pressure at t = 0 (scatter)
    ax = axes[1, 2]
    ax.set_title('Pressure at t = 0')
    ax.scatter(np.linspace(0.0, 1.0, nx0), pres[:, 0, 0, -1], label='Simulator')
    ax.scatter(
        np.linspace(0.0, 1.0, nx0),
        model_prediction[:, 0].reshape(nx0, nx1, nx2)[:, 0, 0],
        label='PINN'
    )
    ax.grid()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    
    fig.tight_layout()
    plt.show()