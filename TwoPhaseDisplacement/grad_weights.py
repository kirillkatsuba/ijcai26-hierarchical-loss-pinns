import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from sklearn.metrics import accuracy_score



def compute_loss_target(loss_orth, loss_cons, loss_smap, loss_lbpr, 
                        kapp_orth, kapp_cons, kapp_smap, kapp_lbpr,
                        kappa, dt, eps=1.0e-6):
    kapp_orth_updated = max(eps, kapp_orth + 0.0 * dt * ((0.0 + loss_orth).cpu().detach().numpy() - kapp_orth))
    kapp_cons_updated = max(eps, kapp_cons + kappa * dt * ((0.0 + loss_cons).cpu().detach().numpy() - kapp_cons))
    kapp_lbpr_updated = max(eps, kapp_lbpr + kappa * dt * ((0.0 + loss_lbpr).cpu().detach().numpy() - kapp_lbpr))
    kapp_smap_updated = max(eps, kapp_smap + kappa * dt * ((0.0 + loss_smap).cpu().detach().numpy() - kapp_smap))
    return (kapp_orth_updated,
            kapp_cons_updated,
            kapp_smap_updated,
            kapp_lbpr_updated)


def compute_loss_weights_simple(loss_orth, loss_cons, loss_smap, loss_lbpr, 
                                kapp_orth, kapp_cons, kapp_smap, kapp_lbpr,
                                eps=1.0e-6):
    w_orth = 1.0
    w_smap = 1.0
    w_cons = 1.0
    loss_orth_numpy = (0.0 + loss_orth).cpu().detach().numpy() / max(eps, kapp_orth)
    loss_cons_numpy = (0.0 + loss_cons).cpu().detach().numpy() / max(eps, kapp_cons)
    loss_smap_numpy = (0.0 + loss_smap).cpu().detach().numpy() / max(eps, kapp_smap)
    loss_lbpr_numpy = (0.0 + loss_lbpr).cpu().detach().numpy() / max(eps, kapp_lbpr)


    w_cons = np.exp(-loss_orth_numpy)
    w_lbpr = np.exp(-max(loss_cons_numpy, loss_orth_numpy))
    w_smap = np.exp(-max(loss_cons_numpy, loss_orth_numpy, loss_lbpr_numpy))

    w_summ = w_orth + w_smap + w_cons + w_lbpr
    w_orth = w_orth / w_summ
    w_cons = w_cons / w_summ
    w_smap = w_smap / w_summ
    w_lbpr = w_lbpr / w_summ
    return (w_orth, w_cons, w_smap, w_lbpr)


def flat_the_gradient(nnet):
    n_params = compute_number_of_parameters(nnet)
    n_layers = len(nnet.fc)
    grad_flat = np.zeros((n_params))
    idx_vec = 0
    for idx in range(0, n_layers, 2):
        w_grad = (0.0 + nnet.fc[idx].weight.grad).cpu().detach().numpy().flatten()
        b_grad = (0.0 + nnet.fc[idx].bias.grad).cpu().detach().numpy().flatten()
        grad_flat[idx_vec : (idx_vec + w_grad.size)] = w_grad.copy()
        idx_vec = idx_vec + w_grad.size
        grad_flat[idx_vec : (idx_vec + b_grad.size)] = b_grad.copy()
        idx_vec = idx_vec + b_grad.size
    return grad_flat




def compute_weights_grad_value(lv_orth, lg_orth,
                               lv_norm, lg_norm,
                               lv_grad, lg_grad,
                               kappa, eps=1.0e-6):
    ### what is the next step?
    w_orth = 1.0 # / (eps + np.sqrt(np.sum(lg_orth, lg_orth)))
    lv_orth_numpy = (0.0 + lv_orth).cpu().detach().numpy()
    lv_norm_numpy = (0.0 + lv_norm).cpu().detach().numpy()
    lv_grad_numpy = (0.0 + lv_grad).cpu().detach().numpy()
    w_norm = np.exp(-kappa * lv_orth_numpy) # / (eps + np.sqrt(np.sum(lg_norm, lg_norm)))
    w_summ = w_orth + w_norm
    w_orth = w_orth / w_summ
    w_norm = w_norm / w_summ

    lg_comb = w_orth * lg_orth + w_norm * lg_norm

    lv_norm_numpy = (0.0 + lv_norm).cpu().detach().numpy()
    w_grad = np.exp(-kappa * max(lv_orth_numpy, lv_norm_numpy))
    # grad_factor = np.sum(lg_orth * lg_orth) + np.sum(lg_norm * lg_norm)
    # grad_factor = np.sqrt(np.sum(lg_comb * lg_comb))
    grad_factor = 0.50 * np.sqrt(np.sum(lg_comb * lg_comb)) / (eps + np.sqrt(np.sum(lg_grad * lg_grad)))
    w_grad = grad_factor * w_grad
    w_summ = eps + w_orth + w_norm + w_grad
    w_orth = w_orth / w_summ
    w_norm = w_norm / w_summ
    w_grad = w_grad / w_summ
    return (w_orth, w_norm, w_grad)