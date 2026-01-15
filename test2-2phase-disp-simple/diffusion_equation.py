import numpy as np

import matplotlib
matplotlib.rcParams['image.cmap'] = 'jet'
import matplotlib.pyplot as plt

from numba import jit, prange
from tqdm import trange
import scipy.special as sp

@jit(nopython=True, fastmath=True)
def compute_rel_phase_perm_wat(s, p, k):
    ### s - water saturation
    ### p - power parameter
    ### k - maximum relative phase permeability
    eps = 1.0e-10
    sp = (s + eps) / (1.0 + eps)
    return k * (sp ** p)

@jit(nopython=True, fastmath=True)
def compute_rel_phase_perm_oil(s, p, k):
    ### s - oil saturation
    ### p - power parameter
    ### k - maximum relative phase permeability
    eps = 1.0e-10
    sp = (s + eps) / (1.0 + eps)
    return k * (sp ** p)

@jit(nopython=True, fastmath=True)
def compute_fractional_flow_wat(swat, pwat, kwat, koil, poil, vr):
    rel_perm_wat = compute_rel_phase_perm_wat(0.0 + swat, pwat, kwat)
    rel_perm_oil = compute_rel_phase_perm_wat(1.0 - swat, poil, koil)
    return rel_perm_wat / (rel_perm_wat + rel_perm_oil * vr)

@jit(nopython=True, fastmath=True)
def compute_fractional_flow_oil(swat, pwat, kwat, koil, poil, vr):
    rel_perm_wat = compute_rel_phase_perm_wat(0.0 + swat, pwat, kwat)
    rel_perm_oil = compute_rel_phase_perm_wat(1.0 - swat, poil, koil)
    return rel_perm_oil / (rel_perm_oil + rel_perm_wat / vr)


@jit(nopython=True, fastmath=True)
def compute_absolute_mobility(swat, pwat, kwat,
                              soil, poil, koil,
                              vr):
    ### vr - is the ratio of viscosities: oil / water
    term0 = 0.0 * swat
    term0 += compute_rel_phase_perm_wat(swat, pwat, kwat)
    term0 += compute_rel_phase_perm_oil(soil, poil, koil) * vr
    lmob = 1.0 / term0
    return lmob

@jit(nopython=True, fastmath=True)
def compute_flow_matrix(d_coef, dx0, dx1, dx2):
    nx0, nx1, nx2 = d_coef.shape
    nglob = nx0 * nx1 * nx2
    tmat = np.zeros((nglob, nglob))
    for idx0 in range(nx0):
        for idx1 in prange(nx1):
            for idx2 in prange(nx2):
                idx_glob0 = idx2 + nx2 * idx1 + nx2 * nx1 * idx0
                if idx0 < (nx0 - 1):
                    idx_glob1 = idx2 + nx2 * idx1 + nx2 * nx1 * (idx0 + 1)
                    dc0 = d_coef[idx0 + 0, idx1, idx2]
                    dc1 = d_coef[idx0 + 1, idx1, idx2]
                    tmat[idx_glob0, idx_glob0] += 2.0 * dc0 * dc1 / (dc0 + dc1) * dx1 * dx2 / dx0
                    tmat[idx_glob0, idx_glob1] -= 2.0 * dc0 * dc1 / (dc0 + dc1) * dx1 * dx2 / dx0
                if idx0 > 0:
                    idx_glob1 = idx2 + nx2 * idx1 + nx2 * nx1 * (idx0 - 1)
                    dc0 = d_coef[idx0 + 0, idx1, idx2]
                    dc1 = d_coef[idx0 - 1, idx1, idx2]
                    tmat[idx_glob0, idx_glob0] += 2.0 * dc0 * dc1 / (dc0 + dc1) * dx1 * dx2 / dx0
                    tmat[idx_glob0, idx_glob1] -= 2.0 * dc0 * dc1 / (dc0 + dc1) * dx1 * dx2 / dx0

                if idx1 < (nx1 - 1):
                    idx_glob1 = idx2 + nx2 * (idx1 + 1) + nx2 * nx1 * idx0
                    dc0 = d_coef[idx0, idx1 + 0, idx2]
                    dc1 = d_coef[idx0, idx1 + 1, idx2]
                    tmat[idx_glob0, idx_glob0] += 2.0 * dc0 * dc1 / (dc0 + dc1) * dx2 * dx0 / dx1
                    tmat[idx_glob0, idx_glob1] -= 2.0 * dc0 * dc1 / (dc0 + dc1) * dx2 * dx0 / dx1
                if idx1 > 0:
                    idx_glob1 = idx2 + nx2 * (idx1 - 1) + nx2 * nx1 * idx0
                    dc0 = d_coef[idx0, idx1 + 0, idx2]
                    dc1 = d_coef[idx0, idx1 - 1, idx2]
                    tmat[idx_glob0, idx_glob0] += 2.0 * dc0 * dc1 / (dc0 + dc1) * dx2 * dx0 / dx1
                    tmat[idx_glob0, idx_glob1] -= 2.0 * dc0 * dc1 / (dc0 + dc1) * dx2 * dx0 / dx1

                if idx2 < (nx2 - 1):
                    idx_glob1 = idx2 + 1 + nx2 * idx1 + nx2 * nx1 * idx0
                    dc0 = d_coef[idx0, idx1, idx2 + 0]
                    dc1 = d_coef[idx0, idx1, idx2 + 1]
                    tmat[idx_glob0, idx_glob0] += 2.0 * dc0 * dc1 / (dc0 + dc1) * dx0 * dx1 / dx2
                    tmat[idx_glob0, idx_glob1] -= 2.0 * dc0 * dc1 / (dc0 + dc1) * dx0 * dx1 / dx2
                if idx1 > 0:
                    idx_glob1 = idx2 - 1 + nx2 * idx1 + nx2 * nx1 * idx0
                    dc0 = d_coef[idx0, idx1, idx2 + 0]
                    dc1 = d_coef[idx0, idx1, idx2 - 1]
                    tmat[idx_glob0, idx_glob0] += 2.0 * dc0 * dc1 / (dc0 + dc1) * dx0 * dx1 / dx2
                    tmat[idx_glob0, idx_glob1] -= 2.0 * dc0 * dc1 / (dc0 + dc1) * dx0 * dx1 / dx2

    return tmat

@jit(nopython=True, fastmath=True)
def compute_linear_system(d_coef, dx0, dx1, dx2, pmin, pmax):
    smat = compute_flow_matrix(d_coef, dx0, dx1, dx2)
    # plt.figure()
    # plt.imshow(smat)
    # plt.show()
    nx0, nx1, nx2 = d_coef.shape
    yvec = np.zeros((smat.shape[0]))
    for idx1 in prange(nx1):
        for idx2 in prange(nx2):
            idx_glob = idx2 + nx2 * idx1 + nx2 * nx1 * 0
            dc = d_coef[0, idx1, idx2]
            smat[idx_glob, idx_glob] += 2.0 * dc * dx1 * dx2 / dx0
            yvec[idx_glob] = 2.0 * dc * dx1 * dx2 / dx0 * pmax
            
            idx_glob = idx2 + nx2 * idx1 + nx2 * nx1 * (nx0 - 1)
            dc = d_coef[nx0 - 1, idx1, idx2]
            smat[idx_glob, idx_glob] += 2.0 * dc * dx1 * dx2 / dx0
            yvec[idx_glob] = 2.0 * dc * dx1 * dx2 / dx0 * pmin
    return (smat, yvec)


def compute_pressure(d_coef, dx0, dx1, dx2, pmin, pmax, niter=100):
    smat, yvec = compute_linear_system(d_coef, dx0, dx1, dx2, pmin, pmax)
    # xvec = compute_soltion_cgd(smat, yvec, niter, pinit.flatten())
    xvec = np.linalg.solve(smat, yvec)
    ### what is the next step
    ### there are several things that you can do
    ### what
    return np.reshape(xvec, d_coef.shape)

@jit(nopython=True, fastmath=True)
def compute_flow(d_coef, dx0, dx1, dx2, pres, pmin, pmax):
    nx0, nx1, nx2 = d_coef.shape
    tmat = np.absolute(compute_flow_matrix(d_coef, dx0, dx1, dx2))
    # print(tmat)
    flow_matrix = np.zeros((nx0 * nx1 * nx2, 6))
    for idx0 in range(nx0):
        for idx1 in prange(nx1):
            for idx2 in prange(nx2):
                idx_glob0 = idx2 + nx2 * idx1 + nx2 * nx1 * idx0

                if idx0 < nx0 - 1:
                    idx_glob1 = idx2 + nx2 * idx1 + nx2 * nx1 * (idx0 + 1)
                    dp = (pres[idx0 + 0, idx1, idx2] - pres[idx0 + 1, idx1, idx2])
                    flow_matrix[idx_glob0, 0] = tmat[idx_glob0, idx_glob1] * dp
                if idx0 > 0:
                    idx_glob1 = idx2 + nx2 * idx1 + nx2 * nx1 * (idx0 - 1)
                    dp = (pres[idx0 + 0, idx1, idx2] - pres[idx0 - 1, idx1, idx2])
                    flow_matrix[idx_glob0, 1] = tmat[idx_glob0, idx_glob1] * dp

                if idx1 < nx1 - 1:
                    idx_glob1 = idx2 + nx2 * (idx1 + 1) + nx2 * nx1 * idx0
                    dp = (pres[idx0, idx1 + 0, idx2] - pres[idx0, idx1 + 1, idx2])
                    flow_matrix[idx_glob0, 2] = tmat[idx_glob0, idx_glob1] * dp
                if idx1 > 0:
                    idx_glob1 = idx2 + nx2 * (idx1 - 1) + nx2 * nx1 * idx0
                    dp = (pres[idx0, idx1 + 0, idx2] - pres[idx0, idx1 - 1, idx2])
                    flow_matrix[idx_glob0, 3] = tmat[idx_glob0, idx_glob1] * dp

                if idx2 < nx2 - 1:
                    idx_glob1 = idx2 + 1 + nx2 * idx1 + nx2 * nx1 * idx0
                    dp = (pres[idx0, idx1, idx2 + 0] - pres[idx0, idx1, idx2 + 1])
                    flow_matrix[idx_glob0, 4] = tmat[idx_glob0, idx_glob1] * dp
                if idx2 > 0:
                    idx_glob1 = idx2 - 1 + nx2 * idx1 + nx2 * nx1 * idx0
                    dp = pres[idx0, idx1, idx2 + 0] - pres[idx0, idx1, idx2 - 1]
                    flow_matrix[idx_glob0, 5] = tmat[idx_glob0, idx_glob1] * dp


    for idx2 in prange(nx2):
        for idx1 in prange(nx1):
            idx_glob = idx2 + nx2 * idx1 + nx2 * nx1 * 0
            dc = d_coef[0, idx1, idx2]
            flow_matrix[idx_glob, 1] = 2.0 * dc * dx1 * dx2 / dx0 * (pres[0, idx1, idx2] - pmax)
            
            idx_glob = idx2 + nx2 * idx1 + nx2 * nx1 * (nx0 - 1)
            dc = d_coef[nx0 - 1, idx1, idx2]
            flow_matrix[idx_glob, 0] = 2.0 * dc * dx1 * dx2 / dx0 * (pres[nx0 - 1, idx1, idx2] - pmin)
            # print('(pres[nx0 - 1, idx1, idx2] - pmin) = ' + str((pres[nx0 - 1, idx1, idx2] - pmin)))
    return flow_matrix

@jit(nopython=True, fastmath=True)
def compute_fractional_flow(swat, soil, pwat, kwat, koil, poil, vr):
    nx0, nx1, nx2 = swat.shape
    frac_flow = np.zeros((nx0, nx1, nx2, 2))
    frac_flow[:, :, :, 0] = compute_fractional_flow_wat(swat, pwat, kwat, koil, poil, vr)
    frac_flow[:, :, :, 1] = 1.0 - frac_flow[:, :, :, 0]
    return frac_flow

@jit(nopython=True, fastmath=True)
def update_saturation(swat, soil, poro, flow_matrix,
                      frac_flow, dt):
    nx0, nx1, nx2 = poro.shape
    swat_updated = swat.copy()
    soil_updated = soil.copy()
    for idx0 in range(nx0):
        for idx1 in prange(nx1):
            for idx2 in prange(nx2):
                idx_glob0 = idx2 + nx2 * idx1 + nx2 * nx1 * idx0
                ### what is the next step?
                if flow_matrix[idx_glob0, 0] > 0:
                    mass_flow = frac_flow[idx0, idx1, idx2, 0] * flow_matrix[idx_glob0, 0]
                    swat_updated[idx0, idx1, idx2] -= dt * mass_flow / poro[idx0, idx1, idx2]
                    mass_flow = frac_flow[idx0, idx1, idx2, 1] * flow_matrix[idx_glob0, 0]
                    soil_updated[idx0, idx1, idx2] -= dt * mass_flow / poro[idx0, idx1, idx2]

                if flow_matrix[idx_glob0, 0] < 0:
                    mass_flow = frac_flow[idx0, idx1, idx2, 0] * flow_matrix[idx_glob0, 0]
                    mass_flow = frac_flow[min(nx0 - 1, idx0 + 1), idx1, idx2, 0] * flow_matrix[idx_glob0, 0]
                    swat_updated[idx0, idx1, idx2] -= dt * mass_flow / poro[idx0, idx1, idx2]
                    mass_flow = frac_flow[min(nx0 - 1, idx0 + 1), idx1, idx2, 1] * flow_matrix[idx_glob0, 0]
                    soil_updated[idx0, idx1, idx2] -= dt * mass_flow / poro[idx0, idx1, idx2]

                if flow_matrix[idx_glob0, 1] > 0:
                    mass_flow = frac_flow[idx0, idx1, idx2, 0] * flow_matrix[idx_glob0, 0]
                    swat_updated[idx0, idx1, idx2] -= dt * mass_flow / poro[idx0, idx1, idx2]
                    mass_flow = frac_flow[idx0, idx1, idx2, 1] * flow_matrix[idx_glob0, 0]
                    soil_updated[idx0, idx1, idx2] -= dt * mass_flow / poro[idx0, idx1, idx2]

                if flow_matrix[idx_glob0, 1] < 0:
                    if idx0 == 0:
                        mass_flow = flow_matrix[idx_glob0, 1]
                        # print('mass_flow = ' + str(mass_flow))
                        # print('mass_flow = ' + str(mass_flow))
                    else:
                        mass_flow = frac_flow[idx0 - 1, idx1, idx2, 0] * flow_matrix[idx_glob0, 1]
                    # print('swat = ' + str(swat_updated[idx0, idx1, idx2]))
                    swat_updated[idx0, idx1, idx2] -= dt * mass_flow / poro[idx0, idx1, idx2]
                    # print('swat_updated = ' + str(swat_updated[idx0, idx1, idx2]))
                    if idx0 == 0:
                        mass_flow = 0.0
                    else:
                        mass_flow = frac_flow[idx0 - 1, idx1, idx2, 1] * flow_matrix[idx_glob0, 1]
                    soil_updated[idx0, idx1, idx2] -= dt * mass_flow / poro[idx0, idx1, idx2]


                if flow_matrix[idx_glob0, 2] > 0:
                    mass_flow = frac_flow[idx0, idx1, idx2, 0] * flow_matrix[idx_glob0, 2]
                    swat_updated[idx0, idx1, idx2] -= dt * mass_flow / poro[idx0, idx1, idx2]
                    mass_flow = frac_flow[idx0, idx1, idx2, 1] * flow_matrix[idx_glob0, 2]
                    soil_updated[idx0, idx1, idx2] -= dt * mass_flow / poro[idx0, idx1, idx2]

                if flow_matrix[idx_glob0, 2] < 0:
                    mass_flow = frac_flow[idx0, idx1 + 1, idx2, 0] * flow_matrix[idx_glob0, 2]
                    swat_updated[idx0, idx1, idx2] -= dt * mass_flow / poro[idx0, idx1, idx2]
                    mass_flow = frac_flow[idx0, idx1 + 1, idx2, 1] * flow_matrix[idx_glob0, 2]
                    soil_updated[idx0, idx1, idx2] -= dt * mass_flow / poro[idx0, idx1, idx2]

                if flow_matrix[idx_glob0, 3] > 0:
                    mass_flow = frac_flow[idx0, idx1, idx2, 0] * flow_matrix[idx_glob0, 3]
                    swat_updated[idx0, idx1, idx2] -= dt * mass_flow / poro[idx0, idx1, idx2]
                    mass_flow = frac_flow[idx0, idx1, idx2, 1] * flow_matrix[idx_glob0, 3]
                    soil_updated[idx0, idx1, idx2] -= dt * mass_flow / poro[idx0, idx1, idx2]

                if flow_matrix[idx_glob0, 3] < 0:
                    mass_flow = frac_flow[idx0, idx1 - 1, idx2, 0] * flow_matrix[idx_glob0, 3]
                    swat_updated[idx0, idx1, idx2] -= dt * mass_flow / poro[idx0, idx1, idx2]
                    mass_flow = frac_flow[idx0, idx1 - 1, idx2, 1] * flow_matrix[idx_glob0, 3]
                    soil_updated[idx0, idx1, idx2] -= dt * mass_flow / poro[idx0, idx1, idx2]


                if flow_matrix[idx_glob0, 4] > 0:
                    mass_flow = frac_flow[idx0, idx1, idx2, 0] * flow_matrix[idx_glob0, 4]
                    swat_updated[idx0, idx1, idx2] -= dt * mass_flow / poro[idx0, idx1, idx2]
                    mass_flow = frac_flow[idx0, idx1, idx2, 1] * flow_matrix[idx_glob0, 4]
                    soil_updated[idx0, idx1, idx2] -= dt * mass_flow / poro[idx0, idx1, idx2]

                if flow_matrix[idx_glob0, 4] < 0:
                    mass_flow = frac_flow[idx0, idx1, idx2 + 1, 0] * flow_matrix[idx_glob0, 4]
                    swat_updated[idx0, idx1, idx2] -= dt * mass_flow / poro[idx0, idx1, idx2]
                    mass_flow = frac_flow[idx0, idx1, idx2 + 1, 1] * flow_matrix[idx_glob0, 4]
                    soil_updated[idx0, idx1, idx2] -= dt * mass_flow / poro[idx0, idx1, idx2]

                if flow_matrix[idx_glob0, 5] > 0:
                    mass_flow = frac_flow[idx0, idx1, idx2, 0] * flow_matrix[idx_glob0, 5]
                    swat_updated[idx0, idx1, idx2] -= dt * mass_flow / poro[idx0, idx1, idx2]
                    mass_flow = frac_flow[idx0, idx1, idx2, 1] * flow_matrix[idx_glob0, 5]
                    soil_updated[idx0, idx1, idx2] -= dt * mass_flow / poro[idx0, idx1, idx2]

                if flow_matrix[idx_glob0, 5] < 0:
                    mass_flow = frac_flow[idx0, idx1, idx2 - 1, 0] * flow_matrix[idx_glob0, 5]
                    swat_updated[idx0, idx1, idx2] -= dt * mass_flow / poro[idx0, idx1, idx2]
                    mass_flow = frac_flow[idx0, idx1, idx2 - 1, 1] * flow_matrix[idx_glob0, 5]
                    soil_updated[idx0, idx1, idx2] -= dt * mass_flow / poro[idx0, idx1, idx2]

    return (swat_updated, soil_updated)


def compute_single_step(swat, soil, perm, poro,
                        dx0, dx1, dx2, dt,
                        pwat, kwat, poil, koil, vr,
                        pmin=0.0, pmax=1.0):
    ### the first thing to do is calculation of pressure
    abs_mob = compute_absolute_mobility(swat, pwat, kwat,
                                        soil, poil, koil,
                                        vr)
    diff_coef = abs_mob * perm
    pres = compute_pressure(diff_coef, dx0, dx1, dx2, pmin, pmax)
    flow_matrix = compute_flow(diff_coef, dx0, dx1, dx2, pres, pmin, pmax)
    # print(pres)
    # print(flow_matrix)
    frac_flow = compute_fractional_flow(swat, soil, pwat, kwat, koil, poil, vr)
    swat_update, soil_update = update_saturation(swat, soil, poro, flow_matrix, frac_flow, dt)
    return (pres, swat_update, soil_update)


def compute_solution(perm, poro,
                     dx0, dx1, dx2, tfinal, niter,
                     pwat, kwat, poil, koil, vr,
                     pmin=0.0, pmax=1.0):
    nx0, nx1, nx2 = poro.shape
    dt = tfinal / niter
    swat = np.zeros((nx0, nx1, nx2))
    soil = np.ones((nx0, nx1, nx2))

    swat_res = np.zeros((nx0, nx1, nx2, niter))
    soil_res = np.ones((nx0, nx1, nx2, niter))
    pres_res = np.ones((nx0, nx1, nx2, niter))
    for kiter in range(niter):
        pres, swat, soil = compute_single_step(swat, soil, perm, poro,
                                               dx0, dx1, dx2, dt,
                                               pwat, kwat, poil, koil, vr,
                                               pmin=0.0, pmax=1.0)
        pres_res[:, :, :, kiter] = pres
        swat_res[:, :, :, kiter] = swat
        soil_res[:, :, :, kiter] = soil
    return (pres_res, swat_res, soil_res)

# def main():
#     print('inside the main function')
#     # poro = np.load('SPE10MODEL2_PHI.npy')
#     # perm = np.load('SPE10MODEL2_PERM.npy')
#     # print('poro.shape = ' + str(poro.shape))
#     # print('perm.shape = ' + str(perm.shape))

#     pwat = 2.0
#     poil = 4.0
#     vr = 0.3
#     kwat = 1.0
#     koil = 0.3

#     # pwat = 1.0
#     # poil = 1.0
#     # vr = 1.0
#     # kwat = 1.0
#     # koil = 1.0
#     pmin = 0.0
#     pmax = 1.0
#     nx0 = 50
#     nx1 = 30
#     nx2 = 1
#     dx0 = 1.0 / nx0
#     dx1 = 1.0 / nx1
#     dx2 = 1.0 / nx2
#     dt = 0.5e-1
#     niter = 100

#     poro = 0.1 + np.zeros((nx0, nx1, nx2))
#     perm = np.ones((nx0, nx1, nx2))
#     swat = np.zeros((nx0, nx1, nx2))
#     soil = np.ones((nx0, nx1, nx2))


#     pres, swat, soil = compute_solution(perm, poro,
#                                         dx0, dx1, dx2, dt * niter, niter,
#                                         pwat, kwat, poil, koil, vr,
#                                         pmin=0.0, pmax=1.0)

#     # plt.figure()
#     # plt.imshow(pres)
#     # plt.show()
#     plt.figure()
#     plt.imshow(swat)
#     plt.show()
#     # plt.figure()
#     # plt.imshow(soil)
#     # plt.show()

#     plt.figure()
#     plt.scatter(np.linspace(0.0, 1.0, nx0), swat[:, 0, 0])
#     plt.grid()
#     plt.show()

#     plt.figure()
#     plt.scatter(np.linspace(0.0, 1.0, nx0), pres[:, 0, 0])
#     plt.grid()
#     plt.show()



#     return 0

# main()



















